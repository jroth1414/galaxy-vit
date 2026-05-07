"""T3.4 — Dirichlet-Multinomial loss tests.

Three layers of assertion:

1. **Math correctness** — single-question NLL agrees with
   :class:`torch.distributions.DirichletMultinomial.log_prob` (modulo the
   data-only log-multinomial-coefficient constant) within fp32 tolerance.
2. **Masking semantics** — ``valid`` zeros contributions from masked
   galaxies; questions with all-False valid contribute exactly 0; the
   per-question mean is over the *valid* subset, not the full batch.
3. **Overfit-100 gut check** — a tiny MLP + DirichletMultinomialHead
   trained on 100 fixed (feature, vote-count) pairs reaches per-answer
   fraction MAE < 5% in < 200 Adam steps. This is the DEVPLAN T3.4
   acceptance gate.

If this file fails, the loss math is wrong (or the head is); it is NOT
a hyperparameter-tuning issue.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("galaxy_datasets")

from galaxy_vit.data.schema import question_index_groups  # noqa: E402
from galaxy_vit.losses.dirichlet_mn import (  # noqa: E402
    dirichlet_multinomial_nll,
    dirichlet_multinomial_nll_per_question,
    expected_fractions,
)
from galaxy_vit.models.dirichlet_head import build_dirichlet_head  # noqa: E402

NUM_ANSWERS = 34


# ---------------------------------------------------------------------------
# Layer 1 — math correctness
# ---------------------------------------------------------------------------


def test_T3_4_per_question_nll_matches_scipy() -> None:
    """Cross-check our DM NLL against scipy.stats.dirichlet_multinomial.logpmf.

    scipy includes the data-only ``log(N! / prod(c_k!))`` log-multinomial
    coefficient that we omit (it doesn't affect gradients on alpha).
    Subtracting that constant from scipy's ``-logpmf`` gives the
    quantity our function returns; equality within fp32 round-off
    proves the gammaln expansion + the sign on the normalization term
    are both right.
    """
    pytest.importorskip("scipy")
    import numpy as np
    from scipy.stats import dirichlet_multinomial as scipy_dm

    torch.manual_seed(0)
    K = 5
    B = 7
    # Concentrations strictly above 1 (matches the head's softplus + 1 floor).
    alpha = torch.rand(B, K) * 4.0 + 1.1
    # Build counts via per-row Dirichlet -> Multinomial sampling.
    totals = torch.tensor([15, 22, 8, 30, 6, 11, 19], dtype=torch.long)
    counts = torch.zeros(B, K, dtype=torch.long)
    for i in range(B):
        p_i = torch.distributions.Dirichlet(alpha[i]).sample()
        counts[i] = torch.distributions.Multinomial(
            int(totals[i]), p_i
        ).sample().to(torch.long)

    # Our NLL.
    ours = dirichlet_multinomial_nll_per_question(alpha, counts)

    # scipy reference: per-row logpmf, negated to get NLL.
    alpha_np = alpha.numpy().astype(np.float64)
    counts_np = counts.numpy().astype(np.int64)
    ref = np.empty(B, dtype=np.float64)
    for i in range(B):
        ref[i] = -scipy_dm.logpmf(
            counts_np[i], alpha=alpha_np[i], n=int(counts_np[i].sum())
        )

    # The log-multinomial coefficient ``log[ N! / prod(c_k!) ]`` that scipy
    # includes but we drop. We add it BACK to scipy's NLL (= -logpmf) to
    # remove that constant from the comparison: scipy's NLL is
    # -log_mn_coef + alpha_dependent_NLL, so adding log_mn_coef leaves the
    # alpha-dependent piece, which is what our function returns.
    log_mn_coef = (
        torch.special.gammaln(counts.sum(dim=-1).float() + 1.0)
        - torch.special.gammaln(counts.float() + 1.0).sum(dim=-1)
    ).numpy()
    ref_alpha_only = ref + log_mn_coef

    diff = ours.numpy() - ref_alpha_only
    assert np.allclose(ours.numpy(), ref_alpha_only, atol=1e-3, rtol=1e-4), (
        f"\nours          = {ours.tolist()}\n"
        f"ref+log_mn    = {ref_alpha_only.tolist()}\n"
        f"diff          = {diff.tolist()}"
    )


def test_T3_4_nll_is_nonnegative_for_typical_inputs() -> None:
    """DM NLL (without the log-multinomial constant) is non-negative for
    counts summing to >= 1 and alpha > 1."""
    torch.manual_seed(1)
    alpha = torch.rand(20, 4) * 5.0 + 1.5
    counts = torch.randint(0, 12, (20, 4))
    nll = dirichlet_multinomial_nll_per_question(alpha, counts)
    assert torch.isfinite(nll).all()
    assert (nll >= -1e-3).all(), f"min NLL = {nll.min()}"


def test_T3_4_perfect_concentration_minimizes_nll() -> None:
    """If alpha matches the empirical fractions of a high-count sample, NLL is small.

    Build counts c, set alpha proportional to c (with sum at least 100x
    the count-total), and verify NLL is much smaller than for a uniform
    alpha of the same magnitude.
    """
    counts = torch.tensor([[40, 30, 20, 10]], dtype=torch.long)
    sharp_alpha = (counts.float() / counts.sum() * 1000.0) + 1.0
    flat_alpha = torch.full_like(sharp_alpha, 251.0)  # same total
    sharp_nll = dirichlet_multinomial_nll_per_question(sharp_alpha, counts).item()
    flat_nll = dirichlet_multinomial_nll_per_question(flat_alpha, counts).item()
    assert sharp_nll < flat_nll, (
        f"sharp alpha should fit better than flat: sharp={sharp_nll}, flat={flat_nll}"
    )


# ---------------------------------------------------------------------------
# Layer 2 — masking semantics
# ---------------------------------------------------------------------------


def _alpha_counts_valid_for_full_head(
    *, B: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helper: build (alpha, counts, valid) tensors for the 34-answer head."""
    torch.manual_seed(seed)
    alpha = torch.rand(B, NUM_ANSWERS) * 3.0 + 1.1
    counts = torch.randint(0, 8, (B, NUM_ANSWERS))
    valid = torch.ones(B, len(question_index_groups()), dtype=torch.bool)
    return alpha, counts, valid


def test_T3_4_mask_zero_valid_question_contributes_zero_loss() -> None:
    """A question with all-False valid contributes exactly 0 to the total loss."""
    alpha, counts, valid = _alpha_counts_valid_for_full_head(B=4)
    groups = question_index_groups()

    # Loss with full validity:
    loss_full = dirichlet_multinomial_nll(alpha, counts, valid, question_groups=groups)

    # Mask the second question (disk-edge-on) entirely off:
    valid_off = valid.clone()
    valid_off[:, 1] = False
    loss_minus_one = dirichlet_multinomial_nll(
        alpha, counts, valid_off, question_groups=groups
    )

    # Also compute that single question's contribution directly.
    _name, start, end = groups[1]
    single_q_loss = dirichlet_multinomial_nll_per_question(
        alpha[:, start:end], counts[:, start:end]
    ).mean()

    # Identity: full = minus_one + single_q (within fp32 tolerance).
    delta = float((loss_full - loss_minus_one).item())
    assert math.isclose(delta, float(single_q_loss.item()), abs_tol=1e-4), (
        f"masking accounting wrong: delta={delta}, single_q={single_q_loss.item()}"
    )


def test_T3_4_per_question_mean_is_over_valid_subset_only() -> None:
    """Masking out half the batch's galaxies for a question changes that
    question's contribution to the *mean over remaining valid galaxies*,
    not the mean over the full batch.
    """
    alpha, counts, valid = _alpha_counts_valid_for_full_head(B=8)
    groups = question_index_groups()
    _name, start, end = groups[3]  # bar

    # Full-validity per-question mean.
    full_mean = dirichlet_multinomial_nll_per_question(
        alpha[:, start:end], counts[:, start:end]
    ).mean().item()

    # Mask out the first 4 galaxies for the bar question.
    valid_partial = valid.clone()
    valid_partial[:4, 3] = False
    partial_loss = dirichlet_multinomial_nll(
        alpha, counts, valid_partial, question_groups=groups
    ).item()
    full_loss = dirichlet_multinomial_nll(
        alpha, counts, valid, question_groups=groups
    ).item()

    # The remaining 4 galaxies' bar NLL averaged should equal partial - (full - full_bar_mean).
    remaining_bar_mean = dirichlet_multinomial_nll_per_question(
        alpha[4:, start:end], counts[4:, start:end]
    ).mean().item()
    expected_partial_loss = (full_loss - full_mean) + remaining_bar_mean
    assert math.isclose(partial_loss, expected_partial_loss, abs_tol=1e-4), (
        f"partial mean does not match the valid-subset mean: "
        f"partial={partial_loss}, expected={expected_partial_loss}"
    )


def test_T3_4_all_questions_invalid_returns_zero() -> None:
    """If no question has any valid sample in the batch, loss is exactly 0."""
    alpha, counts, valid = _alpha_counts_valid_for_full_head(B=4)
    groups = question_index_groups()
    valid_zero = torch.zeros_like(valid)
    loss = dirichlet_multinomial_nll(
        alpha, counts, valid_zero, question_groups=groups
    )
    assert loss.item() == 0.0


def test_T3_4_shape_validation_rejects_misshapen_tensors() -> None:
    """Wrong-shape inputs raise ValueError before we reach the loss math."""
    groups = question_index_groups()
    alpha = torch.rand(3, NUM_ANSWERS) + 1.1
    counts = torch.randint(0, 5, (3, NUM_ANSWERS))
    valid = torch.ones(3, len(groups), dtype=torch.bool)

    # alpha vs counts shape mismatch.
    with pytest.raises(ValueError, match="alpha shape"):
        dirichlet_multinomial_nll(
            alpha, counts[:2], valid, question_groups=groups
        )
    # batch dim mismatch.
    with pytest.raises(ValueError, match="batch dim mismatch"):
        dirichlet_multinomial_nll(
            alpha, counts, valid[:2], question_groups=groups
        )
    # num_questions mismatch.
    with pytest.raises(ValueError, match="num_questions mismatch"):
        dirichlet_multinomial_nll(
            alpha, counts, valid[:, :3], question_groups=groups
        )


def test_T3_4_expected_fractions_normalize_per_question() -> None:
    """expected_fractions makes each question slice sum to 1."""
    alpha, _, _ = _alpha_counts_valid_for_full_head(B=4)
    groups = question_index_groups()
    fracs = expected_fractions(alpha, question_groups=groups)
    for _name, start, end in groups:
        slice_sum = fracs[:, start:end].sum(dim=-1)
        assert torch.allclose(slice_sum, torch.ones_like(slice_sum), atol=1e-5)


# ---------------------------------------------------------------------------
# Layer 3 — overfit-100 gut check (DEVPLAN T3.4 acceptance)
# ---------------------------------------------------------------------------


def test_T3_4_overfit_100_samples_reaches_mae_under_5pct() -> None:
    """A tiny MLP + DirichletMultinomialHead overfits 100 fixed samples to MAE<5%.

    This is the DEVPLAN T3.4 acceptance gate. If it fails, the loss math
    is broken — not your hyperparameters. The MLP is intentionally tiny
    so this runs in ~10-20 s on CPU.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(42)
    groups = question_index_groups()
    n_samples = 100
    feat_dim = 8

    # ---- Synthesize a deterministic dataset --------------------------------
    # Each "image" is a fixed 8-dim feature vector; each galaxy has well-above-
    # min-votes counts on every question (so masking is irrelevant in this
    # test — we are testing the loss math itself).
    features = torch.randn(n_samples, feat_dim)
    # Per-galaxy "true" alpha: random concentration with floor 1.5, sampled once.
    true_alpha = torch.rand(n_samples, NUM_ANSWERS) * 4.0 + 1.5
    # Counts: sample from DM(N=20, true_alpha) per question slice.
    counts = torch.zeros(n_samples, NUM_ANSWERS, dtype=torch.float32)
    for _name, start, end in groups:
        for i in range(n_samples):
            p_i = torch.distributions.Dirichlet(true_alpha[i, start:end]).sample()
            counts[i, start:end] = torch.distributions.Multinomial(
                20, p_i
            ).sample()
    valid = torch.ones(n_samples, len(groups), dtype=torch.bool)

    # ---- Tiny MLP encoder + Dirichlet head --------------------------------
    class _TinyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l1 = nn.Linear(feat_dim, 128)
            self.l2 = nn.Linear(128, 128)
            self.head = build_dirichlet_head(128, num_answers=NUM_ANSWERS)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = F.relu(self.l1(x))
            h = F.relu(self.l2(h))
            return self.head(h)  # type: ignore[no-any-return]

    model = _TinyMLP()
    # LR + width tuned to overfit 100 fixed samples in <= 200 Adam steps
    # without diverging on the bulge-size (K=5) question's gammaln gradient.
    # Width 128 gives the head enough capacity to memorize the 100 distinct
    # (feature -> alpha) mappings within the budget.
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-2)

    # ---- Train for 200 steps ----------------------------------------------
    n_steps = 200
    final_loss = float("inf")
    for _step in range(n_steps):
        alpha = model(features)
        loss = dirichlet_multinomial_nll(alpha, counts, valid, question_groups=groups)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    # ---- MAE on per-answer expected fractions -----------------------------
    with torch.no_grad():
        alpha = model(features)
        pred_fracs = expected_fractions(alpha, question_groups=groups)
        # Ground-truth fractions: empirical c_k / N per question.
        true_fracs = torch.zeros_like(counts)
        for _name, start, end in groups:
            slice_counts = counts[:, start:end]
            denom = slice_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
            true_fracs[:, start:end] = slice_counts / denom
        mae = (pred_fracs - true_fracs).abs().mean().item()

    assert mae < 0.05, (
        f"overfit-100 MAE = {mae:.4f} (>= 5% gate); final_loss={final_loss:.4f}. "
        "Loss math is wrong."
    )
