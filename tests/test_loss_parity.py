"""T3.5 — Loss parity vs Zoobot 2.0.

DEVPLAN T3.5 acceptance: *"loss within 1% of Zoobot's native ``define_model``
+ loss on identical inputs."*

Zoobot's ``get_dirichlet_neg_log_prob`` (in ``zoobot.pytorch.training.losses``)
computes the **full** Dirichlet-Multinomial NLL, including the data-only
log-multinomial coefficient ``log[ N! / prod(c_k!) ]``. Ours
deliberately omits that constant — it doesn't affect gradients on alpha
and keeps the loss numerically smaller. Our + log_mn_coef therefore
equals zoobot's per-galaxy NLL up to fp32 round-off; that is the
quantity DEVPLAN's 1% tolerance is checked against.

Also asserts gradient parity: for any (alpha, counts), the gradient of
ours w.r.t. alpha equals the gradient of zoobot's w.r.t. alpha exactly
(the constant-shift difference has zero gradient on alpha). This is the
functionally-meaningful equivalence: an optimizer running on either
loss takes the same step.

Skipped when ``zoobot`` is not installed (it ships outside the project's
declared extras to keep the install footprint tractable; the parity
test is only meaningful with the upstream library present anyway).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
zoobot_losses = pytest.importorskip(
    "zoobot.pytorch.training.losses",
    reason="install zoobot to run the T3.5 loss-parity test against the upstream NLL",
)

from galaxy_vit.losses.dirichlet_mn import (  # noqa: E402
    dirichlet_multinomial_nll_per_question,
)


def _build_synthetic_batch(
    *, B: int, K: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct (alpha, counts) tensors for a single question.

    alpha is sampled in [1.1, 5.1] (matches the head's softplus + 1 floor);
    counts are drawn per row from DM(N=20, alpha) so the comparison is on
    the kind of (alpha, counts) pair the trainer would actually feed in.
    """
    torch.manual_seed(seed)
    alpha = torch.rand(B, K) * 4.0 + 1.1
    counts = torch.zeros(B, K, dtype=torch.long)
    for i in range(B):
        p_i = torch.distributions.Dirichlet(alpha[i]).sample()
        counts[i] = torch.distributions.Multinomial(20, p_i).sample().to(torch.long)
    return alpha, counts


def _log_multinomial_coef(counts: torch.Tensor) -> torch.Tensor:
    """log[ N! / prod(c_k!) ] per row."""
    return (
        torch.special.gammaln(counts.sum(dim=-1).float() + 1.0)
        - torch.special.gammaln(counts.float() + 1.0).sum(dim=-1)
    )


def test_T3_5_per_galaxy_nll_matches_zoobot_within_1pct() -> None:
    """Our NLL equals Zoobot's per-galaxy NLL + log_mn_coef within 1% relative.

    Zoobot's ``get_dirichlet_neg_log_prob`` returns the FULL DM NLL
    (including the log-multinomial coefficient ``log[ N! / prod(c_k!) ]``).
    Ours omits that constant — it doesn't affect gradients on alpha.
    So the math relationship is ``ours == zoobot + log_mn_coef`` per row.
    DEVPLAN's 1% tolerance is checked against ``zoobot + log_mn_coef``,
    which is exactly the alpha-dependent NLL the optimizer sees.
    """
    K_values = (3, 5, 6)  # cover GZ DESI's full range of per-question answer counts
    for K in K_values:
        alpha, counts = _build_synthetic_batch(B=12, K=K, seed=K)
        ours = dirichlet_multinomial_nll_per_question(alpha, counts)
        zoobot_nll = zoobot_losses.get_dirichlet_neg_log_prob(alpha, counts)
        log_mn_coef = _log_multinomial_coef(counts)

        zoobot_alpha_only = zoobot_nll + log_mn_coef
        # Relative error per galaxy. Use the larger of the two
        # magnitudes as the denominator; both quantities can drift
        # near zero on tightly-fit Dirichlet samples.
        denom = torch.maximum(ours.abs(), zoobot_alpha_only.abs()).clamp_min(1e-6)
        rel_err = (ours - zoobot_alpha_only).abs() / denom
        max_rel = float(rel_err.max().item())
        assert max_rel < 0.01, (
            f"K={K}: max relative error {max_rel:.5f} exceeds 1% gate.\n"
            f"  ours              = {ours.tolist()}\n"
            f"  zoobot+log_mn_coef = {zoobot_alpha_only.tolist()}\n"
            f"  rel_err            = {rel_err.tolist()}"
        )


def test_T3_5_gradient_parity_on_alpha() -> None:
    """The gradient w.r.t. alpha is identical for our NLL and Zoobot's NLL.

    The two losses differ by a constant w.r.t. alpha (the log-multinomial
    coefficient depends only on counts), so the gradient on alpha must
    match exactly — modulo fp32 round-off. This is the optimization-
    equivalence statement: any optimizer is stepping the same direction
    under either loss.
    """
    for K in (3, 5, 6):
        alpha_a, counts = _build_synthetic_batch(B=8, K=K, seed=100 + K)
        alpha_a = alpha_a.clone().requires_grad_(True)
        alpha_b = alpha_a.detach().clone().requires_grad_(True)

        ours = dirichlet_multinomial_nll_per_question(alpha_a, counts).sum()
        ours.backward()

        zoobot_nll = zoobot_losses.get_dirichlet_neg_log_prob(alpha_b, counts).sum()
        zoobot_nll.backward()

        assert alpha_a.grad is not None and alpha_b.grad is not None
        diff = (alpha_a.grad - alpha_b.grad).abs()
        denom = alpha_b.grad.abs().clamp_min(1e-6)
        rel = (diff / denom).max().item()
        assert rel < 1e-3, (
            f"K={K}: alpha-gradient disagrees with zoobot by {rel:.6f} "
            f"(absolute max diff = {diff.max().item():.6e})"
        )


def test_T3_5_constant_offset_is_log_multinomial_coefficient() -> None:
    """Document the offset between our NLL and Zoobot's: it's exactly log_mn_coef.

    This is the documentation of the calling convention difference. If
    a future zoobot release changes its NLL convention (e.g., starts
    omitting the log_mn_coef itself, or computes it differently), this
    test will fail and force a review.
    """
    alpha, counts = _build_synthetic_batch(B=20, K=5, seed=42)
    ours = dirichlet_multinomial_nll_per_question(alpha, counts)
    zoobot_nll = zoobot_losses.get_dirichlet_neg_log_prob(alpha, counts)
    observed_offset = ours - zoobot_nll
    expected_offset = _log_multinomial_coef(counts)
    # Per-galaxy offsets agree to fp32 tolerance.
    assert torch.allclose(
        observed_offset, expected_offset, atol=1e-3, rtol=1e-4
    ), (
        f"\nobserved offset = {observed_offset.tolist()}\n"
        f"expected (log_mn_coef) = {expected_offset.tolist()}\n"
        f"diff = {(observed_offset - expected_offset).tolist()}"
    )


def test_T3_5_zero_count_question_matches_zoobot() -> None:
    """Edge case: when a galaxy has zero votes on a question (N=0), both
    losses degenerate consistently.

    With counts all zero: log_mn_coef = lgamma(1) - sum_k lgamma(1) = 0,
    so ours and zoobot's per-galaxy NLLs should agree exactly (no
    constant-offset to track).
    """
    K = 4
    B = 6
    torch.manual_seed(7)
    alpha = torch.rand(B, K) * 3.0 + 1.1
    counts = torch.zeros(B, K, dtype=torch.long)

    ours = dirichlet_multinomial_nll_per_question(alpha, counts)
    zoobot_nll = zoobot_losses.get_dirichlet_neg_log_prob(alpha, counts)
    assert torch.allclose(ours, zoobot_nll, atol=1e-4, rtol=1e-4), (
        f"zero-count NLL mismatch:\n  ours={ours.tolist()}\n  zoobot={zoobot_nll.tolist()}"
    )
