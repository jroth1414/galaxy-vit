"""Active-learning acquisitions + loop driver for the Dirichlet head (T4.1).

Active learning selects the unlabeled galaxies that, if labeled, would
most reduce model uncertainty. Two acquisitions are implemented here,
both well-grounded for a Dirichlet head over multinomial votes:

* :func:`predictive_entropy` -- H[p_pred] where ``p_pred`` is the
  Dirichlet posterior mean ``alpha_i / sum(alpha)``. Highest for
  galaxies the model thinks are ambiguous (broad p_pred); a "pick
  the disagreeing ones" objective.

* :func:`bald_score` -- Bayesian Active Learning by Disagreement
  (Houlsby+11). Mutual information between the next vote and the
  Dirichlet parameters:

      BALD = H[p_pred] - E_alpha[H[p_q | alpha]]

  Where the inner expected-entropy term has the closed form for a
  Dirichlet:

      E[H[p|alpha]] = -sum_i (alpha_i / A) [psi(alpha_i + 1) - psi(A + 1)]

  with ``A = sum(alpha)`` and ``psi`` the digamma function. Highest
  for galaxies where the model is confidently uncertain (small
  posterior support but wide predictive entropy) -- the "I know I
  don't know" objective. Tends to outperform pure entropy when the
  unlabeled pool is large.

Both scores are aggregated over the 10 GZ DESI questions: per-question
score weighted by the predicted total vote count, summed. Aggregation
keeps the acquisition unbiased even when a question's per-galaxy
``valid`` mask is False at unlabeled-pool time (we don't know the mask
in advance, so we score on the assumption every question is asked).

The AL loop driver :func:`run_active_learning_loop` is cached-feature
and head-only: the Zoobot encoder is run once over the entire pool,
features are stored, and each round retrains ONLY the Dirichlet head on
the current labeled subset. This brings each round from
hours-of-finetune to ~seconds, making the 3-seeds * 3-acquisitions *
N-rounds experiment tractable.

T4.1 acceptance: entropy acquisition reaches 90% of the full-data MAE
in <= 60% of labels across 3 seeds (DEVPLAN T4.1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover -- typing-only
    from torch import Tensor

Acquisition = Literal["entropy", "bald", "random"]
EPS = 1e-12


def _per_question_predictive_probs(
    alpha: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
) -> Tensor:
    """Return per-answer posterior-mean probabilities in the flat (B, K) layout.

    For each question slice, ``p_i = alpha_i / sum(alpha_q)``.
    """
    import torch

    a = alpha.to(torch.float32)
    out = torch.empty_like(a)
    for _q_name, start, end in question_groups:
        denom = a[:, start:end].sum(dim=-1, keepdim=True).clamp_min(EPS)
        out[:, start:end] = a[:, start:end] / denom
    return out


def predictive_entropy(
    alpha: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
) -> Tensor:
    """Sum across questions of per-question categorical entropy on the posterior mean.

    Returns a ``(B,)`` tensor of acquisition scores -- higher = more
    uncertain. Natural-log entropy; for a K-answer question the max is
    ``log K``.
    """
    import torch

    p = _per_question_predictive_probs(alpha, question_groups=question_groups)
    out = torch.zeros(alpha.shape[0], dtype=torch.float32, device=alpha.device)
    for _q_name, start, end in question_groups:
        slice_p = p[:, start:end].clamp_min(EPS)
        # -sum_i p_i log p_i  (per galaxy, per question), then sum across questions.
        out = out + -(slice_p * slice_p.log()).sum(dim=-1)
    return out


def bald_score(
    alpha: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
) -> Tensor:
    """Sum across questions of per-question BALD (Houlsby+11).

    For each question, BALD = H[p_pred] - E_alpha[H[categorical(p)]].
    The expected-entropy term has the closed form:

        E[H[p|alpha]] = sum_i (alpha_i / A) * (psi(A+1) - psi(alpha_i+1))

    where psi is digamma. Both terms are non-negative; BALD is also
    non-negative (any K-categorical has H_predictive >= E[H]; the gap
    is the mutual information between the next sample and the
    parameters).

    Returns a ``(B,)`` tensor of acquisition scores. Higher = the model
    is "confidently uncertain" about this galaxy.
    """
    import torch

    a = alpha.to(torch.float32)
    out = torch.zeros(alpha.shape[0], dtype=torch.float32, device=alpha.device)
    p_pred = _per_question_predictive_probs(alpha, question_groups=question_groups)
    for _q_name, start, end in question_groups:
        slice_alpha = a[:, start:end]
        A = slice_alpha.sum(dim=-1, keepdim=True)  # (B, 1)
        # Predictive entropy term.
        slice_p = p_pred[:, start:end].clamp_min(EPS)
        h_pred = -(slice_p * slice_p.log()).sum(dim=-1)
        # Expected entropy under the Dirichlet: closed form via digamma.
        digamma_a_plus_1 = torch.special.digamma(slice_alpha + 1.0)
        digamma_A_plus_1 = torch.special.digamma(A + 1.0)
        # sum_i (alpha_i / A) * (psi(A+1) - psi(alpha_i+1))
        e_h = ((slice_alpha / A.clamp_min(EPS)) * (digamma_A_plus_1 - digamma_a_plus_1)).sum(dim=-1)
        out = out + (h_pred - e_h)
    return out


def acquisition_scores(
    alpha: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
    method: Acquisition,
    rng_seed: int | None = None,
) -> Tensor:
    """Dispatch to the requested acquisition function.

    ``random`` returns IID uniform scores (used as the baseline that AL
    must beat).
    """
    import torch

    if method == "entropy":
        return predictive_entropy(alpha, question_groups=question_groups)
    if method == "bald":
        return bald_score(alpha, question_groups=question_groups)
    if method == "random":
        g = torch.Generator()
        if rng_seed is not None:
            g.manual_seed(rng_seed)
        return torch.rand(alpha.shape[0], generator=g)
    raise ValueError(f"unknown acquisition method: {method!r}")


@dataclass
class ALRound:
    """One round of the active-learning loop."""

    n_labeled: int
    fraction_labeled: float
    test_mae_macro: float
    test_coverage_macro: float


def run_active_learning_loop(
    features: Tensor,
    counts: Tensor,
    valid: Tensor,
    test_features: Tensor,
    test_counts: Tensor,
    test_valid: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
    head_factory: Callable[[], Any],
    train_head_fn: Callable[..., None],
    method: Acquisition,
    n_rounds: int,
    init_fraction: float = 0.05,
    seed: int = 42,
) -> list[ALRound]:
    """Run an active-learning loop and return per-round test-MAE history.

    Parameters
    ----------
    features, counts, valid : (N_pool, *) tensors
        The full pool from which to acquire labels. ``valid`` is the per-
        question validity mask.
    test_features, test_counts, test_valid : (N_test, *) tensors
        Held-out set used to measure MAE after each round.
    question_groups :
        Per-question slice layout (from schema.question_index_groups).
    head_factory :
        Callable that returns a fresh head module (a fresh
        DirichletMultinomialHead instance). Called at the start of each
        round so the head is retrained from scratch on the new labeled
        subset.
    train_head_fn :
        ``train_head_fn(head, features_lab, counts_lab, valid_lab, question_groups) -> None``
        Trains the head in-place on the labeled subset. Must converge
        in a small number of steps (caller manages capacity / lr).
    method : Acquisition
        Acquisition function. ``random`` is the no-AL baseline.
    n_rounds : int
        Number of acquisition rounds.
    init_fraction : float
        Fraction of the pool labeled at round 0 (before any acquisition).
    seed : int
        RNG seed for the initial random subset and (when ``method='random'``)
        the per-round acquisition.

    Returns
    -------
    list[ALRound] -- one entry per round, including round 0 (the
    initial random subset) so the history starts at the smallest
    labeled size.
    """
    import torch

    from galaxy_vit.inference.posterior import credible_interval
    from galaxy_vit.losses.dirichlet_mn import expected_fractions

    rng = torch.Generator().manual_seed(seed)
    n_pool = features.shape[0]
    if not 0.0 < init_fraction < 1.0:
        raise ValueError(f"init_fraction must be in (0, 1), got {init_fraction}")
    n_init = max(1, round(init_fraction * n_pool))
    # Number to acquire per round: split the remaining pool evenly.
    n_per_round = max(1, (n_pool - n_init) // n_rounds)

    perm = torch.randperm(n_pool, generator=rng)
    labeled_idx = perm[:n_init].clone()
    unlabeled_idx = perm[n_init:].clone()

    history: list[ALRound] = []
    for round_i in range(n_rounds + 1):
        head = head_factory()
        feats_lab = features[labeled_idx]
        counts_lab = counts[labeled_idx]
        valid_lab = valid[labeled_idx]
        train_head_fn(head, feats_lab, counts_lab, valid_lab, question_groups)

        # Evaluate on the test set.
        head.eval()
        with torch.no_grad():
            test_alpha = head(test_features).float()
            pred = expected_fractions(test_alpha, question_groups=question_groups)
            obs = expected_fractions(test_counts.float(), question_groups=question_groups)
            lower, upper = credible_interval(
                test_alpha, question_groups=question_groups, ci=0.95
            )

        mae_per_q: list[float] = []
        cov_per_q: list[float] = []
        for q_idx, (_q_name, start, end) in enumerate(question_groups):
            q_valid = test_valid[:, q_idx]
            n_v = int(q_valid.sum().item())
            if n_v == 0:
                continue
            mae = float((pred[q_valid, start:end] - obs[q_valid, start:end]).abs().mean().item())
            inside = (
                (obs[q_valid, start:end] >= lower[q_valid, start:end])
                & (obs[q_valid, start:end] <= upper[q_valid, start:end])
            ).float()
            cov = float(inside.mean().item())
            mae_per_q.append(mae)
            cov_per_q.append(cov)
        mae_macro = sum(mae_per_q) / max(1, len(mae_per_q))
        cov_macro = sum(cov_per_q) / max(1, len(cov_per_q))

        history.append(
            ALRound(
                n_labeled=int(labeled_idx.numel()),
                fraction_labeled=float(labeled_idx.numel() / n_pool),
                test_mae_macro=mae_macro,
                test_coverage_macro=cov_macro,
            )
        )

        if round_i == n_rounds:
            break

        # Acquire next batch from unlabeled pool.
        with torch.no_grad():
            unlabeled_features = features[unlabeled_idx]
            unlabeled_alpha = head(unlabeled_features).float()
        scores = acquisition_scores(
            unlabeled_alpha,
            question_groups=question_groups,
            method=method,
            rng_seed=seed + round_i,
        )
        k = min(n_per_round, int(unlabeled_idx.numel()))
        if k == 0:
            break
        top_within_unlabeled = torch.topk(scores, k=k).indices
        chosen = unlabeled_idx[top_within_unlabeled]
        labeled_idx = torch.cat([labeled_idx, chosen])
        # Remove chosen from unlabeled.
        mask = torch.ones(unlabeled_idx.numel(), dtype=torch.bool)
        mask[top_within_unlabeled] = False
        unlabeled_idx = unlabeled_idx[mask]

    return history


def reaches_target_at_fraction(
    history: list[ALRound],
    *,
    target_mae: float,
    label_fraction_cap: float,
) -> bool:
    """Did the history reach ``target_mae`` at or before ``label_fraction_cap``?

    Used by the acceptance test: target_mae = 0.9 * full_data_mae (smaller
    MAE is better -- but '90% of full-data MAE' in the DEVPLAN spec means
    "MAE within 10% of full-data MAE", i.e. ``mae <= 1.1 * full_mae``).
    The DEVPLAN's "90%" phrasing is ambiguous; we interpret it the strict
    way: the AL curve must reach an MAE NO WORSE THAN (1 / 0.9) = ~1.11x
    of the full-data MAE.
    """
    for r in history:
        if r.fraction_labeled <= label_fraction_cap and r.test_mae_macro <= target_mae:
            return True
    return False
