"""Analytic posterior summaries for the Dirichlet head (T3.7).

Given a Dirichlet ``concentration`` vector ``alpha`` per question (output of
:class:`galaxy_vit.models.dirichlet_head.DirichletMultinomialHead`), each
per-answer probability has a closed-form marginal:

    p_i ~ Beta(alpha_i, sum(alpha) - alpha_i)

so we can produce point estimates (``posterior_mean = alpha_i / sum(alpha)``)
and analytic credible intervals (Beta inverse CDF on (1-ci)/2 and
(1+ci)/2) without sampling. This module wraps scipy.stats.beta.ppf
because torch.distributions.Beta.icdf raises NotImplementedError.

Used by:

* T3.6 — coverage @ 95% CI gate (the acceptance metric for the full
  500k training run).
* T4.3 — Multi-Question Posteriors frontend tab.
* T5.x — release model card uncertainty figures.

The functions return torch tensors so they compose cleanly with the rest
of the pipeline; gradients do NOT flow through the inverse-CDF path
(scipy boundary), which is fine because credible intervals are an
inference-time summary, never an objective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from torch import Tensor

DEFAULT_CI = 0.95


def _validate_ci(ci: float) -> None:
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")


def posterior_mean(
    alpha: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
) -> Tensor:
    """Per-answer posterior mean = ``alpha_i / sum(alpha_q)`` per question slice.

    Returns a tensor with the same flat ``(B, num_answers)`` shape as
    ``alpha``. Slices that sum to zero (impossible in practice given the
    head's ``alpha_floor > 0`` contract, but defensive) are clamped to a
    tiny epsilon to avoid division-by-zero.
    """
    import torch

    out = torch.zeros_like(alpha, dtype=torch.float32)
    a = alpha.to(torch.float32)
    for _q_name, start, end in question_groups:
        slice_alpha = a[:, start:end]
        denom = slice_alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        out[:, start:end] = slice_alpha / denom
    return out


def marginal_beta_params(
    alpha: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
) -> tuple[Tensor, Tensor]:
    """Per-answer Beta marginal parameters ``(a_i, b_i) = (alpha_i, A_q - alpha_i)``.

    Returns ``(a, b)`` tensors each shaped like ``alpha`` (B, num_answers),
    where ``a[:, i]`` and ``b[:, i]`` are the parameters of the Beta
    distribution describing the marginal posterior on the i-th answer's
    probability.
    """
    import torch

    a = alpha.to(torch.float32)
    a_param = torch.empty_like(a)
    b_param = torch.empty_like(a)
    for _q_name, start, end in question_groups:
        slice_alpha = a[:, start:end]
        A_q = slice_alpha.sum(dim=-1, keepdim=True)
        a_param[:, start:end] = slice_alpha
        b_param[:, start:end] = A_q - slice_alpha
    return a_param, b_param


def credible_interval(
    alpha: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
    ci: float = DEFAULT_CI,
) -> tuple[Tensor, Tensor]:
    """Equal-tailed credible interval on each per-answer marginal.

    Parameters
    ----------
    alpha : (B, num_answers) Tensor
        Concentration vector from the Dirichlet head.
    question_groups : list[(name, start, end)]
        Slices into ``alpha`` per question (from
        :func:`galaxy_vit.data.schema.question_index_groups`).
    ci : float in (0, 1)
        Total interval mass. Default 0.95 = 95% CI = symmetric 2.5% / 97.5%
        Beta percentiles.

    Returns
    -------
    ``(lower, upper)`` — each ``(B, num_answers)`` Tensor on the same
    device / dtype as ``alpha`` (downcast to float32). Both bounds in
    ``[0, 1]``; ``lower <= upper`` everywhere.
    """
    import torch
    from scipy.stats import beta as scipy_beta

    _validate_ci(ci)
    a_param, b_param = marginal_beta_params(alpha, question_groups=question_groups)
    a_np = a_param.detach().cpu().numpy()
    b_np = b_param.detach().cpu().numpy()
    q_lo = (1.0 - ci) / 2.0
    q_hi = 1.0 - q_lo
    lower_np = scipy_beta.ppf(q_lo, a_np, b_np)
    upper_np = scipy_beta.ppf(q_hi, a_np, b_np)
    lower = torch.as_tensor(lower_np, dtype=torch.float32, device=alpha.device)
    upper = torch.as_tensor(upper_np, dtype=torch.float32, device=alpha.device)
    return lower, upper


def coverage(
    alpha: Tensor,
    observed_fractions: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
    valid: Tensor | None = None,
    ci: float = DEFAULT_CI,
) -> float:
    """Fraction of (galaxy, answer) pairs where ``observed`` falls inside the CI.

    A well-calibrated model on i.i.d. test data should have
    ``coverage approx ci``. T3.6 gates on this being at least 0.85 with
    ``ci=0.95`` (DEVPLAN target).

    Parameters
    ----------
    alpha : (B, num_answers) Tensor
        Predicted concentrations.
    observed_fractions : (B, num_answers) Tensor
        Empirical per-answer vote fractions, e.g. ``count_i / total_q``.
    question_groups : list[(name, start, end)]
    valid : optional (B, num_questions) bool Tensor
        Per-galaxy, per-question validity mask. When given, only answers
        of questions with ``valid[b, q] = True`` enter the denominator.
        ``None`` means count every (galaxy, answer) pair.
    ci : float in (0, 1)

    Returns
    -------
    float in ``[0, 1]`` — the empirical coverage. ``0.0`` when no valid
    pair exists in the batch.
    """
    import torch

    _validate_ci(ci)
    if observed_fractions.shape != alpha.shape:
        raise ValueError(
            f"observed shape {tuple(observed_fractions.shape)} != "
            f"alpha shape {tuple(alpha.shape)}"
        )

    lower, upper = credible_interval(alpha, question_groups=question_groups, ci=ci)
    obs = observed_fractions.to(torch.float32)
    inside = (obs >= lower) & (obs <= upper)  # (B, num_answers) bool

    if valid is None:
        n_valid = inside.numel()
        n_inside = int(inside.sum().item())
    else:
        if valid.shape[0] != alpha.shape[0]:
            raise ValueError(
                f"valid batch dim {valid.shape[0]} != alpha batch dim {alpha.shape[0]}"
            )
        if valid.shape[1] != len(question_groups):
            raise ValueError(
                f"valid num_questions {valid.shape[1]} != "
                f"len(question_groups) {len(question_groups)}"
            )
        # Broadcast per-question validity onto answer columns.
        per_answer_valid = torch.zeros_like(inside)
        for q_idx, (_q_name, start, end) in enumerate(question_groups):
            q_valid = valid[:, q_idx].unsqueeze(-1)  # (B, 1)
            per_answer_valid[:, start:end] = q_valid
        n_valid = int(per_answer_valid.sum().item())
        n_inside = int((inside & per_answer_valid).sum().item())

    if n_valid == 0:
        return 0.0
    return n_inside / n_valid
