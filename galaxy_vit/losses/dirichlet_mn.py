"""Dirichlet-Multinomial negative log-likelihood with per-question masking (T3.4).

Models the GZ DESI volunteer-vote distribution as Dirichlet-Multinomial:
for each question, the per-galaxy concentration vector ``alpha`` (output of
:class:`galaxy_vit.models.dirichlet_head.DirichletMultinomialHead`)
parameterizes a Dirichlet prior over the per-answer probabilities, and
the observed integer counts are a Multinomial draw from those
probabilities. The marginal likelihood (collapsing the latent Dirichlet)
is:

    log P(c | alpha) = log Gamma(A) - log Gamma(N + A)
                       + sum_k [ log Gamma(alpha_k + c_k) - log Gamma(alpha_k) ]

where ``A = sum(alpha_k)`` and ``N = sum(c_k)``. We omit the
data-only term ``log[ N! / prod(c_k!) ]`` — it doesn't affect gradients
on alpha, and skipping it keeps the loss numerically smaller.

Per-question normalization (the "right" reading of DEVPLAN's vague
spec, deliberately pinned here): for each question we take the **mean**
of per-galaxy NLLs over the valid subset, then **sum** across questions.
This balances rare and common questions on equal footing while still
letting a galaxy with more valid questions contribute proportionally
more loss than one with fewer.

Numerical considerations:

* ``torch.special.gammaln`` is computed in fp32 even when the model
  trains in bf16 — bf16 ``gammaln`` near small alpha is dominated by
  rounding error. The trainer is expected to run the encoder + head
  forward in bf16 then upcast right before this loss.
* alpha is required to be strictly positive (the head's
  ``softplus + alpha_floor`` enforces this); we don't re-clamp here so
  any failure becomes loud (NaN propagation) rather than silent.

T3.5 will add a parity test against Zoobot 2.0's native Dirichlet-
Multinomial loss on identical inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from torch import Tensor


def dirichlet_multinomial_nll_per_question(
    alpha_q: Tensor,
    counts_q: Tensor,
) -> Tensor:
    """Single-question DM NLL, no masking, no reduction.

    Parameters
    ----------
    alpha_q : (B, K) float
        Per-galaxy concentration parameters for the question's K answers.
        Must be strictly positive.
    counts_q : (B, K) int or float
        Per-galaxy observed vote counts.

    Returns
    -------
    (B,) float — per-galaxy NLL with the data-only ``log multinomial
    coefficient`` term omitted. Higher = worse fit.
    """
    import torch

    # Force fp32 — gammaln in bf16 / fp16 is unreliable near small alpha.
    alpha = alpha_q.to(torch.float32)
    counts = counts_q.to(torch.float32)

    A = alpha.sum(dim=-1)  # (B,)
    N = counts.sum(dim=-1)  # (B,)

    # log Gamma(A) - log Gamma(N + A)
    log_norm = torch.special.gammaln(A) - torch.special.gammaln(N + A)
    # sum_k [ log Gamma(alpha_k + c_k) - log Gamma(alpha_k) ]
    log_per_answer = (
        torch.special.gammaln(alpha + counts) - torch.special.gammaln(alpha)
    ).sum(dim=-1)
    out = -(log_norm + log_per_answer)
    assert isinstance(out, torch.Tensor)
    return out


def dirichlet_multinomial_nll(
    alpha: Tensor,
    counts: Tensor,
    valid: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
) -> Tensor:
    """Masked, per-question-normalized Dirichlet-Multinomial NLL.

    Parameters
    ----------
    alpha : (B, num_answers) float
        Concentration vector per galaxy across the flat 34-answer head;
        sliced into per-question groups by ``question_groups``.
    counts : (B, num_answers) int or float
        Observed vote counts in the same flat layout. The trainer
        constructs this directly from the catalog ``<q>_<a>`` columns.
    valid : (B, num_questions) bool
        Per-galaxy, per-question validity mask (output of
        :func:`galaxy_vit.data.masking.compute_question_mask`).
    question_groups : list[(name, start, end)]
        Slices into ``alpha`` / ``counts`` for each question. Output of
        :func:`galaxy_vit.data.schema.question_index_groups`.

    Returns
    -------
    Scalar tensor — sum over questions of the per-question mean NLL,
    where the per-question mean is taken over the valid galaxies of
    that question. Questions with zero valid galaxies in the batch
    contribute 0 (no gradient, but also no NaN).
    """
    import torch

    if alpha.shape != counts.shape:
        raise ValueError(
            f"alpha shape {tuple(alpha.shape)} != counts shape {tuple(counts.shape)}"
        )
    if valid.shape[0] != alpha.shape[0]:
        raise ValueError(
            f"batch dim mismatch: alpha {alpha.shape[0]} vs valid {valid.shape[0]}"
        )
    if valid.shape[1] != len(question_groups):
        raise ValueError(
            f"num_questions mismatch: valid has {valid.shape[1]}, "
            f"question_groups has {len(question_groups)}"
        )

    total = torch.zeros((), device=alpha.device, dtype=torch.float32)
    for q_idx, (_q_name, start, end) in enumerate(question_groups):
        q_valid = valid[:, q_idx]
        n_valid = int(q_valid.sum().item())
        if n_valid == 0:
            continue
        alpha_q = alpha[q_valid, start:end]
        counts_q = counts[q_valid, start:end]
        per_galaxy = dirichlet_multinomial_nll_per_question(alpha_q, counts_q)
        total = total + per_galaxy.mean()
    return total


def expected_fractions(alpha: Tensor, *, question_groups: list[tuple[str, int, int]]) -> Tensor:
    """Posterior-mean per-answer fractions: alpha_k / sum_k(alpha) per question slice.

    Returned tensor has the same flat (B, num_answers) shape as ``alpha``.
    Useful for the T3.6 vote-MAE metric and the T3.7 posterior module.
    """
    import torch

    out = torch.zeros_like(alpha, dtype=torch.float32)
    alpha_f = alpha.to(torch.float32)
    for _q_name, start, end in question_groups:
        slice_alpha = alpha_f[:, start:end]
        denom = slice_alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        out[:, start:end] = slice_alpha / denom
    return out
