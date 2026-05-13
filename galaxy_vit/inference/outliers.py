"""Outlier-ranking metrics for the "most interesting galaxies" panel (S-3).

Three sort orders are surfaced to the live demo:

* **Predictive entropy** — sum across questions of ``H[p_pred]`` where
  ``p_pred = alpha / sum(alpha_q)`` per question slice. Highest for
  galaxies the model thinks are categorically ambiguous.
* **BALD** (Houlsby+11) — sum across questions of mutual information
  between the next volunteer vote and the Dirichlet parameters; closed
  form via digamma. Highest for galaxies where the model is
  "confidently uncertain" (small posterior support but wide predictive
  spread).
* **Volunteer disagreement** — per-galaxy mean over valid questions of
  the L1 distance ``|expected_fractions(alpha) - vote_fracs|`` summed
  per question. Only galaxies with at least one valid question are
  included.

The first two reuse the closed-form helpers in
:mod:`galaxy_vit.training.active_learning` (originally written for the
T4.1 active-learning loop) so the math is shared with the AL
acquisitions.

This module is heavyweight-dep-free at import time; torch is lazy-
imported inside the functions so unit tests can run on the
``[dev]``-only CI install.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover -- typing-only
    from torch import Tensor


EPS = 1e-12


@dataclass(frozen=True)
class OutlierEntry:
    """One row of an outlier ranking (idx into test_thumbs/ + metric value)."""

    idx: int
    value: float


def predictive_entropy_total(
    alpha: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
) -> Tensor:
    """Per-galaxy predictive entropy summed across all 10 questions.

    Thin re-export of
    :func:`galaxy_vit.training.active_learning.predictive_entropy` so
    the outlier scoring module presents a single import surface.
    """
    from galaxy_vit.training.active_learning import predictive_entropy

    return predictive_entropy(alpha, question_groups=question_groups)


def bald_total(
    alpha: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
) -> Tensor:
    """Per-galaxy BALD score summed across all 10 questions.

    Thin re-export of
    :func:`galaxy_vit.training.active_learning.bald_score`.
    """
    from galaxy_vit.training.active_learning import bald_score

    return bald_score(alpha, question_groups=question_groups)


def volunteer_disagreement(
    alpha: Tensor,
    volunteer_fracs: Tensor,
    valid: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
) -> Tensor:
    """Per-galaxy mean L1 disagreement vs volunteers across valid questions.

    Parameters
    ----------
    alpha : (B, K) float
        Dirichlet concentration parameters per galaxy.
    volunteer_fracs : (B, K) float
        Per-answer volunteer fractions (e.g. ``count_i / total`` per
        question slice). Invalid questions should be zeroed by the
        caller; the per-question L1 contribution is gated by ``valid``
        either way.
    valid : (B, Q) bool
        Per-galaxy, per-question validity mask. Galaxies with all
        questions invalid receive a disagreement of 0 (and should be
        filtered out by the caller).
    question_groups : list[(name, start, end)]
        Slice layout from
        :func:`galaxy_vit.data.schema.question_index_groups`.

    Returns
    -------
    ``(B,)`` float tensor — mean over valid questions of the per-
    question L1.
    """
    import torch

    from galaxy_vit.losses.dirichlet_mn import expected_fractions

    if alpha.shape != volunteer_fracs.shape:
        raise ValueError(
            f"alpha shape {tuple(alpha.shape)} != "
            f"volunteer_fracs shape {tuple(volunteer_fracs.shape)}"
        )
    if valid.shape[0] != alpha.shape[0]:
        raise ValueError(
            f"batch dim mismatch: alpha {alpha.shape[0]} vs "
            f"valid {valid.shape[0]}"
        )
    if valid.shape[1] != len(question_groups):
        raise ValueError(
            f"num_questions mismatch: valid has {valid.shape[1]}, "
            f"question_groups has {len(question_groups)}"
        )

    pred = expected_fractions(alpha, question_groups=question_groups)
    b = alpha.shape[0]
    accum = torch.zeros(b, dtype=torch.float32, device=alpha.device)
    n_valid = torch.zeros(b, dtype=torch.float32, device=alpha.device)
    for q_idx, (_q_name, start, end) in enumerate(question_groups):
        mask = valid[:, q_idx].float()
        per_q = (pred[:, start:end] - volunteer_fracs[:, start:end]).abs().sum(
            dim=-1
        )
        accum = accum + per_q * mask
        n_valid = n_valid + mask
    return accum / n_valid.clamp_min(1.0)


def topk_indices(
    scores: Tensor,
    *,
    k: int,
    descending: bool = True,
) -> list[OutlierEntry]:
    """Return the top-K rows of ``scores`` as ``OutlierEntry`` records.

    Stable across calls (uses ``torch.topk`` which is deterministic on
    a fixed-shape tensor on CPU). Result is sorted by ``value``
    descending when ``descending=True``.
    """
    import torch

    if scores.ndim != 1:
        raise ValueError(
            f"scores must be 1-D; got shape {tuple(scores.shape)}"
        )
    if k <= 0:
        raise ValueError(f"k must be >= 1; got {k}")
    n = scores.shape[0]
    k = min(k, n)
    topk = torch.topk(scores, k=k, largest=descending, sorted=True)
    return [
        OutlierEntry(idx=int(i.item()), value=float(v.item()))
        for i, v in zip(topk.indices, topk.values, strict=True)
    ]
