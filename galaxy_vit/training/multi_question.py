"""Per-question masked cross-entropy + per-question top-1 accuracy (T2.3).

The Walmsley+23 reproduction baseline trains a single Zoobot encoder
with a 34-logit Linear head and computes cross-entropy independently
on each of the 10 question slices. Galaxies with ``total-votes <
min_votes`` on a question contribute zero loss (and zero gradient
signal) for that question — the per-sample, per-question ``valid``
mask is computed by :func:`galaxy_vit.data.gz_desi_labels.extract_plurality_labels`.

T3.4 will replace this with the full Dirichlet-Multinomial likelihood;
T2.3 is the simpler softmax-on-plurality calibration check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from torch import Tensor


def multi_question_loss(
    logits: Tensor,
    plurality: Tensor,
    valid: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
) -> Tensor:
    """Sum of per-question cross-entropy, masked by ``valid``.

    Parameters
    ----------
    logits:
        ``(B, num_answers)`` raw logits from the multi-question head.
    plurality:
        ``(B, num_questions)`` int plurality answer per question.
    valid:
        ``(B, num_questions)`` bool mask — True if galaxy ``b`` has
        ``>= min_votes`` on question ``q``.
    question_groups:
        ``[(name, start, end)]`` slices into the flat logit vector,
        one per question. Output of
        :func:`galaxy_vit.data.gz_desi_labels.question_index_groups`.

    Returns
    -------
    Scalar loss tensor — the sum of per-question mean cross-entropies
    over the questions that have any valid examples in the batch.
    Returns 0.0 (still a tensor on the same device) if no questions
    have any valid examples.
    """
    import torch
    import torch.nn.functional as F

    total = torch.zeros((), device=logits.device, dtype=logits.dtype)
    n_active = 0
    for q_idx, (_q_name, start, end) in enumerate(question_groups):
        q_valid = valid[:, q_idx]
        if not q_valid.any():
            continue
        q_logits = logits[q_valid, start:end]
        q_target = plurality[q_valid, q_idx]
        ce = F.cross_entropy(q_logits, q_target)
        total = total + ce
        n_active += 1
    return total


def multi_question_top1(
    logits: Tensor,
    plurality: Tensor,
    valid: Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
) -> dict[str, dict[str, float]]:
    """Per-question argmax top-1 accuracy + valid sample count.

    Returns a dict ``{question_name: {"top1": float, "n_valid": int}}``.
    Questions with zero valid samples in the batch report ``top1=0.0``
    and ``n_valid=0`` — caller should weight by ``n_valid`` when
    aggregating across batches.
    """
    out: dict[str, dict[str, float]] = {}
    for q_idx, (q_name, start, end) in enumerate(question_groups):
        q_valid = valid[:, q_idx]
        n_valid = int(q_valid.sum().item())
        if n_valid == 0:
            out[q_name] = {"top1": 0.0, "n_valid": 0}
            continue
        q_logits = logits[q_valid, start:end]
        q_target = plurality[q_valid, q_idx]
        q_pred = q_logits.argmax(dim=-1)
        correct = int((q_pred == q_target).sum().item())
        out[q_name] = {
            "top1": correct / n_valid,
            "n_valid": n_valid,
            "n_correct": float(correct),
        }
    return out


class MultiQuestionAccumulator:
    """Online aggregator for per-question top-1 across many batches.

    Keeps running totals of ``n_valid`` and ``n_correct`` per question
    so the trainer can compute the dataset-wide accuracy at the end of
    the eval loop without storing every batch's predictions.
    """

    def __init__(self, question_groups: list[tuple[str, int, int]]) -> None:
        self.question_groups = question_groups
        self._n_correct: dict[str, int] = {q: 0 for q, _, _ in question_groups}
        self._n_valid: dict[str, int] = {q: 0 for q, _, _ in question_groups}

    def update(
        self,
        logits: Tensor,
        plurality: Tensor,
        valid: Tensor,
    ) -> None:
        for q_idx, (q_name, start, end) in enumerate(self.question_groups):
            q_valid = valid[:, q_idx]
            n_v = int(q_valid.sum().item())
            if n_v == 0:
                continue
            q_logits = logits[q_valid, start:end]
            q_target = plurality[q_valid, q_idx]
            q_pred = q_logits.argmax(dim=-1)
            self._n_correct[q_name] += int((q_pred == q_target).sum().item())
            self._n_valid[q_name] += n_v

    def result(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for q_name, _, _ in self.question_groups:
            n_v = self._n_valid[q_name]
            n_c = self._n_correct[q_name]
            top1 = (n_c / n_v) if n_v > 0 else 0.0
            out[q_name] = {"top1": top1, "n_valid": n_v, "n_correct": n_c}
        return out

    @property
    def macro_top1(self) -> float:
        """Unweighted mean of per-question top-1 across questions with n_valid>0."""
        per_q = self.result()
        active = [v["top1"] for v in per_q.values() if v["n_valid"] > 0]
        return sum(active) / len(active) if active else 0.0
