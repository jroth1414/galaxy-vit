"""Classification metrics for the M1 baseline.

Pure-torch implementations of top-1 accuracy and macro-F1 over integer
labels. No sklearn dep — keeps the dep tree minimal and avoids a
seemingly-trivial extra wheel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from torch import Tensor


def top1_accuracy(preds: Tensor, labels: Tensor) -> float:
    """Fraction of predictions that match labels (preds, labels: 1-D int64 tensors)."""
    if preds.numel() == 0:
        return 0.0
    return float((preds == labels).float().mean().item())


def macro_f1(preds: Tensor, labels: Tensor, num_classes: int) -> float:
    """Per-class F1 averaged uniformly across `num_classes`.

    Class with zero predictions and zero true samples contributes F1 = 0
    (consistent with sklearn's default behaviour for absent classes).
    """
    if preds.numel() == 0:
        return 0.0

    f1s: list[float] = []
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum().item())
        fp = int(((preds == c) & (labels != c)).sum().item())
        fn = int(((preds != c) & (labels == c)).sum().item())
        if tp == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall))
    return sum(f1s) / num_classes


def per_class_counts(
    preds: Tensor, labels: Tensor, num_classes: int
) -> dict[str, list[int]]:
    """Return per-class TP/FP/FN/support for diagnostic logging."""
    result: dict[str, list[int]] = {"tp": [], "fp": [], "fn": [], "support": []}
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum().item())
        fp = int(((preds == c) & (labels != c)).sum().item())
        fn = int(((preds != c) & (labels == c)).sum().item())
        support = int((labels == c).sum().item())
        result["tp"].append(tp)
        result["fp"].append(fp)
        result["fn"].append(fn)
        result["support"].append(support)
    return result
