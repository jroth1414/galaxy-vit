"""Calibration metrics: binned reliability, ECE, MCE, Brier (T2.5).

Inputs to every function are flat 1-D tensors of:

* ``confidence`` — top-class softmax probability (model's max-class
  confidence on each example) in ``[0, 1]``.
* ``correct``    — 0/1 indicator of whether the model's argmax matched
  the plurality target on each example.

The functions are pure-torch and dependency-free so they can be unit-
tested with toy distributions (no checkpoint, no dataset). Their
outputs are the standard quantities reported in calibration tables:

* :func:`binned_reliability` — per-bin (mean_confidence, mean_accuracy,
  count) tuples for plotting reliability diagrams.
* :func:`expected_calibration_error` — Naeini+15 / Guo+17 ECE: weighted
  mean of |confidence - accuracy| across bins.
* :func:`maximum_calibration_error` — MCE: max bin-level
  |confidence - accuracy|.
* :func:`brier_score_topclass` — top-class Brier (mean squared error
  between confidence and correctness indicator).

T3.x's Dirichlet head will compare its calibration against the T2.3
baseline using these same functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from torch import Tensor


def _validate_inputs(confidence: Tensor, correct: Tensor) -> None:
    if confidence.shape != correct.shape:
        raise ValueError(
            f"shape mismatch: confidence {tuple(confidence.shape)} "
            f"vs correct {tuple(correct.shape)}"
        )
    if confidence.ndim != 1:
        raise ValueError(f"expected 1-D tensors, got ndim={confidence.ndim}")
    if confidence.numel() == 0:
        raise ValueError("empty input tensors")


def binned_reliability(
    confidence: Tensor,
    correct: Tensor,
    *,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Bin per-example confidence and accuracy into ``n_bins`` equal-width bins.

    Bin edges span ``[0.0, 1.0]``. The lowest-edge bin is closed on the
    left (``confidence == 0`` lands in bin 0); every other bin is
    half-open on the left.

    Returns
    -------
    Dict with parallel lists of length ``n_bins``:

    * ``bin_lower``       — bin's left edge
    * ``bin_upper``       — bin's right edge
    * ``count``           — number of examples in the bin
    * ``mean_confidence`` — mean confidence of examples in the bin
                            (NaN-safe: 0.0 for empty bins)
    * ``accuracy``        — fraction of bin examples with correct=1
                            (0.0 for empty bins)
    """
    import torch

    _validate_inputs(confidence, correct)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=confidence.device)
    # bucketize with right=False puts a value v into bin i where edges[i] < v <= edges[i+1].
    # We want [edges[i], edges[i+1]) bins (right-open) so subtract 1 and clamp.
    bin_idx = torch.bucketize(confidence, edges[1:-1], right=False).clamp_(0, n_bins - 1)

    counts = torch.zeros(n_bins, dtype=torch.long, device=confidence.device)
    sum_conf = torch.zeros(n_bins, dtype=torch.float64, device=confidence.device)
    sum_correct = torch.zeros(n_bins, dtype=torch.float64, device=confidence.device)
    counts.scatter_add_(0, bin_idx, torch.ones_like(bin_idx))
    sum_conf.scatter_add_(0, bin_idx, confidence.to(torch.float64))
    sum_correct.scatter_add_(0, bin_idx, correct.to(torch.float64))

    safe_counts = counts.clamp(min=1).to(torch.float64)
    mean_conf = (sum_conf / safe_counts).where(counts > 0, torch.zeros_like(sum_conf))
    accuracy = (sum_correct / safe_counts).where(counts > 0, torch.zeros_like(sum_correct))

    return {
        "bin_lower": edges[:-1].tolist(),
        "bin_upper": edges[1:].tolist(),
        "count": counts.tolist(),
        "mean_confidence": mean_conf.tolist(),
        "accuracy": accuracy.tolist(),
    }


def expected_calibration_error(
    confidence: Tensor,
    correct: Tensor,
    *,
    n_bins: int = 10,
) -> float:
    """Weighted mean |conf - acc| across bins (Naeini+15 ECE).

    Bins with zero examples contribute nothing. Returns 0.0 only when
    every bin's confidence equals its accuracy exactly.
    """
    rel = binned_reliability(confidence, correct, n_bins=n_bins)
    total = float(sum(rel["count"]))
    if total == 0.0:
        return 0.0
    ece = 0.0
    for cnt, conf, acc in zip(
        rel["count"], rel["mean_confidence"], rel["accuracy"], strict=True
    ):
        if cnt == 0:
            continue
        ece += (cnt / total) * abs(conf - acc)
    return ece


def maximum_calibration_error(
    confidence: Tensor,
    correct: Tensor,
    *,
    n_bins: int = 10,
) -> float:
    """Maximum bin-level |conf - acc| across bins (worst miscalibration)."""
    rel = binned_reliability(confidence, correct, n_bins=n_bins)
    worst = 0.0
    for cnt, conf, acc in zip(
        rel["count"], rel["mean_confidence"], rel["accuracy"], strict=True
    ):
        if cnt == 0:
            continue
        worst = max(worst, abs(conf - acc))
    return worst


def brier_score_topclass(confidence: Tensor, correct: Tensor) -> float:
    """Top-class Brier score: mean squared error between confidence and correctness.

    Always in ``[0, 1]``. A perfectly calibrated and accurate predictor
    has Brier=0; a uniform predictor on a balanced binary correctness
    distribution has Brier=0.25.
    """
    import torch

    _validate_inputs(confidence, correct)
    err = (confidence.to(torch.float64) - correct.to(torch.float64)).pow(2)
    return float(err.mean().item())
