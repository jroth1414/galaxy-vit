"""Unit tests for galaxy_vit.training.calibration.

Toy-distribution tests pinned against analytically computed reference
values. No checkpoint, no dataset — these run on every CI invocation.
"""

from __future__ import annotations

import math

import pytest
import torch

from galaxy_vit.training.calibration import (
    binned_reliability,
    brier_score_topclass,
    expected_calibration_error,
    maximum_calibration_error,
)


def test_perfect_calibrator_has_zero_ece_and_mce() -> None:
    """A predictor whose confidence == empirical accuracy in every bin → ECE/MCE 0."""
    # Two bins: 50% confidence with 50% accuracy, 90% confidence with 90% accuracy.
    confidence = torch.tensor(
        [0.5] * 10 + [0.9] * 10,
        dtype=torch.float32,
    )
    correct = torch.tensor(
        [1] * 5 + [0] * 5 + [1] * 9 + [0] * 1,
        dtype=torch.float32,
    )
    ece = expected_calibration_error(confidence, correct, n_bins=10)
    mce = maximum_calibration_error(confidence, correct, n_bins=10)
    # float32 rounding makes "exact zero" infeasible; tolerance is well below
    # any realistic ECE we would care about.
    assert ece == pytest.approx(0.0, abs=1e-6)
    assert mce == pytest.approx(0.0, abs=1e-6)


def test_overconfident_predictor_has_positive_ece() -> None:
    """All examples confidence=0.95 but only 50% correct → ECE = 0.45."""
    confidence = torch.full((20,), 0.95, dtype=torch.float32)
    correct = torch.tensor([1, 0] * 10, dtype=torch.float32)
    ece = expected_calibration_error(confidence, correct, n_bins=10)
    mce = maximum_calibration_error(confidence, correct, n_bins=10)
    assert ece == pytest.approx(0.45, abs=1e-6)
    assert mce == pytest.approx(0.45, abs=1e-6)


def test_underconfident_predictor_has_positive_ece() -> None:
    """Confidence 0.4 but always correct → bin 3 has |0.4 - 1.0| = 0.6 gap."""
    confidence = torch.full((20,), 0.4, dtype=torch.float32)
    correct = torch.ones(20, dtype=torch.float32)
    ece = expected_calibration_error(confidence, correct, n_bins=10)
    mce = maximum_calibration_error(confidence, correct, n_bins=10)
    assert ece == pytest.approx(0.6, abs=1e-6)
    assert mce == pytest.approx(0.6, abs=1e-6)


def test_brier_perfect_predictor_is_zero() -> None:
    """Confidence exactly matches correctness on every example → Brier 0."""
    confidence = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    correct = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    assert brier_score_topclass(confidence, correct) == pytest.approx(0.0, abs=1e-9)


def test_brier_uniform_predictor() -> None:
    """Confidence 0.5 on a 50/50 correctness distribution → Brier 0.25."""
    confidence = torch.full((100,), 0.5, dtype=torch.float32)
    correct = torch.tensor([1.0, 0.0] * 50, dtype=torch.float32)
    assert brier_score_topclass(confidence, correct) == pytest.approx(0.25, abs=1e-9)


def test_brier_constant_confident_wrong() -> None:
    """Confidence 1.0 but always wrong → Brier 1.0 (worst possible)."""
    confidence = torch.ones(20, dtype=torch.float32)
    correct = torch.zeros(20, dtype=torch.float32)
    assert brier_score_topclass(confidence, correct) == pytest.approx(1.0, abs=1e-9)


def test_binned_reliability_bin_assignment() -> None:
    """Confidence values map to the right bin indices."""
    # 10 bins of width 0.1; values 0.05, 0.15, 0.95 land in bins 0, 1, 9.
    confidence = torch.tensor([0.05, 0.15, 0.95], dtype=torch.float32)
    correct = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    rel = binned_reliability(confidence, correct, n_bins=10)
    assert rel["count"] == [1, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    assert rel["mean_confidence"][0] == pytest.approx(0.05, abs=1e-6)
    assert rel["mean_confidence"][1] == pytest.approx(0.15, abs=1e-6)
    assert rel["mean_confidence"][9] == pytest.approx(0.95, abs=1e-6)
    assert rel["accuracy"][0] == pytest.approx(1.0, abs=1e-6)
    assert rel["accuracy"][1] == pytest.approx(0.0, abs=1e-6)
    assert rel["accuracy"][9] == pytest.approx(1.0, abs=1e-6)


def test_binned_reliability_empty_bins_are_safe() -> None:
    """Bins with zero examples report count=0 and 0.0 placeholders for conf/acc."""
    confidence = torch.full((5,), 0.85, dtype=torch.float32)
    correct = torch.ones(5, dtype=torch.float32)
    rel = binned_reliability(confidence, correct, n_bins=10)
    assert rel["count"][8] == 5
    assert sum(rel["count"]) == 5
    for i in range(10):
        if rel["count"][i] == 0:
            assert rel["mean_confidence"][i] == 0.0
            assert rel["accuracy"][i] == 0.0


def test_binned_reliability_full_range_sum() -> None:
    """Total count across all bins equals input length, regardless of n_bins."""
    torch.manual_seed(0)
    confidence = torch.rand(1000)
    correct = (torch.rand(1000) < confidence).float()
    for n_bins in (5, 10, 20, 50):
        rel = binned_reliability(confidence, correct, n_bins=n_bins)
        assert sum(rel["count"]) == 1000


def test_ece_invariant_to_bin_widening() -> None:
    """A well-calibrated predictor stays close to ECE=0 even with coarse binning."""
    torch.manual_seed(1)
    n = 5000
    confidence = torch.rand(n)
    correct = (torch.rand(n) < confidence).float()
    # The Bernoulli sampling variance shrinks as n grows; with 5 k samples
    # ECE should be small for any sensible bin count.
    for n_bins in (5, 10, 20):
        ece = expected_calibration_error(confidence, correct, n_bins=n_bins)
        assert ece < 0.05, f"calibrated predictor has ECE={ece} at n_bins={n_bins}"


def test_ece_mce_relationship() -> None:
    """For any nonempty input, MCE >= ECE (max >= weighted mean)."""
    torch.manual_seed(2)
    confidence = torch.rand(200)
    correct = (torch.rand(200) < 0.5).float()
    ece = expected_calibration_error(confidence, correct, n_bins=10)
    mce = maximum_calibration_error(confidence, correct, n_bins=10)
    assert mce >= ece - 1e-9


def test_validation_rejects_shape_mismatch() -> None:
    confidence = torch.tensor([0.5, 0.5])
    correct = torch.tensor([1.0])
    with pytest.raises(ValueError, match="shape mismatch"):
        expected_calibration_error(confidence, correct)


def test_validation_rejects_empty() -> None:
    empty = torch.tensor([])
    with pytest.raises(ValueError, match="empty"):
        binned_reliability(empty, empty)


def test_validation_rejects_2d_input() -> None:
    confidence = torch.zeros(5, 2)
    correct = torch.zeros(5, 2)
    with pytest.raises(ValueError, match="1-D"):
        expected_calibration_error(confidence, correct)


def test_validation_rejects_zero_bins() -> None:
    confidence = torch.tensor([0.5])
    correct = torch.tensor([1.0])
    with pytest.raises(ValueError, match="n_bins must be >= 1"):
        binned_reliability(confidence, correct, n_bins=0)


def test_ece_finite_on_skewed_distribution() -> None:
    """A real-world-like skewed distribution (many high-confidence) gives a finite ECE."""
    confidence = torch.cat([torch.full((900,), 0.95), torch.full((100,), 0.55)])
    correct = torch.cat([torch.ones(810), torch.zeros(90), torch.ones(60), torch.zeros(40)])
    ece = expected_calibration_error(confidence, correct, n_bins=10)
    assert math.isfinite(ece)
    assert 0.0 <= ece <= 1.0
