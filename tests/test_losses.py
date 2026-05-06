"""T1.5 — class-balanced loss weight computation tests.

Pure stdlib — no torch import; only exercises the closed-form formula in
``galaxy_vit.training.losses``.
"""

from __future__ import annotations

import math

import pytest

from galaxy_vit.training.losses import class_balanced_weights


def test_cb_weights_sum_to_K() -> None:
    """Normalisation invariant: weights sum to len(class_counts) within FP noise."""
    counts = [100, 200, 50, 1000, 300]
    weights = class_balanced_weights(counts, beta=0.9999)
    assert len(weights) == len(counts)
    assert math.isclose(sum(weights), len(counts), abs_tol=1e-9)


def test_cb_weights_smaller_class_higher_weight() -> None:
    """Strict monotone: smaller class -> higher weight."""
    counts = [100, 500, 1000, 5000]
    weights = class_balanced_weights(counts, beta=0.9999)
    for i in range(len(counts) - 1):
        assert weights[i] > weights[i + 1], (
            f"weight[{i}] (n={counts[i]}) = {weights[i]:.6f} should exceed "
            f"weight[{i + 1}] (n={counts[i + 1]}) = {weights[i + 1]:.6f}"
        )


def test_cb_weights_uniform_classes_uniform_weights() -> None:
    """When all classes have equal counts, weights are uniform (= 1.0)."""
    counts = [500] * 10
    weights = class_balanced_weights(counts, beta=0.9999)
    for w in weights:
        assert math.isclose(w, 1.0, abs_tol=1e-9)


def test_cb_weights_galaxy10_distribution() -> None:
    """Realistic Galaxy10 train counts (from T1.1's run_config.json)."""
    counts = [757, 1297, 1851, 1419, 234, 1430, 1280, 1840, 996, 1311]
    weights = class_balanced_weights(counts, beta=0.9999)
    # Class 4 has the smallest count (234) -> highest weight.
    assert weights[4] == max(weights)
    # Class 2 has the largest count (1851) -> lowest weight.
    assert weights[2] == min(weights)
    # Sanity: weight ratio between smallest and largest class should be
    # noticeably > 1 (we're rebalancing) but not absurd (< 50x).
    ratio = max(weights) / min(weights)
    assert 2.0 < ratio < 50.0, f"unexpected min/max ratio {ratio:.2f}"


def test_cb_weights_rejects_invalid_beta() -> None:
    with pytest.raises(ValueError, match="beta"):
        class_balanced_weights([100, 100], beta=0.0)
    with pytest.raises(ValueError, match="beta"):
        class_balanced_weights([100, 100], beta=1.0)
    with pytest.raises(ValueError, match="beta"):
        class_balanced_weights([100, 100], beta=-0.5)


def test_cb_weights_rejects_zero_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        class_balanced_weights([100, 0, 200])
