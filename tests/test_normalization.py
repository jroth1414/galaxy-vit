"""T1.2 — per-channel normalization stats acceptance test.

Hermetic: builds a deterministic synthetic fixture in-memory via
``np.random.default_rng(42)`` (PCG64 — stable across numpy 1.x and 2.x for a
fixed seed) and asserts the function reproduces hard-coded cached statistics
to within 1e-4. No image I/O, no network, no `datasets` import.
"""

from __future__ import annotations

import numpy as np

from galaxy_vit.data.transforms import compute_normalization_stats

# Fixture parameters — change any of these and the cached values below must be
# regenerated. We pin the stream check (FIXTURE_SUM) so that a future numpy
# behavior change in PCG64 surfaces as a *fixture* failure rather than a
# spurious normalization-stats mismatch.
FIXTURE_SEED = 42
FIXTURE_SHAPE = (10, 32, 32, 3)  # (n_images, H, W, channels)
FIXTURE_SUM = 3883900  # int sum of all uint8 pixel values; integrity check

# Cached expected normalization stats — captured from a one-shot run of
# `compute_normalization_stats` against the fixture above. Within the 1e-4
# tolerance these constants pin both the algorithm and the fixture stream.
EXPECTED_MEAN = [0.4961554075, 0.4920955882, 0.4991494332]
EXPECTED_STD = [0.2901098778, 0.2896214063, 0.2900966152]
TOLERANCE = 1e-4


def _build_fixture() -> np.ndarray:
    rng = np.random.default_rng(seed=FIXTURE_SEED)
    images = rng.integers(0, 256, size=FIXTURE_SHAPE, dtype=np.uint8)
    return images


def test_fixture_stream_unchanged() -> None:
    """Guard: numpy's PCG64 stream for seed=42 produces the expected fixture.

    If this fails, numpy changed its bit generator output and the cached
    EXPECTED_* constants in `test_normalization_matches_cached` must be
    re-derived (re-run the one-shot in the T1.2 commit message).
    """
    images = _build_fixture()
    assert images.shape == FIXTURE_SHAPE
    assert images.dtype == np.uint8
    assert int(images.sum()) == FIXTURE_SUM, (
        f"fixture stream changed: pixel sum {int(images.sum())} != {FIXTURE_SUM}"
    )


def test_normalization_matches_cached() -> None:
    """T1.2 acceptance: stats reproduce cached reference within 1e-4.

    Running ``compute_normalization_stats`` on the deterministic 10x32x32x3
    uint8 fixture must return per-channel mean and std that each agree with
    EXPECTED_MEAN / EXPECTED_STD to within an absolute tolerance of 1e-4.
    """
    images = _build_fixture()
    stats = compute_normalization_stats(iter(images))

    assert set(stats.keys()) == {"mean", "std"}
    assert len(stats["mean"]) == 3
    assert len(stats["std"]) == 3

    for ch in range(3):
        diff_mean = abs(stats["mean"][ch] - EXPECTED_MEAN[ch])
        diff_std = abs(stats["std"][ch] - EXPECTED_STD[ch])
        assert diff_mean <= TOLERANCE, (
            f"channel {ch} mean {stats['mean'][ch]:.10f} differs from "
            f"cached {EXPECTED_MEAN[ch]:.10f} by {diff_mean:.2e} > {TOLERANCE}"
        )
        assert diff_std <= TOLERANCE, (
            f"channel {ch} std  {stats['std'][ch]:.10f} differs from "
            f"cached {EXPECTED_STD[ch]:.10f} by {diff_std:.2e} > {TOLERANCE}"
        )
