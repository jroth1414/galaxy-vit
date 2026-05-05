"""Image transforms and dataset-level statistics.

T1.2 lives here: :func:`compute_normalization_stats` produces the per-channel
mean / std that downstream augmentation pipelines (T1.3+) consume via
``configs/normalization.json``. The augmentation operators themselves
(rotation, flips, MixUp, etc.) land in T1.3 — this module is intentionally
narrow at T1.2.

Numpy is in the project's ``[dev]`` extra so this module is importable in CI
without the heavier ``[m1]`` HF stack; the lazy ``import numpy as np`` only
fires when a function is called.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from numpy.typing import NDArray

DEFAULT_CHANNELS = 3


def compute_normalization_stats(
    images: Iterable[NDArray[Any]],
    *,
    channels: int = DEFAULT_CHANNELS,
) -> dict[str, list[float]]:
    """Streaming per-channel mean / std over an iterable of HxWxC uint8 images.

    The computation is single-pass via running first and second moments::

        mean = sum(x) / N
        var  = sum(x^2) / N - mean^2        (clamped at 0 for FP noise)
        std  = sqrt(var)

    Numerically stable for natural-image value ranges (uint8 -> float64 / 255).
    Returns values in ``[0, 1]``.

    Parameters
    ----------
    images:
        Any iterable yielding ``ndarray`` of shape ``(H, W, channels)`` and
        ``dtype uint8``. Heights and widths may vary between elements; only
        the channel count must be consistent.
    channels:
        Expected channel count (default 3 for RGB DECaLS thumbnails).

    Returns
    -------
    ``{"mean": [c0, c1, c2], "std": [c0, c1, c2]}`` — plain Python floats so
    the result is JSON-serialisable as-is.

    Raises
    ------
    ValueError
        If any image has the wrong rank / channel count, or if the iterable
        yields zero images.
    """
    import numpy as np

    n_pixels = 0
    n_images = 0
    sums = np.zeros(channels, dtype=np.float64)
    sums_sq = np.zeros(channels, dtype=np.float64)

    for img in images:
        if img.ndim != 3 or img.shape[2] != channels:
            raise ValueError(
                f"image #{n_images}: expected (H, W, {channels}); got shape {img.shape}"
            )
        arr = img.astype(np.float64) / 255.0
        h, w, _ = arr.shape
        n_pixels += h * w
        sums += arr.sum(axis=(0, 1))
        sums_sq += (arr**2).sum(axis=(0, 1))
        n_images += 1

    if n_images == 0:
        raise ValueError("compute_normalization_stats received an empty iterable")

    means = sums / n_pixels
    # Clamp tiny negative variance from float subtraction noise.
    variances = np.maximum(sums_sq / n_pixels - means**2, 0.0)
    stds = np.sqrt(variances)

    return {"mean": means.tolist(), "std": stds.tolist()}
