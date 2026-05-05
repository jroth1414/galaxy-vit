"""Image transforms and dataset-level statistics.

T1.2 lives here: :func:`compute_normalization_stats` produces the per-channel
mean / std that downstream augmentation pipelines consume via
``configs/normalization.json``.

T1.3 lives here too: the per-image train/eval augmentation operators built
on torchvision.transforms.v2 plus a custom ``rotate_with_reflect_pad`` that
torchvision lacks (its ``RandomRotation`` only supports constant fill).

Heavy deps (numpy, torch, torchvision) are imported lazily inside functions
so this module remains importable in environments that only have the base
package installed (e.g. CI running ``[dev]`` alone without ``[m1]`` /
``[torch-*]``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — typing-only
    import torch
    from numpy.typing import NDArray
    from torch import Tensor

DEFAULT_CHANNELS = 3
DEFAULT_IMAGE_SIZE = 224
ROTATION_RANGE: tuple[float, float] = (-180.0, 180.0)
RESIZED_CROP_SCALE: tuple[float, float] = (0.85, 1.0)
MIXUP_ALPHA = 0.2


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


# ---------------------------------------------------------------------------
# T1.3 — per-image augmentation operators
# ---------------------------------------------------------------------------


def load_normalization(path: Path) -> tuple[list[float], list[float]]:
    """Load ``(mean, std)`` lists from a ``configs/normalization.json`` file.

    The file format is the one produced by ``scripts/compute_normalization.py``
    in T1.2 — see that script for the full schema; only ``mean`` and ``std``
    keys are read here.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["mean"]), list(payload["std"])


def _required_pad_for_rotation(size: int) -> int:
    """Pad needed so any rotation up to 180 deg preserves the inscribed image.

    Worst-case the rotated source pixels span the diagonal of the original
    frame (``size * sqrt(2)``), so each side needs at least
    ``ceil((size * sqrt(2) - size) / 2)`` extra pixels. We add a one-pixel
    safety margin to absorb bilinear-interp boundary effects.
    """
    import numpy as np

    diag = int(np.ceil(size * np.sqrt(2)))
    return (diag - size + 1) // 2 + 1


def rotate_with_reflect_pad(
    img: Tensor,
    angle: float,
    out_size: int = DEFAULT_IMAGE_SIZE,
) -> Tensor:
    """Rotate ``img`` by ``angle`` degrees with reflect padding to avoid corner zeros.

    Pipeline: reflect-pad -> rotate (bilinear) -> center-crop to out_size.

    Operates on ``(C, H, W)`` or ``(B, C, H, W)`` tensors. Reflect padding
    in :func:`torch.nn.functional.pad` requires a 4D float tensor, so uint8
    inputs are cast to float32 / 255 internally and cast back to uint8 on
    return so callers can treat this as dtype-preserving.
    """
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import v2 as T

    squeeze_back = img.ndim == 3
    x = img.unsqueeze(0) if squeeze_back else img

    original_dtype = x.dtype
    cast_back_uint8 = original_dtype == torch.uint8
    if cast_back_uint8:
        x = x.float() / 255.0
    elif not torch.is_floating_point(x):
        x = x.float()

    h, w = int(x.shape[-2]), int(x.shape[-1])
    pad = _required_pad_for_rotation(max(h, w))
    padded = F.pad(x, [pad, pad, pad, pad], mode="reflect")

    rotated = T.functional.rotate(
        padded,
        angle,
        interpolation=T.InterpolationMode.BILINEAR,
        fill=0.0,
    )
    cropped = T.functional.center_crop(rotated, [out_size, out_size])

    if cast_back_uint8:
        cropped = (cropped * 255.0).clamp(0, 255).to(torch.uint8)

    if squeeze_back:
        cropped = cropped.squeeze(0)

    # Narrow mypy's view (torchvision.functional.* lose type info through the
    # `ignore_missing_imports` override that keeps CI lean without torch).
    assert isinstance(cropped, torch.Tensor)
    return cropped


class RandomRotateReflect:
    """torchvision-compatible callable: random angle wrapper around :func:`rotate_with_reflect_pad`.

    Samples a fresh angle on every call. Use inside a ``v2.Compose`` chain.
    """

    def __init__(
        self,
        degrees: tuple[float, float] = ROTATION_RANGE,
        out_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.degrees = degrees
        self.out_size = out_size

    def __call__(self, img: Tensor) -> Tensor:
        import torch

        angle = float(
            torch.empty(1).uniform_(self.degrees[0], self.degrees[1]).item()
        )
        return rotate_with_reflect_pad(img, angle, self.out_size)


def build_train_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
    *,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
    rotation_range: tuple[float, float] = ROTATION_RANGE,
    resized_crop_scale: tuple[float, float] = RESIZED_CROP_SCALE,
) -> Any:
    """Build the T1.3 train-time augmentation pipeline.

    Order:
      1. ``ToImage`` + ``ToDtype(float32, scale=True)`` (uint8 [0,255] -> float [0,1])
      2. :class:`RandomRotateReflect` (pad-rotate-crop, bilinear, fill via reflection)
      3. :class:`v2.RandomHorizontalFlip` (p=0.5)
      4. :class:`v2.RandomVerticalFlip` (p=0.5)
      5. :class:`v2.RandomResizedCrop` with scale in ``resized_crop_scale``
      6. :class:`v2.Normalize` with ``mean`` / ``std`` (defaults to identity)

    No hue jitter (DEVPLAN constraint — colour is morphology signal in DECaLS).
    MixUp is applied at the batch level in the training loop, not here.
    """
    import torch
    from torchvision.transforms import v2 as T

    if mean is None:
        mean = [0.0, 0.0, 0.0]
    if std is None:
        std = [1.0, 1.0, 1.0]

    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            RandomRotateReflect(degrees=rotation_range, out_size=image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomResizedCrop(
                size=image_size,
                scale=resized_crop_scale,
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_eval_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
    *,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> Any:
    """Build the eval-time pipeline: ``ToImage`` + ``ToDtype`` + ``CenterCrop`` + ``Normalize``.

    No randomness, no augmentation.
    """
    import torch
    from torchvision.transforms import v2 as T

    if mean is None:
        mean = [0.0, 0.0, 0.0]
    if std is None:
        std = [1.0, 1.0, 1.0]

    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.CenterCrop(image_size),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def mixup_batch(
    images: Tensor,
    labels: Tensor,
    *,
    alpha: float = MIXUP_ALPHA,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Apply MixUp at the batch level.

    Returns ``(mixed_images, labels_a, labels_b, lam)``. The trainer applies
    the soft-label loss as::

        loss = lam * CE(logits, labels_a) + (1 - lam) * CE(logits, labels_b)

    With ``alpha <= 0`` mixing is disabled and the call is a no-op
    (``lam = 1.0``, both label tensors are the original).
    """
    import torch

    if alpha <= 0:
        return images, labels, labels, 1.0

    n = images.size(0)
    lam_t = torch.distributions.Beta(alpha, alpha).sample()
    lam = float(lam_t.item())
    perm = torch.randperm(n, generator=generator)

    mixed = lam * images + (1.0 - lam) * images[perm]
    return mixed, labels, labels[perm], lam
