"""T1.3 — augmentation pipeline acceptance tests.

The whole module is skipped if torch / torchvision aren't installed, so CI
running ``[dev]`` alone still passes (the augmentation gates are exercised
locally, where the developer has the matching ``[torch-cu128]`` extra).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

# Imports below depend on torch but module-level skip above guards them.
from galaxy_vit.data.transforms import (  # noqa: E402
    DEFAULT_IMAGE_SIZE,
    build_eval_transform,
    build_train_transform,
    mixup_batch,
    rotate_with_reflect_pad,
)


def test_transforms_shape_dtype() -> None:
    """T1.3 acceptance: train transform yields (3, image_size, image_size) float32.

    With identity normalization (mean=0, std=1) the output should remain in
    [0, 1] (uint8 inputs scaled by ToDtype(scale=True)).
    """
    img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
    transform = build_train_transform(
        image_size=DEFAULT_IMAGE_SIZE,
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
    )
    out = transform(img)

    assert out.shape == (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE), (
        f"expected (3, {DEFAULT_IMAGE_SIZE}, {DEFAULT_IMAGE_SIZE}); got {tuple(out.shape)}"
    )
    assert out.dtype == torch.float32, f"expected float32; got {out.dtype}"
    assert out.min().item() >= -1e-5, f"unexpected sub-zero value {out.min().item()}"
    assert out.max().item() <= 1.0 + 1e-5, f"unexpected super-one value {out.max().item()}"


def test_no_corner_zeros_after_rotation() -> None:
    """T1.3 acceptance: reflect padding -> rotated images have no zero corners.

    With ``fill=0`` and no padding, a 45-degree rotation of a uniform image
    leaves triangular black regions in the corners. Our pad-rotate-crop
    pipeline must keep every corner pixel strictly positive.
    """
    img = torch.full((3, 64, 64), 200, dtype=torch.uint8)

    # 45 deg is the worst case for showing fill regions in a square frame.
    rotated = rotate_with_reflect_pad(img, angle=45.0, out_size=64)

    assert rotated.shape == (3, 64, 64)
    assert rotated.dtype == torch.uint8

    cs = 4
    corners = {
        "top-left": rotated[:, :cs, :cs],
        "top-right": rotated[:, :cs, -cs:],
        "bottom-left": rotated[:, -cs:, :cs],
        "bottom-right": rotated[:, -cs:, -cs:],
    }
    for name, corner in corners.items():
        assert (corner > 0).all(), (
            f"{name} corner contains zero pixels (reflect padding likely failed): "
            f"{corner.tolist()}"
        )


def test_eval_transform_shape_dtype() -> None:
    """Eval transform: center-crop + normalize, no randomness."""
    img = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
    transform = build_eval_transform(
        image_size=DEFAULT_IMAGE_SIZE,
        mean=[0.5, 0.5, 0.5],
        std=[0.25, 0.25, 0.25],
    )
    out = transform(img)
    assert out.shape == (3, DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    assert out.dtype == torch.float32


def test_mixup_batch_shapes_and_lambda_passthrough() -> None:
    """MixUp returns matching tensors plus a scalar lambda; alpha=0 is a no-op."""
    images = torch.randn(8, 3, 32, 32)
    labels = torch.arange(8)

    # alpha=0: no mixing, lam=1, labels_a == labels_b == original.
    out_img, la, lb, lam = mixup_batch(images, labels, alpha=0.0)
    assert lam == 1.0
    assert torch.equal(la, labels)
    assert torch.equal(lb, labels)
    assert torch.equal(out_img, images)

    # alpha>0: mixed image is a convex combination; labels_a and labels_b
    # cover the original batch (labels_b is a permutation of labels).
    out_img, la, lb, lam = mixup_batch(images, labels, alpha=0.2)
    assert 0.0 <= lam <= 1.0
    assert out_img.shape == images.shape
    assert la.shape == labels.shape
    assert lb.shape == labels.shape
    assert torch.equal(la, labels)
    assert torch.equal(torch.sort(lb).values, torch.sort(labels).values)
