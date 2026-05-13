"""C-15 — Unit tests for the trainer's per-epoch demo-feature helpers.

We test the helper API (load thumbs, extract features, write parquet)
without spinning up the full training loop -- the round trip would be
3 hours of GPU time and the hook is the only piece worth isolating.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from galaxy_vit.training.dirichlet_trainer import (  # noqa: E402
    _extract_demo_features,
    _load_demo_galaxy_thumbs,
    _write_per_epoch_features,
)


def _make_demo_bundle(
    tmp_path: Path,
    *,
    n_galaxies: int = 4,
    size: int = 32,
) -> Path:
    """Create a tiny artifacts/demo_galaxies/ layout the loader can read."""
    bundle = tmp_path / "demo_galaxies"
    (bundle / "thumbs").mkdir(parents=True)
    from PIL import Image

    manifest: list[dict[str, Any]] = []
    for i in range(n_galaxies):
        gid = f"{i:04d}"
        img = Image.new("RGB", (size, size), color=(i * 30 % 255, 100, 200))
        img.save(bundle / "thumbs" / f"{gid}.jpg", format="JPEG", quality=85)
        manifest.append(
            {
                "id": gid,
                "smooth_or_featured_plurality": "smooth",
                "counts": [],
                "valid": [],
            }
        )
    (bundle / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return bundle


def test_load_demo_galaxy_thumbs_returns_aligned_batch(tmp_path: Path) -> None:
    bundle = _make_demo_bundle(tmp_path, n_galaxies=3, size=64)
    loaded = _load_demo_galaxy_thumbs(
        bundle, image_size=32, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]
    )
    assert loaded is not None
    ids, batch = loaded
    assert ids == ["0000", "0001", "0002"]
    assert batch.shape == (3, 3, 32, 32)
    assert batch.dtype == torch.float32


def test_load_demo_galaxy_thumbs_returns_none_when_manifest_missing(
    tmp_path: Path,
) -> None:
    loaded = _load_demo_galaxy_thumbs(
        tmp_path / "nope",
        image_size=32,
        mean=[0.5, 0.5, 0.5],
        std=[0.25, 0.25, 0.25],
    )
    assert loaded is None


def test_load_demo_galaxy_thumbs_skips_missing_thumb_files(
    tmp_path: Path,
) -> None:
    bundle = _make_demo_bundle(tmp_path, n_galaxies=3)
    # Remove one thumbnail file but keep its manifest entry.
    (bundle / "thumbs" / "0001.jpg").unlink()
    loaded = _load_demo_galaxy_thumbs(
        bundle,
        image_size=32,
        mean=[0.5, 0.5, 0.5],
        std=[0.25, 0.25, 0.25],
    )
    assert loaded is not None
    ids, batch = loaded
    assert ids == ["0000", "0002"]
    assert batch.shape[0] == 2


def test_extract_demo_features_runs_on_toy_encoder() -> None:
    """The helper accepts any model exposing `model.encoder`."""

    class _Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3, padding=1),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
            )

        def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(alpha=torch.zeros(pixel_values.shape[0], 4))

    model = _Toy()
    batch = torch.randn(5, 3, 16, 16)
    feats = _extract_demo_features(model, batch, device=torch.device("cpu"))
    assert feats.shape == (5, 8)
    assert feats.device.type == "cpu"


def test_extract_demo_features_restores_training_mode() -> None:
    """The helper must not leave the model in eval() afterwards."""

    class _Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten()
            )

        def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(alpha=torch.zeros(pixel_values.shape[0], 4))

    model = _Toy()
    model.train()
    _extract_demo_features(
        model, torch.randn(2, 3, 4, 4), device=torch.device("cpu")
    )
    assert model.training, "encoder feature extract left model in eval()"


def test_write_per_epoch_features_round_trip(tmp_path: Path) -> None:
    """Writing then reading the parquet preserves epoch, galaxy_id, features."""
    pd = pytest.importorskip("pandas")
    rows = [
        {"epoch": 0, "galaxy_id": "0000", "features": [0.1, 0.2, 0.3]},
        {"epoch": 0, "galaxy_id": "0001", "features": [0.4, 0.5, 0.6]},
        {"epoch": 1, "galaxy_id": "0000", "features": [0.7, 0.8, 0.9]},
    ]
    out_path = tmp_path / "feats.parquet"
    _write_per_epoch_features(out_path, rows)
    df = pd.read_parquet(out_path)
    assert list(df.columns) == ["epoch", "galaxy_id", "features"]
    assert len(df) == 3
    assert df["epoch"].tolist() == [0, 0, 1]
    assert df["galaxy_id"].tolist() == ["0000", "0001", "0000"]
