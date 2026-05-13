"""C-15 — API tests for /api/training_movie.

Skipped when artifacts/training_movie.parquet is missing (it's only
produced by the optional retrain + post-process pipeline).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")
pytest.importorskip("pandas")

ZOOBOT_CKPT = Path("runs/m1_zoobot_finetune/best.pt")
TRAINING_MOVIE = Path("artifacts/training_movie.parquet")

if not ZOOBOT_CKPT.is_file():  # pragma: no cover
    pytest.skip("M1 checkpoint missing", allow_module_level=True)
if not TRAINING_MOVIE.is_file():
    pytest.skip(
        "training_movie.parquet missing; run the retrain + "
        "`python -m scripts.build_training_movie` first",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient  # noqa: E402

from galaxy_vit.serve.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_C15_training_movie_response_shape(client: TestClient) -> None:
    response = client.get("/api/training_movie")
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body["epochs"], list) and len(body["epochs"]) >= 1
    assert isinstance(body["label_names"], list) and len(body["label_names"]) >= 1
    assert isinstance(body["frames"], list) and len(body["frames"]) >= 1
    for f in body["frames"][:5]:
        assert {"epoch", "galaxy_id", "umap_x", "umap_y", "label_name"} == set(
            f.keys()
        )


def test_C15_training_movie_every_epoch_has_same_galaxy_count(
    client: TestClient,
) -> None:
    """Per-epoch row counts should be uniform (n_galaxies stable across epochs)."""
    body = client.get("/api/training_movie").json()
    counts: dict[int, int] = {}
    for f in body["frames"]:
        counts[f["epoch"]] = counts.get(f["epoch"], 0) + 1
    unique_counts = set(counts.values())
    assert len(unique_counts) == 1, (
        f"per-epoch row counts vary: {counts}; the trainer's per-epoch "
        f"hook dropped some galaxies"
    )


def test_C15_training_movie_epochs_sorted(client: TestClient) -> None:
    body = client.get("/api/training_movie").json()
    assert body["epochs"] == sorted(body["epochs"])


def test_C15_training_movie_label_palette_subset_of_canonical(
    client: TestClient,
) -> None:
    """Demo-galaxy labels are drawn from the canonical 3-class set."""
    body = client.get("/api/training_movie").json()
    canonical = {"smooth", "featured-or-disk", "artifact"}
    assert set(body["label_names"]).issubset(canonical | {"unknown"})
