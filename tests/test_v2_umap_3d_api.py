"""A-6 — API tests for /api/umap_points?n_dims=3.

Skipped when artifacts/umap_3d_coords.parquet (or any prerequisite) is
missing, mirroring the gating pattern in tests/test_v2_*.py.
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
UMAP_2D_COORDS = Path("artifacts/umap_coords.parquet")
UMAP_3D_COORDS = Path("artifacts/umap_3d_coords.parquet")
TEST_THUMBS_DIR = Path("artifacts/test_thumbs")

if not ZOOBOT_CKPT.is_file():  # pragma: no cover -- local-only gate
    pytest.skip("M1 checkpoint missing", allow_module_level=True)
if not UMAP_2D_COORDS.is_file():
    pytest.skip(
        "umap_coords.parquet missing; run scripts/extract_umap.py first",
        allow_module_level=True,
    )
if not UMAP_3D_COORDS.is_file():
    pytest.skip(
        "umap_3d_coords.parquet missing; run scripts/extract_umap_3d.py first",
        allow_module_level=True,
    )
if not TEST_THUMBS_DIR.is_dir() or not list(TEST_THUMBS_DIR.glob("*.jpg")):
    pytest.skip(
        "test_thumbs missing; run scripts/build_test_thumbs.py first",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient  # noqa: E402

from galaxy_vit.serve.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_A6_umap_2d_has_null_z(client: TestClient) -> None:
    """Default n_dims=2 returns z=None on every point."""
    response = client.get("/api/umap_points")
    assert response.status_code == 200, response.text
    body = response.json()
    for p in body["points"][:5]:
        assert p["z"] is None


def test_A6_umap_3d_returns_populated_z(client: TestClient) -> None:
    """n_dims=3 returns finite z for every point."""
    response = client.get("/api/umap_points", params={"n_dims": 3})
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["points"]) >= 2000
    for p in body["points"][:10]:
        assert p["z"] is not None
        z = float(p["z"])
        assert z == z  # not NaN
        # UMAP coords are typically O(10); cap to a generous sanity bound.
        assert -200.0 < z < 200.0


def test_A6_umap_3d_row_count_matches_2d(client: TestClient) -> None:
    """DEVPLAN acceptance: 3-D coords have the same row count as 2-D."""
    body_2d = client.get("/api/umap_points").json()
    body_3d = client.get("/api/umap_points", params={"n_dims": 3}).json()
    assert len(body_2d["points"]) == len(body_3d["points"])


def test_A6_umap_3d_row_order_matches_2d(client: TestClient) -> None:
    """The 3-D parquet's row order matches the 2-D parquet's by idx + label.

    The two parquets come from different UMAP fits but share the same
    test-thumb iteration order; row i in both files describes the
    same galaxy. The label_name should match per row.
    """
    body_2d = client.get("/api/umap_points").json()
    body_3d = client.get("/api/umap_points", params={"n_dims": 3}).json()
    for p2, p3 in zip(body_2d["points"], body_3d["points"], strict=True):
        assert p2["idx"] == p3["idx"]
        assert p2["label_name"] == p3["label_name"]


def test_A6_umap_3d_label_palette_intact(client: TestClient) -> None:
    """All three canonical smooth-or-featured labels appear in the 3-D view."""
    body = client.get("/api/umap_points", params={"n_dims": 3}).json()
    label_names = set(body["label_names"])
    assert {"smooth", "featured-or-disk", "artifact"}.issubset(label_names)


def test_A6_umap_invalid_n_dims_400(client: TestClient) -> None:
    response = client.get("/api/umap_points", params={"n_dims": 4})
    assert response.status_code == 400


def test_A6_umap_n_dims_5_400(client: TestClient) -> None:
    response = client.get("/api/umap_points", params={"n_dims": 5})
    assert response.status_code == 400
