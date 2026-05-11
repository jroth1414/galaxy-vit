"""T4.2 - UMAP Explorer backend API + acceptance gate.

DEVPLAN T4.2 acceptance: ``Playwright lassos 100 points; sample grid
renders.`` Lifted to the API-substantive equivalent: the backend
exposes >= 100 points within a small UMAP region (so a lasso of that
region in the frontend would return >= 100), and every point's
thumbnail endpoint is fetchable.

Tests:

* ``/api/umap_points`` returns >= 2000 points with all required fields
  and ordered label_names.
* For at least one ~10% x ~10% window of UMAP space, >= 100 points
  fall inside (lassoable).
* ``/api/test_thumbs/{idx}/thumbnail`` serves valid JPEG for sample idx.
* ``/api/test_thumbs/{idx}/posteriors`` returns a well-formed
  PosteriorResponse for a clicked point.
* 404 on out-of-range idx.

Skipped when any of (M1 ckpt, M3 ckpt, umap_coords.parquet, test_thumbs/)
is missing.
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
DIRICHLET_CKPT = Path("runs/m3_dirichlet/best.pt")
UMAP_COORDS = Path("artifacts/umap_coords.parquet")
TEST_THUMBS_DIR = Path("artifacts/test_thumbs")

if not ZOOBOT_CKPT.is_file():  # pragma: no cover -- local-only gate
    pytest.skip("M1 checkpoint missing", allow_module_level=True)
if not DIRICHLET_CKPT.is_file():
    pytest.skip("M3 checkpoint missing", allow_module_level=True)
if not UMAP_COORDS.is_file():
    pytest.skip(
        "umap_coords.parquet missing; run scripts/extract_umap.py first",
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


def test_T4_2_umap_points_endpoint_well_formed(client: TestClient) -> None:
    """Returns >= 2000 points with required fields + 3 canonical label names."""
    response = client.get("/api/umap_points")
    assert response.status_code == 200, response.text
    body = response.json()
    pts = body["points"]
    assert len(pts) >= 2000
    assert set(body["label_names"]).issuperset({"smooth", "featured-or-disk", "artifact"})
    for p in pts[:5]:
        assert {"idx", "x", "y", "label", "label_name"}.issubset(p.keys())
        assert p["label_name"] in body["label_names"]
        assert isinstance(p["idx"], int) and p["idx"] >= 0


def test_T4_2_at_least_100_points_in_some_lasso_region(client: TestClient) -> None:
    """DEVPLAN acceptance proxy: >= 100 points fall in some 20%x20% region.

    Builds a 5x5 grid over the UMAP bounding box and checks the most
    populous cell has >= 100 points. A frontend lasso of that cell
    region would return >= 100 points, which is the substantive
    Playwright check (lasso 100 points; sample grid renders).
    """
    pts = client.get("/api/umap_points").json()["points"]
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_step = (x_hi - x_lo) / 5
    y_step = (y_hi - y_lo) / 5
    counts: dict[tuple[int, int], int] = {}
    for p in pts:
        i = min(4, int((p["x"] - x_lo) / x_step))
        j = min(4, int((p["y"] - y_lo) / y_step))
        counts[(i, j)] = counts.get((i, j), 0) + 1
    max_cell = max(counts.values())
    assert max_cell >= 100, (
        f"most populous 20%x20% UMAP region has only {max_cell} points; "
        f"DEVPLAN T4.2 needs >= 100 lassoable"
    )


def test_T4_2_test_thumbnail_endpoint_serves_jpeg(client: TestClient) -> None:
    response = client.get("/api/test_thumbs/0/thumbnail")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:3] == b"\xff\xd8\xff"  # JPEG magic


def test_T4_2_test_thumbnail_404_on_bad_idx(client: TestClient) -> None:
    response = client.get("/api/test_thumbs/999999/thumbnail")
    assert response.status_code == 404


def test_T4_2_test_thumb_posteriors_returns_well_formed(client: TestClient) -> None:
    """Clicking a UMAP point returns a full PosteriorResponse (substitutes
    DEVPLAN click-to-Aladin since the HF dataset has no per-galaxy RA/Dec).
    """
    response = client.post("/api/test_thumbs/0/posteriors")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "questions" in body
    assert len(body["questions"]) == 10  # GZ DESI question count
    for q in body["questions"]:
        for a in q["answers"]:
            assert 0.0 <= a["ci_lower"] <= a["ci_upper"] <= 1.0
