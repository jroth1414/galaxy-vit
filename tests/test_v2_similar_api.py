"""S-1 — API tests for /api/similar/{idx} and /api/similar (upload).

Skipped when any of (M1 ckpt, M3 ckpt, test_thumbs/, test_thumb_features
parquet) is missing. Mirrors the gating pattern in
``test_t4_2_umap_explorer_api.py`` so we can drop this file into CI
once those artifacts are precomputed by the local pipeline.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")
pytest.importorskip("pandas")
pytest.importorskip("PIL")

ZOOBOT_CKPT = Path("runs/m1_zoobot_finetune/best.pt")
DIRICHLET_CKPT = Path("runs/m3_dirichlet/best.pt")
TEST_THUMBS_DIR = Path("artifacts/test_thumbs")
SIMILAR_FEATURES = Path("artifacts/test_thumb_features.parquet")

if not ZOOBOT_CKPT.is_file():  # pragma: no cover -- local-only gate
    pytest.skip("M1 checkpoint missing", allow_module_level=True)
if not DIRICHLET_CKPT.is_file():
    pytest.skip("M3 checkpoint missing", allow_module_level=True)
if not TEST_THUMBS_DIR.is_dir() or not list(TEST_THUMBS_DIR.glob("*.jpg")):
    pytest.skip(
        "test_thumbs missing; run scripts/build_test_thumbs.py first",
        allow_module_level=True,
    )
if not SIMILAR_FEATURES.is_file():
    pytest.skip(
        "test_thumb_features.parquet missing; "
        "run `python -m scripts.cache_test_thumb_features` first",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient  # noqa: E402

from galaxy_vit.serve.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_S1_similar_by_idx_returns_self_first(client: TestClient) -> None:
    """`/api/similar/0?k=20` returns idx 0 first with distance ≈ 0."""
    response = client.get("/api/similar/0", params={"k": 20})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["query_idx"] == 0
    hits = body["hits"]
    assert len(hits) == 20
    assert hits[0]["idx"] == 0
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-5)


def test_S1_similar_by_idx_response_shape(client: TestClient) -> None:
    """Each hit has idx, distance, thumbnail_url; bounds + URL format hold."""
    response = client.get("/api/similar/0", params={"k": 5})
    hits = response.json()["hits"]
    for h in hits:
        assert {"idx", "distance", "thumbnail_url"} == set(h.keys())
        assert isinstance(h["idx"], int) and h["idx"] >= 0
        assert 0.0 <= h["distance"] <= 2.0
        assert h["thumbnail_url"] == f"/api/test_thumbs/{h['idx']}/thumbnail"


def test_S1_similar_by_idx_is_deterministic(client: TestClient) -> None:
    """Two calls with the same idx + k return identical hits."""
    a = client.get("/api/similar/3", params={"k": 10}).json()["hits"]
    b = client.get("/api/similar/3", params={"k": 10}).json()["hits"]
    assert [(h["idx"], h["distance"]) for h in a] == [
        (h["idx"], h["distance"]) for h in b
    ]


def test_S1_similar_distances_sorted_ascending(client: TestClient) -> None:
    hits = client.get("/api/similar/5", params={"k": 15}).json()["hits"]
    distances = [h["distance"] for h in hits]
    assert distances == sorted(distances)


def test_S1_similar_clamps_k_to_max(client: TestClient) -> None:
    """k > SIMILAR_MAX_K is clamped silently (no 400)."""
    response = client.get("/api/similar/0", params={"k": 9999})
    assert response.status_code == 200
    hits = response.json()["hits"]
    # SIMILAR_MAX_K is 60 in the app constants.
    assert len(hits) <= 60


def test_S1_similar_rejects_zero_k(client: TestClient) -> None:
    response = client.get("/api/similar/0", params={"k": 0})
    assert response.status_code == 400


def test_S1_similar_404_on_bad_idx(client: TestClient) -> None:
    response = client.get("/api/similar/999999", params={"k": 5})
    assert response.status_code == 404


def test_S1_similar_upload_endpoint(client: TestClient) -> None:
    """POST /api/similar with the idx-0 thumbnail returns idx 0 ≈ first."""
    thumb_path = TEST_THUMBS_DIR / "00000.jpg"
    with thumb_path.open("rb") as f:
        contents = f.read()
    files = {"file": ("query.jpg", io.BytesIO(contents), "image/jpeg")}
    response = client.post("/api/similar", files=files, params={"k": 10})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["query_idx"] is None  # uploads don't have a cache idx
    hits = body["hits"]
    assert len(hits) == 10
    # The uploaded file is bit-for-bit identical to test_thumbs/00000.jpg,
    # so idx 0 should be the closest hit with distance ≈ 0.
    assert hits[0]["idx"] == 0
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-3)


def test_S1_similar_thumbnail_url_resolves(client: TestClient) -> None:
    """The thumbnail_url returned by /api/similar resolves to a valid JPEG."""
    hits = client.get("/api/similar/0", params={"k": 3}).json()["hits"]
    thumb = client.get(hits[0]["thumbnail_url"])
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"
