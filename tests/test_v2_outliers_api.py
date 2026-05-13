"""S-3 — API tests for /api/outliers.

Skipped when artifacts/outliers.json (or any prerequisite) is missing,
mirroring the gating pattern in tests/test_v2_similar_api.py.
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
OUTLIERS_PATH = Path("artifacts/outliers.json")
TEST_THUMBS_DIR = Path("artifacts/test_thumbs")

if not ZOOBOT_CKPT.is_file():  # pragma: no cover -- local-only gate
    pytest.skip("M1 checkpoint missing", allow_module_level=True)
if not DIRICHLET_CKPT.is_file():
    pytest.skip("M3 checkpoint missing", allow_module_level=True)
if not OUTLIERS_PATH.is_file():
    pytest.skip(
        "outliers.json missing; run "
        "`python -m scripts.build_outlier_indices` first",
        allow_module_level=True,
    )
if not TEST_THUMBS_DIR.is_dir() or not list(TEST_THUMBS_DIR.glob("*.jpg")):
    pytest.skip(
        "test_thumbs missing; run scripts/build_test_thumbs.py first",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient  # noqa: E402

from galaxy_vit.serve.app import app  # noqa: E402

METRICS = ("entropy", "bald", "disagreement")


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_S3_outliers_default_metric_is_entropy(client: TestClient) -> None:
    response = client.get("/api/outliers", params={"k": 5})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["metric"] == "entropy"
    assert isinstance(body["median"], float)
    assert len(body["items"]) == 5


@pytest.mark.parametrize("metric", METRICS)
def test_S3_outliers_response_shape(client: TestClient, metric: str) -> None:
    """Each metric returns top-K with idx, value, thumbnail_url."""
    response = client.get("/api/outliers", params={"metric": metric, "k": 10})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["metric"] == metric
    items = body["items"]
    assert 1 <= len(items) <= 10
    for it in items:
        assert {"idx", "value", "thumbnail_url"} == set(it.keys())
        assert isinstance(it["idx"], int) and it["idx"] >= 0
        assert it["thumbnail_url"] == f"/api/test_thumbs/{it['idx']}/thumbnail"


@pytest.mark.parametrize("metric", METRICS)
def test_S3_outliers_sorted_descending(client: TestClient, metric: str) -> None:
    items = client.get(
        "/api/outliers", params={"metric": metric, "k": 20}
    ).json()["items"]
    values = [it["value"] for it in items]
    assert values == sorted(values, reverse=True)


@pytest.mark.parametrize("metric", METRICS)
def test_S3_outliers_top_value_above_median(client: TestClient, metric: str) -> None:
    """DEVPLAN acceptance: outlier galaxies have visibly higher metric than median."""
    body = client.get(
        "/api/outliers", params={"metric": metric, "k": 5}
    ).json()
    top_value = body["items"][0]["value"]
    median = body["median"]
    assert top_value > median, (
        f"top {metric}={top_value} is not strictly greater than "
        f"median={median}; outlier list is not actually selecting outliers"
    )


@pytest.mark.parametrize("metric", METRICS)
def test_S3_outliers_is_deterministic(client: TestClient, metric: str) -> None:
    a = client.get("/api/outliers", params={"metric": metric, "k": 10}).json()
    b = client.get("/api/outliers", params={"metric": metric, "k": 10}).json()
    assert a == b


def test_S3_outliers_clamps_k(client: TestClient) -> None:
    """k beyond OUTLIER_MAX_K is silently clamped (no 400)."""
    body = client.get("/api/outliers", params={"k": 9999}).json()
    assert len(body["items"]) <= 100  # OUTLIER_MAX_K


def test_S3_outliers_400_on_unknown_metric(client: TestClient) -> None:
    response = client.get("/api/outliers", params={"metric": "bogus"})
    assert response.status_code == 400


def test_S3_outliers_400_on_zero_k(client: TestClient) -> None:
    response = client.get("/api/outliers", params={"k": 0})
    assert response.status_code == 400


def test_S3_outlier_thumbnail_url_resolves(client: TestClient) -> None:
    items = client.get("/api/outliers", params={"k": 1}).json()["items"]
    thumb = client.get(items[0]["thumbnail_url"])
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"
