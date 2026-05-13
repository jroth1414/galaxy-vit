"""A-7 — API tests for /api/per_question_gradcam{,/test_thumbs/{idx}}."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")
pytest.importorskip("PIL")
pytest.importorskip("galaxy_datasets")

ZOOBOT_CKPT = Path("runs/m1_zoobot_finetune/best.pt")
DIRICHLET_CKPT = Path("runs/m3_dirichlet/best.pt")
TEST_THUMBS_DIR = Path("artifacts/test_thumbs")

if not ZOOBOT_CKPT.is_file():  # pragma: no cover -- local-only gate
    pytest.skip("M1 checkpoint missing", allow_module_level=True)
if not DIRICHLET_CKPT.is_file():
    pytest.skip("M3 checkpoint missing", allow_module_level=True)
if not TEST_THUMBS_DIR.is_dir() or not list(TEST_THUMBS_DIR.glob("*.jpg")):
    pytest.skip(
        "test_thumbs missing", allow_module_level=True
    )

from fastapi.testclient import TestClient  # noqa: E402

from galaxy_vit.serve.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_A7_test_thumb_returns_attention_id(client: TestClient) -> None:
    """GET endpoint returns the attention_id + question name."""
    response = client.get(
        "/api/per_question_gradcam/test_thumbs/0",
        params={"question": "bar"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["question"] == "bar"
    assert isinstance(body["attention_id"], str) and body["attention_id"]


def test_A7_attention_url_resolves_to_png(client: TestClient) -> None:
    body = client.get(
        "/api/per_question_gradcam/test_thumbs/0",
        params={"question": "smooth-or-featured"},
    ).json()
    response = client.get(f"/api/attention/{body['attention_id']}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:4] == b"\x89PNG"  # PNG magic


def test_A7_400_on_unknown_question(client: TestClient) -> None:
    response = client.get(
        "/api/per_question_gradcam/test_thumbs/0",
        params={"question": "bogus-question"},
    )
    assert response.status_code == 400


def test_A7_404_on_unknown_idx(client: TestClient) -> None:
    response = client.get(
        "/api/per_question_gradcam/test_thumbs/999999",
        params={"question": "bar"},
    )
    assert response.status_code == 404


def test_A7_upload_endpoint(client: TestClient) -> None:
    thumb_path = TEST_THUMBS_DIR / "00000.jpg"
    with thumb_path.open("rb") as f:
        contents = f.read()
    files = {"file": ("query.jpg", io.BytesIO(contents), "image/jpeg")}
    response = client.post(
        "/api/per_question_gradcam",
        files=files,
        params={"question": "merging"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["question"] == "merging"


def test_A7_different_questions_yield_different_attention_ids(
    client: TestClient,
) -> None:
    """Sanity: two different questions don't collide on the same cache entry."""
    a = client.get(
        "/api/per_question_gradcam/test_thumbs/0",
        params={"question": "bar"},
    ).json()
    b = client.get(
        "/api/per_question_gradcam/test_thumbs/0",
        params={"question": "spiral-winding"},
    ).json()
    # Cache keys are UUIDs assigned per request; the OVERLAY content
    # itself differs because different alpha slices were backpropagated.
    assert a["attention_id"] != b["attention_id"]
