"""C-16 — API tests for /api/compare.

Skipped when either checkpoint is missing (the endpoint requires both
M1 and M3 to be loaded).
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")
pytest.importorskip("PIL")

ZOOBOT_CKPT = Path("runs/m1_zoobot_finetune/best.pt")
DIRICHLET_CKPT = Path("runs/m3_dirichlet/best.pt")
TEST_THUMBS_DIR = Path("artifacts/test_thumbs")

if not ZOOBOT_CKPT.is_file():  # pragma: no cover
    pytest.skip("M1 checkpoint missing", allow_module_level=True)
if not DIRICHLET_CKPT.is_file():
    pytest.skip("M3 checkpoint missing", allow_module_level=True)
if not TEST_THUMBS_DIR.is_dir() or not list(TEST_THUMBS_DIR.glob("*.jpg")):
    pytest.skip("test_thumbs missing", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from galaxy_vit.serve.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _post_compare(client: TestClient, jpg_path: Path) -> dict:
    with jpg_path.open("rb") as f:
        contents = f.read()
    files = {"file": ("query.jpg", io.BytesIO(contents), "image/jpeg")}
    response = client.post("/api/compare", files=files)
    assert response.status_code == 200, response.text
    return response.json()


def test_C16_compare_response_shape(client: TestClient) -> None:
    """Response wraps both an M1 PredictResponse and an M3 PosteriorResponse."""
    body = _post_compare(client, TEST_THUMBS_DIR / "00000.jpg")
    assert "m1" in body and "m3" in body

    m1 = body["m1"]
    assert {"top_k", "attention_id"} == set(m1.keys())
    assert isinstance(m1["top_k"], list) and len(m1["top_k"]) >= 1
    for item in m1["top_k"]:
        assert {"class_id", "class_name", "probability"} == set(item.keys())

    m3 = body["m3"]
    assert "questions" in m3
    assert len(m3["questions"]) == 10
    for q in m3["questions"]:
        for a in q["answers"]:
            assert 0.0 <= a["ci_lower"] <= a["ci_upper"] <= 1.0


def test_C16_compare_attention_url_resolves(client: TestClient) -> None:
    body = _post_compare(client, TEST_THUMBS_DIR / "00000.jpg")
    aid = body["m1"]["attention_id"]
    response = client.get(f"/api/attention/{aid}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_C16_compare_top_k_probabilities_sum_close_to_one(
    client: TestClient,
) -> None:
    """M1 top-3 sum is bounded above by 1 (softmax normalisation)."""
    body = _post_compare(client, TEST_THUMBS_DIR / "00000.jpg")
    total = sum(item["probability"] for item in body["m1"]["top_k"])
    # Top-3 of a 10-class softmax can sum to less than 1; just assert the
    # bound + that no individual probability is out of range.
    assert 0.0 < total <= 1.0
    for item in body["m1"]["top_k"]:
        assert 0.0 <= item["probability"] <= 1.0


def test_C16_compare_m3_smooth_or_featured_present(client: TestClient) -> None:
    """First M3 question is always smooth-or-featured (canonical schema)."""
    body = _post_compare(client, TEST_THUMBS_DIR / "00000.jpg")
    assert body["m3"]["questions"][0]["question"] == "smooth-or-featured"


def test_C16_compare_deterministic(client: TestClient) -> None:
    """Two calls on the same image return identical M1 + M3 numbers."""
    a = _post_compare(client, TEST_THUMBS_DIR / "00000.jpg")
    b = _post_compare(client, TEST_THUMBS_DIR / "00000.jpg")
    # GradCAM attention_id is a fresh UUID per request; everything
    # else should match exactly.
    assert a["m1"]["top_k"] == b["m1"]["top_k"]
    assert a["m3"]["questions"] == b["m3"]["questions"]
