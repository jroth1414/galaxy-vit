"""A-5 — API tests for /api/tree_flow + /api/tree_flow/test_thumbs/{idx}."""

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
        "test_thumbs missing; run scripts/build_test_thumbs.py first",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient  # noqa: E402

from galaxy_vit.serve.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_A5_tree_flow_test_thumb_response_shape(client: TestClient) -> None:
    """Endpoint returns 44 nodes (10 questions + 34 answers) with required fields."""
    response = client.get("/api/tree_flow/test_thumbs/0")
    assert response.status_code == 200, response.text
    body = response.json()
    nodes = body["nodes"]
    assert len(nodes) == 44
    question_nodes = [n for n in nodes if n["kind"] == "question"]
    answer_nodes = [n for n in nodes if n["kind"] == "answer"]
    assert len(question_nodes) == 10
    assert len(answer_nodes) == 34
    for n in nodes[:5]:
        assert {
            "id",
            "label",
            "kind",
            "question",
            "answer",
            "reach",
            "parent_question",
            "parent_answer",
        } == set(n.keys())
        assert 0.0 <= n["reach"] <= 1.0


def test_A5_tree_flow_test_thumb_always_asked_have_reach_one(
    client: TestClient,
) -> None:
    body = client.get("/api/tree_flow/test_thumbs/0").json()
    by_id = {n["id"]: n for n in body["nodes"]}
    assert by_id["q:smooth-or-featured"]["reach"] == pytest.approx(1.0, abs=1e-6)
    assert by_id["q:merging"]["reach"] == pytest.approx(1.0, abs=1e-6)


def test_A5_tree_flow_test_thumb_monotonic(client: TestClient) -> None:
    """Every node's reach is bounded above by its parent's reach."""
    body = client.get("/api/tree_flow/test_thumbs/0").json()
    by_id = {n["id"]: n for n in body["nodes"]}
    for n in body["nodes"]:
        if n["kind"] == "answer":
            parent = by_id[f"q:{n['question']}"]
            assert n["reach"] <= parent["reach"] + 1e-6


def test_A5_tree_flow_test_thumb_404_on_unknown_idx(client: TestClient) -> None:
    response = client.get("/api/tree_flow/test_thumbs/999999")
    assert response.status_code == 404


def test_A5_tree_flow_upload_endpoint(client: TestClient) -> None:
    """POST /api/tree_flow with the idx-0 thumbnail returns the same shape."""
    thumb_path = TEST_THUMBS_DIR / "00000.jpg"
    with thumb_path.open("rb") as f:
        contents = f.read()
    files = {"file": ("query.jpg", io.BytesIO(contents), "image/jpeg")}
    response = client.post("/api/tree_flow", files=files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["nodes"]) == 44
