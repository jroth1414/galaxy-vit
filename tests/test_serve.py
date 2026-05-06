"""T1.7 — FastAPI server acceptance tests.

The whole module skips if torch / transformers / fastapi aren't
installed (CI runs ``[dev]`` only) or if no Zoobot checkpoint exists
(``runs/m1_zoobot_finetune/best.pt`` — produced by the T1.5 trainer
run). Locally with ``[m1-train, m1-serve, torch-cu128]`` installed
and the checkpoint built, all four tests run.

Coverage:

* ``/api/health`` returns 200 with a well-formed body
* ``/api/predict`` accepts a multipart upload, returns top-3 + a
  cached attention overlay handle that ``/api/attention/{id}`` then
  serves as a valid PNG
* ``/api/predict_sdss`` succeeds when the SDSS cutout fetch is mocked
  to return a synthetic image
* p95 latency for ``/api/predict`` is below the 800 ms acceptance gate
  (DEVPLAN T1.7) on whatever device the classifier was loaded on
  (typically CPU; locally GPU on dev machines)
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")

ZOOBOT_CKPT = Path("runs/m1_zoobot_finetune/best.pt")
if not ZOOBOT_CKPT.is_file():  # pragma: no cover — local-only gate
    pytest.skip(
        "checkpoint not built: run "
        "`python -m galaxy_vit.training.trainer "
        "--config configs/m1_zoobot_finetune.yaml` first",
        allow_module_level=True,
    )

from collections.abc import Iterator  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image as PILImage_  # noqa: E402

from galaxy_vit.serve.app import ATTENTION_CACHE_MAX_SIZE, app  # noqa: E402

LATENCY_P95_THRESHOLD_S = 0.800  # DEVPLAN T1.7
N_LATENCY_SAMPLES = 20


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """TestClient with FastAPI lifespan: model loads once for the module."""
    with TestClient(app) as c:
        yield c


def _make_synthetic_image_bytes(
    *, size: tuple[int, int] = (256, 256), color: tuple[int, int, int] = (40, 40, 100)
) -> bytes:
    """Return PNG bytes for a small uniform RGB image."""
    img = PILImage_.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_returns_ok_with_run_metadata(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert isinstance(body["model_run_id"], str) and body["model_run_id"]
    assert body["model_kind"] in {"vit_baseline", "zoobot_convnext"}
    assert body["uptime_s"] >= 0.0


def test_predict_returns_top3_and_caches_attention(client: TestClient) -> None:
    img_bytes = _make_synthetic_image_bytes()
    resp = client.post(
        "/api/predict",
        files={"file": ("synthetic.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Top-k structure.
    assert len(body["top_k"]) == 3
    seen_classes: set[int] = set()
    for item in body["top_k"]:
        assert 0 <= item["class_id"] <= 9
        assert item["class_id"] not in seen_classes  # no duplicates
        seen_classes.add(item["class_id"])
        assert 0.0 <= item["probability"] <= 1.0
        assert isinstance(item["class_name"], str) and item["class_name"]
    # Top-3 probabilities should be in descending order.
    probs = [item["probability"] for item in body["top_k"]]
    assert probs == sorted(probs, reverse=True), probs

    # Attention round-trip: the returned UUID resolves to a PNG.
    aid = body["attention_id"]
    assert isinstance(aid, str) and len(aid) > 0
    overlay_resp = client.get(f"/api/attention/{aid}")
    assert overlay_resp.status_code == 200
    assert overlay_resp.headers["content-type"] == "image/png"
    overlay_img = PILImage_.open(io.BytesIO(overlay_resp.content))
    assert overlay_img.mode == "RGB"
    assert overlay_img.size == (256, 256)


def test_predict_sdss_with_mocked_cutout(client: TestClient) -> None:
    """SDSS endpoint succeeds when the cutout fetch is mocked."""
    fake_image = PILImage_.new("RGB", (256, 256), (50, 80, 120))

    with patch(
        "galaxy_vit.serve.app.fetch_sdss_cutout", return_value=fake_image
    ) as mock_fetch:
        resp = client.get("/api/predict_sdss", params={"ra": 184.6, "dec": 47.2})
    assert resp.status_code == 200, resp.text
    mock_fetch.assert_called_once()
    body = resp.json()
    assert len(body["top_k"]) == 3
    assert body["attention_id"]


def test_predict_p95_latency_under_800ms(client: TestClient) -> None:
    """T1.7 acceptance: p95 of /api/predict is below 800 ms."""
    img_bytes = _make_synthetic_image_bytes()
    timings: list[float] = []
    for _ in range(N_LATENCY_SAMPLES):
        start = time.perf_counter()
        resp = client.post(
            "/api/predict",
            files={"file": ("synthetic.png", img_bytes, "image/png")},
        )
        timings.append(time.perf_counter() - start)
        assert resp.status_code == 200
    timings.sort()
    p95_index = max(0, int(0.95 * len(timings)) - 1)
    p95 = timings[p95_index]
    median = timings[len(timings) // 2]
    print(
        f"[latency] median={median * 1000:.1f}ms  p95={p95 * 1000:.1f}ms  "
        f"(n={N_LATENCY_SAMPLES})"
    )
    assert p95 < LATENCY_P95_THRESHOLD_S, (
        f"p95 {p95 * 1000:.1f}ms exceeds {LATENCY_P95_THRESHOLD_S * 1000:.0f}ms threshold"
    )


def test_attention_cache_bounded(client: TestClient) -> None:
    """The attention LRU cache evicts oldest entries past its size cap."""
    img_bytes = _make_synthetic_image_bytes()

    # Fire enough predicts to overflow the cache.
    first_aid: str | None = None
    for i in range(ATTENTION_CACHE_MAX_SIZE + 5):
        resp = client.post(
            "/api/predict",
            files={"file": (f"s{i}.png", img_bytes, "image/png")},
        )
        if i == 0:
            first_aid = resp.json()["attention_id"]

    # The very first attention id should be evicted.
    assert first_aid is not None
    overlay_resp = client.get(f"/api/attention/{first_aid}")
    assert overlay_resp.status_code == 404
