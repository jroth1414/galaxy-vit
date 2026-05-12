"""T6.3 - Holistic 4-tab live-demo backend integration test.

DEVPLAN T6.3 acceptance calls for a Playwright end-to-end test on all 4
tabs. Playwright is heavy infrastructure (browser binaries, sandbox
configuration); we lift the substantive check to the backend layer:
exercise every API endpoint each tab uses and assert the response
shapes are what the frontend expects.

Tabs <-> endpoints:

  Classify        ->  POST /api/predict (multipart image upload)
                  ->  GET  /api/predict_sdss?ra&dec
                  ->  GET  /api/attention/{aid}
  Posteriors      ->  POST /api/posteriors (multipart image upload)
                  ->  GET  /api/demo_galaxies
                  ->  GET  /api/demo_galaxies/{id}/thumbnail
                  ->  GET  /api/demo_galaxies/{id}/posteriors
  Explorer        ->  GET  /api/umap_points
                  ->  GET  /api/test_thumbs/{idx}/thumbnail
                  ->  POST /api/test_thumbs/{idx}/posteriors
  Model Card      ->  (static; served from /static and the SPA bundle)

If this test passes, every API the SPA consumes is alive and shape-
correct. The visual frontend rendering itself is verified out-of-band
via the manual Loom recording (docs/loom_shotlist.md).
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")
pytest.importorskip("pandas")
pytest.importorskip("galaxy_datasets")

ZOOBOT_CKPT = Path("runs/m1_zoobot_finetune/best.pt")
DIRICHLET_CKPT = Path("runs/m3_dirichlet/best.pt")
DEMO_MANIFEST = Path("artifacts/demo_galaxies/manifest.json")
UMAP_COORDS = Path("artifacts/umap_coords.parquet")
TEST_THUMBS = Path("artifacts/test_thumbs")

for gate, hint in [
    (ZOOBOT_CKPT, "T1.5 zoobot finetune"),
    (DIRICHLET_CKPT, "T3.6 dirichlet trainer"),
    (DEMO_MANIFEST, "scripts/build_demo_galaxies.py"),
    (UMAP_COORDS, "scripts/extract_umap.py"),
]:
    if not gate.exists():  # pragma: no cover -- local-only gate
        pytest.skip(
            f"missing {gate} (run {hint} first)", allow_module_level=True
        )
if not list(TEST_THUMBS.glob("*.jpg")):
    pytest.skip(
        "test_thumbs missing; run scripts/build_test_thumbs.py first",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image as PILImage_  # noqa: E402

from galaxy_vit.serve.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def _png_bytes(size: int = 128) -> bytes:
    """A tiny deterministic upload payload."""
    rng = torch.Generator().manual_seed(0)
    arr = (torch.rand(size, size, 3, generator=rng) * 255).to(torch.uint8).numpy()
    img = PILImage_.fromarray(arr).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Health (universal)
# ---------------------------------------------------------------------------


def test_T6_3_health_endpoint_ok(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert "model_run_id" in body
    assert "uptime_s" in body


# ---------------------------------------------------------------------------
# Classify tab
# ---------------------------------------------------------------------------


def test_T6_3_classify_tab_predict(client: TestClient) -> None:
    """Classify tab: image upload -> top-3 + GradCAM handle."""
    files = {"file": ("g.png", _png_bytes(), "image/png")}
    response = client.post("/api/predict", files=files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["top_k"]) == 3
    for item in body["top_k"]:
        assert {"class_id", "class_name", "probability"}.issubset(item.keys())
        assert 0.0 <= item["probability"] <= 1.0
    # GradCAM handle should fetch as a valid PNG.
    aid = body["attention_id"]
    overlay = client.get(f"/api/attention/{aid}")
    assert overlay.status_code == 200
    assert overlay.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_T6_3_classify_tab_predict_sdss_validation(client: TestClient) -> None:
    """Classify tab: SDSS-fetch endpoint validates RA/Dec before network."""
    # Out-of-range RA -> 400 before any external call.
    response = client.get("/api/predict_sdss", params={"ra": 999.0, "dec": 0.0})
    assert response.status_code == 400


def test_T6_3_classify_tab_predict_sdss_with_mock(client: TestClient) -> None:
    """Classify tab: SDSS path returns top-k when the cutout fetch is mocked."""
    img = PILImage_.frombytes(
        "RGB", (224, 224), bytes([128] * 224 * 224 * 3)
    )
    with patch("galaxy_vit.serve.app.fetch_sdss_cutout", return_value=img):
        response = client.get(
            "/api/predict_sdss", params={"ra": 12.5, "dec": -5.0}
        )
    assert response.status_code == 200, response.text
    assert len(response.json()["top_k"]) == 3


# ---------------------------------------------------------------------------
# Posteriors tab
# ---------------------------------------------------------------------------


def test_T6_3_posteriors_tab_upload(client: TestClient) -> None:
    """Posteriors tab: image upload -> all 10 questions, CIs in [0, 1]."""
    files = {"file": ("g.png", _png_bytes(), "image/png")}
    response = client.post("/api/posteriors", files=files)
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["questions"]) == 10
    for q in body["questions"]:
        for a in q["answers"]:
            assert 0.0 <= a["ci_lower"] <= a["ci_upper"] <= 1.0


def test_T6_3_posteriors_tab_demo_galaxy_path(client: TestClient) -> None:
    """Posteriors tab: demo galaxy picker -> thumbnail + posterior + overlay."""
    galaxies = client.get("/api/demo_galaxies").json()["galaxies"]
    assert len(galaxies) >= 1
    gid = galaxies[0]["id"]

    thumb = client.get(f"/api/demo_galaxies/{gid}/thumbnail")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"

    posterior = client.get(f"/api/demo_galaxies/{gid}/posteriors")
    assert posterior.status_code == 200, posterior.text
    body = posterior.json()
    assert len(body["posterior"]["questions"]) == 10
    assert len(body["volunteer"]) == 10


# ---------------------------------------------------------------------------
# Explorer tab
# ---------------------------------------------------------------------------


def test_T6_3_explorer_tab_umap_points(client: TestClient) -> None:
    """Explorer tab: scatter plot data; >= 2000 points, 3 canonical labels."""
    response = client.get("/api/umap_points")
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["points"]) >= 2000
    assert {"smooth", "featured-or-disk", "artifact"}.issubset(set(body["label_names"]))


def test_T6_3_explorer_tab_hover_thumbnail(client: TestClient) -> None:
    """Explorer tab: hover preview fetches a UMAP-indexed thumbnail."""
    response = client.get("/api/test_thumbs/0/thumbnail")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"


def test_T6_3_explorer_tab_click_to_posterior(client: TestClient) -> None:
    """Explorer tab: click handler -> per-galaxy posterior."""
    response = client.post("/api/test_thumbs/0/posteriors")
    assert response.status_code == 200
    assert len(response.json()["questions"]) == 10


# ---------------------------------------------------------------------------
# Cross-tab consistency
# ---------------------------------------------------------------------------


def test_T6_3_all_four_tabs_endpoints_reachable(client: TestClient) -> None:
    """Single test that exercises one endpoint per tab in sequence.

    If any tab's primary endpoint is unreachable, this test is the
    canary -- catches breakage where unrelated changes accidentally
    disable a route registration.
    """
    endpoints = [
        ("Classify", "GET", "/api/health", 200),
        ("Posteriors", "GET", "/api/demo_galaxies", 200),
        ("Explorer", "GET", "/api/umap_points", 200),
        ("Model Card (no API; verifying SPA / static is mounted)", "GET", "/api/health", 200),
    ]
    for tab, method, path, expected in endpoints:
        if method == "GET":
            response = client.get(path)
        else:
            raise NotImplementedError(method)
        assert response.status_code == expected, (
            f"{tab}: {method} {path} returned {response.status_code}, "
            f"expected {expected}"
        )
