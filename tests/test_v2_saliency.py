"""S-4 — Endpoint smoke + acceptance gates for /api/test_thumbs/{idx}/saliency.

Skipped when the precomputed saliency directory is empty / missing,
mirroring the gating pattern in the other test_v2_*.py files.
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
TEST_THUMBS_DIR = Path("artifacts/test_thumbs")
TEST_SALIENCIES_DIR = Path("artifacts/test_saliencies")

if not ZOOBOT_CKPT.is_file():  # pragma: no cover -- local-only gate
    pytest.skip("M1 checkpoint missing", allow_module_level=True)
if not TEST_THUMBS_DIR.is_dir() or not list(TEST_THUMBS_DIR.glob("*.jpg")):
    pytest.skip(
        "test_thumbs missing; run scripts/build_test_thumbs.py first",
        allow_module_level=True,
    )
if not TEST_SALIENCIES_DIR.is_dir() or not list(
    TEST_SALIENCIES_DIR.glob("*.jpg")
):
    pytest.skip(
        "test_saliencies missing; run "
        "`python -m scripts.build_test_saliencies` first",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient  # noqa: E402

from galaxy_vit.serve.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_S4_saliency_count_matches_thumbs() -> None:
    """DEVPLAN acceptance: 2,462 saliency JPEGs (one per UMAP-set thumbnail)."""
    n_thumbs = len(list(TEST_THUMBS_DIR.glob("*.jpg")))
    n_sals = len(list(TEST_SALIENCIES_DIR.glob("*.jpg")))
    assert n_sals == n_thumbs, (
        f"saliency count ({n_sals}) != thumbnail count ({n_thumbs}); "
        "build_test_saliencies.py iteration drifted from build_test_thumbs.py"
    )


def test_S4_saliency_endpoint_serves_jpeg(client: TestClient) -> None:
    """`/api/test_thumbs/0/saliency` returns a valid JPEG."""
    response = client.get("/api/test_thumbs/0/saliency")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:3] == b"\xff\xd8\xff"  # JPEG magic


def test_S4_saliency_endpoint_size_matches_thumbnail(client: TestClient) -> None:
    """The saliency overlay shares pixel dimensions with the source thumb.

    The Explorer hover crossfade stacks the two `<img>` tags absolute-
    positioned over each other; same dims = pixel-perfect overlay.
    """
    from PIL import Image

    sal_bytes = client.get("/api/test_thumbs/0/saliency").content
    thumb_bytes = client.get("/api/test_thumbs/0/thumbnail").content
    sal_img = Image.open(io.BytesIO(sal_bytes))
    thumb_img = Image.open(io.BytesIO(thumb_bytes))
    assert sal_img.size == thumb_img.size, (
        f"saliency {sal_img.size} != thumbnail {thumb_img.size}; "
        "Explorer hover crossfade will be misaligned"
    )


def test_S4_saliency_404_on_negative_idx(client: TestClient) -> None:
    response = client.get("/api/test_thumbs/-1/saliency")
    # FastAPI parses the negative int but the handler 404s on negatives.
    # (FastAPI may also 422 on path-param parsing failures; either is
    # an acceptable "not a valid index" signal.)
    assert response.status_code in (404, 422)


def test_S4_saliency_404_on_unknown_idx(client: TestClient) -> None:
    response = client.get("/api/test_thumbs/999999/saliency")
    assert response.status_code == 404


def test_S4_saliency_is_nontrivial(client: TestClient) -> None:
    """A real GradCAM overlay isn't a constant-color image.

    Computes pixel std-dev across the green channel; a flat overlay
    (which would be the all-zero edge case) has std=0, while a real
    blended heatmap has noticeably nonzero variation.
    """
    from PIL import Image

    sal_bytes = client.get("/api/test_thumbs/0/saliency").content
    img = Image.open(io.BytesIO(sal_bytes)).convert("RGB")
    pixels = list(img.getdata())
    greens = [p[1] for p in pixels]
    mean = sum(greens) / len(greens)
    var = sum((g - mean) ** 2 for g in greens) / len(greens)
    std = var**0.5
    assert std > 5.0, (
        f"green-channel std={std:.2f} suggests a flat / corrupt overlay"
    )
