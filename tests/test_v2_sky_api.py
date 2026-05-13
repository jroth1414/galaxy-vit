"""S-2 — API tests for /api/sky_points.

Skipped when artifacts/sky_points.parquet (or any prerequisite) is
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
SKY_POINTS = Path("artifacts/sky_points.parquet")

if not ZOOBOT_CKPT.is_file():  # pragma: no cover -- local-only gate
    pytest.skip("M1 checkpoint missing", allow_module_level=True)
if not SKY_POINTS.is_file():
    pytest.skip(
        "sky_points.parquet missing; run "
        "`python -m scripts.build_sky_points` first",
        allow_module_level=True,
    )

from fastapi.testclient import TestClient  # noqa: E402

from galaxy_vit.serve.app import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_S2_sky_points_returns_at_least_10k(client: TestClient) -> None:
    """DEVPLAN acceptance: >=10,000 entries (~14k expected)."""
    response = client.get("/api/sky_points")
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["points"]) >= 10_000


def test_S2_sky_points_response_shape(client: TestClient) -> None:
    body = client.get("/api/sky_points").json()
    for p in body["points"][:5]:
        assert {"dr8_id", "ra", "dec", "label", "label_name", "entropy"} == set(
            p.keys()
        )
        assert isinstance(p["dr8_id"], str) and p["dr8_id"]
        assert isinstance(p["label"], int) and p["label"] >= 0
        assert isinstance(p["entropy"], float) and p["entropy"] >= 0.0


def test_S2_sky_ra_dec_in_valid_range(client: TestClient) -> None:
    """DEVPLAN acceptance: every point has ra in [0, 360], dec in [-90, 90]."""
    body = client.get("/api/sky_points").json()
    for p in body["points"]:
        assert 0.0 <= p["ra"] <= 360.0, f"ra out of range: {p['ra']}"
        assert -90.0 <= p["dec"] <= 90.0, f"dec out of range: {p['dec']}"


def test_S2_sky_label_palette_intact(client: TestClient) -> None:
    """All three canonical labels appear in the response palette."""
    body = client.get("/api/sky_points").json()
    label_names = set(body["label_names"])
    assert {"smooth", "featured-or-disk", "artifact"}.issubset(label_names)


def test_S2_sky_dr8_id_unique(client: TestClient) -> None:
    """No duplicate dr8_id across the joined catalog."""
    body = client.get("/api/sky_points").json()
    ids = [p["dr8_id"] for p in body["points"]]
    assert len(ids) == len(set(ids)), "duplicate dr8_id in /api/sky_points"


def test_S2_sky_entropy_finite(client: TestClient) -> None:
    """Predictive entropy is always finite (no NaN/inf leaks from the build)."""
    import math

    body = client.get("/api/sky_points").json()
    for p in body["points"][:1000]:
        assert math.isfinite(p["entropy"]), p
