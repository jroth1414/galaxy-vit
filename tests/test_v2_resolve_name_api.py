"""A-8 — API tests for /api/resolve_name.

The Sesame call inside resolve_object_name is monkeypatched at
module scope so this test stays offline.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from galaxy_vit.serve import sdss as sdss_mod  # noqa: E402
from galaxy_vit.serve.app import app  # noqa: E402
from galaxy_vit.serve.sdss import resolve_object_name  # noqa: E402


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def mock_sesame(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    def fake_get(url: str, **_kw: object) -> httpx.Response:
        # Return M31's coords for any name; tests below only exercise
        # the contract, not the catalog content.
        return httpx.Response(
            200,
            text="#=Simbad: 1\n%J 10.6847 +41.2691 = ...\n",
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(sdss_mod.httpx, "get", fake_get)
    resolve_object_name.cache_clear()
    yield
    resolve_object_name.cache_clear()


def test_A8_resolve_decimal_coords_short_circuits(client: TestClient) -> None:
    """'10.68 41.27' parses as decimal coords without hitting Sesame."""
    response = client.get("/api/resolve_name", params={"name": "10.68 41.27"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source"] == "coords"
    assert body["ra"] == pytest.approx(10.68, abs=1e-4)
    assert body["dec"] == pytest.approx(41.27, abs=1e-4)


def test_A8_resolve_decimal_coords_comma_separated(client: TestClient) -> None:
    """'10.68, 41.27' also parses as coords."""
    response = client.get("/api/resolve_name", params={"name": "10.68, 41.27"})
    assert response.status_code == 200, response.text
    assert response.json()["source"] == "coords"


def test_A8_resolve_name_uses_sesame(client: TestClient) -> None:
    """A non-numeric name falls through to the Sesame mock."""
    response = client.get("/api/resolve_name", params={"name": "M31"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["source"] == "sesame"
    assert body["ra"] == pytest.approx(10.68, abs=1e-2)
    assert body["dec"] == pytest.approx(41.27, abs=1e-2)


def test_A8_resolve_invalid_coords_falls_through_to_sesame(
    client: TestClient,
) -> None:
    """RA out of range isn't valid coords; should be sent to Sesame."""
    # The mock returns M31's coords for everything; 999 999 doesn't
    # parse as coords, so the request hits Sesame and gets a hit.
    response = client.get("/api/resolve_name", params={"name": "999 999"})
    assert response.status_code == 200
    assert response.json()["source"] == "sesame"


def test_A8_resolve_unknown_name_returns_404(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    """An unresolvable name returns 404 with a clear error."""

    def fake_get(url: str, **_kw: object) -> httpx.Response:
        return httpx.Response(
            200,
            text="#=Simbad:    0\n#!Sesame: nothing found\n",
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(sdss_mod.httpx, "get", fake_get)
    resolve_object_name.cache_clear()

    response = client.get(
        "/api/resolve_name", params={"name": "totally-unknown-zzz"}
    )
    assert response.status_code == 404
