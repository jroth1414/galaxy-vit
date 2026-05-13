"""Unit tests for the A-8 RA/Dec name resolver.

The Sesame call is mocked via ``httpx.MockTransport`` so the suite
stays offline (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are honoured by
the rest of the suite; this file asserts the resolver itself doesn't
escape to the network).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

httpx = pytest.importorskip("httpx")

from galaxy_vit.serve import sdss as sdss_mod  # noqa: E402
from galaxy_vit.serve.sdss import (  # noqa: E402
    NameResolutionError,
    _parse_sesame_ra_dec,
    resolve_object_name,
)

# ---------------------------------------------------------------------------
# Pure parser tests (no I/O)
# ---------------------------------------------------------------------------


def test_parse_sesame_returns_first_J_line() -> None:
    payload = """\
#=Simbad:    1
%@ 00 42 44.330 +41 16 09.39
%I.0 NAME M  31
%C.0 G
%J 10.6847082 +41.2691222 = ...
%J.E [ 0.001 0.001 ] = ...
"""
    parsed = _parse_sesame_ra_dec(payload)
    assert parsed is not None
    ra, dec = parsed
    assert ra == pytest.approx(10.6847082, abs=1e-6)
    assert dec == pytest.approx(41.2691222, abs=1e-6)


def test_parse_sesame_skips_J_E_error_block() -> None:
    """The %J.E line is uncertainty info; must not be parsed as coords."""
    payload = """\
%J 50.5 -10.0 = ...
%J.E [ 0.001 0.001 ]
"""
    parsed = _parse_sesame_ra_dec(payload)
    assert parsed == pytest.approx((50.5, -10.0))


def test_parse_sesame_returns_none_on_missing_J() -> None:
    payload = """#=Simbad:    0
#!Sesame: nothing found
"""
    assert _parse_sesame_ra_dec(payload) is None


def test_parse_sesame_returns_none_on_out_of_range() -> None:
    payload = "%J 999.0 -200.0 = bogus\n"
    assert _parse_sesame_ra_dec(payload) is None


# ---------------------------------------------------------------------------
# Mocked end-to-end resolver
# ---------------------------------------------------------------------------


@pytest.fixture
def mocked_sesame_m31(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Patch httpx.get to return a canned Sesame response for M31."""

    def fake_get(url: str, **_kw: object) -> httpx.Response:
        body = (
            "#=Simbad:    1\n"
            "%J 10.6847 +41.2691 = ...\n"
            "%J.E [ 0.001 0.001 ]\n"
        )
        return httpx.Response(200, text=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(sdss_mod.httpx, "get", fake_get)
    # Clear the LRU cache so other tests in the same process don't
    # leak resolved-name state in.
    resolve_object_name.cache_clear()
    yield
    resolve_object_name.cache_clear()


def test_resolve_object_name_m31(mocked_sesame_m31: None) -> None:
    """DEVPLAN: 'M31' resolves to RA ≈ 10.68, Dec ≈ 41.27 (within 0.01°)."""
    ra, dec = resolve_object_name("M31")
    assert ra == pytest.approx(10.68, abs=0.01)
    assert dec == pytest.approx(41.27, abs=0.01)


def test_resolve_object_name_uses_cache(mocked_sesame_m31: None) -> None:
    """LRU cache means a second call doesn't re-fetch."""
    a = resolve_object_name("M31")
    b = resolve_object_name("M31")
    assert a == b
    info = resolve_object_name.cache_info()
    assert info.hits >= 1


def test_resolve_object_name_strips_whitespace(
    mocked_sesame_m31: None,
) -> None:
    """Leading/trailing whitespace doesn't bypass the cache."""
    ra1, _ = resolve_object_name("M31")
    ra2, _ = resolve_object_name("  M31  ")
    assert ra1 == ra2


def test_resolve_object_name_rejects_empty() -> None:
    with pytest.raises(NameResolutionError, match="empty"):
        resolve_object_name("")


def test_resolve_object_name_unresolved(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sesame returns no %J line -> NameResolutionError."""

    def fake_get(url: str, **_kw: object) -> httpx.Response:
        return httpx.Response(
            200,
            text="#=Simbad:    0\n#!Sesame: nothing found\n",
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(sdss_mod.httpx, "get", fake_get)
    resolve_object_name.cache_clear()
    try:
        with pytest.raises(NameResolutionError, match="did not resolve"):
            resolve_object_name("BogusObject_xyz")
    finally:
        resolve_object_name.cache_clear()


def test_resolve_object_name_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent HTTP error -> NameResolutionError after retries."""

    def fake_get(url: str, **_kw: object) -> httpx.Response:
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(sdss_mod.httpx, "get", fake_get)
    resolve_object_name.cache_clear()
    try:
        with pytest.raises(NameResolutionError, match="failed after"):
            resolve_object_name("any-name")
    finally:
        resolve_object_name.cache_clear()
