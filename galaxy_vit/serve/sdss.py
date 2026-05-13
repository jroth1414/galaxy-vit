"""SDSS SkyServer DR18 cutout client + Sesame name resolver (T1.7 + A-8).

Used by ``/api/predict_sdss?ra&dec`` to fetch a JPEG cutout for a sky
position and feed it to the classifier. Per ARCHITECTURE.md §2.2:

* ``functools.lru_cache(maxsize=1024)`` — same (ra, dec, scale, w, h)
  request hits cache after the first call.
* Exponential backoff on transient HTTP errors. SkyServer's informal
  rate limit is ~100 req/min, so the cache + backoff matter.

Sync httpx (not async) because the calling endpoints already run in a
thread pool — keeps the integration with FastAPI's threadpool simple.

A-8 adds :func:`resolve_object_name`: a tiny client against the CDS
Sesame name-resolver service. Plain-text Sesame output, parsed with
no external astropy / astroquery dependency.
"""

from __future__ import annotations

import time
from functools import lru_cache
from io import BytesIO
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from PIL.Image import Image as PILImage

SDSS_CUTOUT_URL = "https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg"
DEFAULT_SCALE_ARCSEC_PIX = 0.396  # SDSS native pixel scale
DEFAULT_WIDTH = 256
DEFAULT_HEIGHT = 256
DEFAULT_TIMEOUT_S = 10.0
MAX_RETRIES = 3

# CDS Sesame name resolver. `-oI/A` selects plain-text "info" mode
# across all services (Simbad first, then VizieR, then NED); `%J`
# lines carry RA + Dec in degrees. Documented at:
# https://cds.u-strasbg.fr/cgi-bin/Sesame
SESAME_URL = "https://cdsweb.u-strasbg.fr/cgi-bin/nph-sesame/-oI/A"
SESAME_TIMEOUT_S = 8.0
SESAME_MAX_RETRIES = 2


class SDSSError(RuntimeError):
    """Raised when SkyServer can't be reached or returns an unexpected payload."""


class NameResolutionError(RuntimeError):
    """Raised when the Sesame service can't resolve an object name."""


@lru_cache(maxsize=1024)
def fetch_sdss_cutout(
    ra: float,
    dec: float,
    *,
    scale: float = DEFAULT_SCALE_ARCSEC_PIX,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> PILImage:
    """Return a PIL RGB image fetched from SkyServer ImgCutout/getjpeg.

    Cached on (ra, dec, scale, width, height); re-asks at most ~1024
    distinct positions before evicting LRU. On HTTP error / timeout the
    call retries up to ``MAX_RETRIES`` times with exponential backoff
    (0.5, 1.0, 2.0 s) and raises :class:`SDSSError` if all attempts fail.
    """
    from PIL import Image as PILImage_

    params = {
        "ra": ra,
        "dec": dec,
        "scale": scale,
        "width": width,
        "height": height,
    }
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = httpx.get(SDSS_CUTOUT_URL, params=params, timeout=timeout)
            response.raise_for_status()
            return PILImage_.open(BytesIO(response.content)).convert("RGB")
        except (httpx.HTTPError, OSError) as exc:
            last_err = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.5 * (2**attempt))
    raise SDSSError(
        f"SDSS cutout fetch failed after {MAX_RETRIES} attempts: {last_err}"
    ) from last_err


def _parse_sesame_ra_dec(payload: str) -> tuple[float, float] | None:
    """Pull the first ``%J <ra> <dec>`` pair (degrees) out of Sesame text output.

    Sesame's plain-text format includes lines like::

        %J 10.6847082 +41.2691222 = ... (J2000)
        %J.E [ 0.001 0.001 ] = ...

    We accept the bare ``%J `` lines (note the trailing space which
    distinguishes them from the ``%J.E`` error block); the first
    match wins. Returns ``None`` if no usable line is found.
    """
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line.startswith("%J "):
            continue
        # Tokens after the "%J " prefix: ra dec ... (sometimes with
        # trailing "=" and equinox info). Take the first two.
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            ra = float(parts[1])
            dec = float(parts[2])
        except ValueError:
            continue
        if not (0.0 <= ra <= 360.0) or not (-90.0 <= dec <= 90.0):
            continue
        return ra, dec
    return None


@lru_cache(maxsize=512)
def resolve_object_name(name: str) -> tuple[float, float]:
    """A-8: resolve an astronomical object name to (RA, Dec) in degrees.

    Hits the CDS Sesame web service which queries Simbad, VizieR and
    NED in order. Returns the FIRST successful resolution. Cached on
    name to absorb the demo's repeat queries.

    Raises :class:`NameResolutionError` when the service is
    unreachable or the name can't be resolved.
    """
    name_clean = name.strip()
    if not name_clean:
        raise NameResolutionError("empty name")

    import urllib.parse as _up

    # Sesame takes the name in the URL path (after the "?", but as a
    # bare query string, not a "key=value" pair). Percent-encode so
    # names with spaces ("NGC 1300") survive.
    url = f"{SESAME_URL}?{_up.quote(name_clean)}"

    last_err: Exception | None = None
    for attempt in range(SESAME_MAX_RETRIES):
        try:
            response = httpx.get(
                url,
                timeout=SESAME_TIMEOUT_S,
                headers={"User-Agent": "galaxy-vit/1.0"},
            )
            response.raise_for_status()
            parsed = _parse_sesame_ra_dec(response.text)
            if parsed is not None:
                return parsed
            raise NameResolutionError(
                f"Sesame did not resolve {name_clean!r}"
            )
        except NameResolutionError:
            raise
        except (httpx.HTTPError, OSError) as exc:
            last_err = exc
            if attempt < SESAME_MAX_RETRIES - 1:
                time.sleep(0.5 * (2**attempt))
    raise NameResolutionError(
        f"Sesame query failed after {SESAME_MAX_RETRIES} attempts: {last_err}"
    ) from last_err
