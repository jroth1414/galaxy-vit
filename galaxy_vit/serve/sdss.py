"""SDSS SkyServer DR18 cutout client (T1.7).

Used by ``/api/predict_sdss?ra&dec`` to fetch a JPEG cutout for a sky
position and feed it to the classifier. Per ARCHITECTURE.md §2.2:

* ``functools.lru_cache(maxsize=1024)`` — same (ra, dec, scale, w, h)
  request hits cache after the first call.
* Exponential backoff on transient HTTP errors. SkyServer's informal
  rate limit is ~100 req/min, so the cache + backoff matter.

Sync httpx (not async) because the calling endpoints already run in a
thread pool — keeps the integration with FastAPI's threadpool simple.
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


class SDSSError(RuntimeError):
    """Raised when SkyServer can't be reached or returns an unexpected payload."""


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
