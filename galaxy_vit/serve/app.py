"""FastAPI server for the Galaxy-ViT live demo (T1.7).

Endpoints (all under ``/api`` per ARCHITECTURE.md §2.2):

* ``GET  /api/health``               -> HealthResponse
* ``POST /api/predict``              -> PredictResponse  (multipart upload)
* ``GET  /api/predict_sdss?ra&dec``  -> PredictResponse  (SDSS cutout)
* ``GET  /api/attention/{id}``       -> image/png        (cached overlay)

The classifier is loaded once in the lifespan startup hook from
``$GALAXY_VIT_CKPT`` (default ``runs/m1_zoobot_finetune/best.pt``) on
``$GALAXY_VIT_DEVICE`` (default ``cpu``, matching the HF Spaces target).
GradCAM overlays are cached in an in-memory FIFO bounded at
``ATTENTION_CACHE_MAX_SIZE`` entries; clients fetch them by the UUID
handle returned in the predict response.
"""

from __future__ import annotations

import io
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage_

from galaxy_vit.inference.predict import GalaxyClassifier
from galaxy_vit.serve.schemas import (
    GALAXY10_LABEL_NAMES,
    HealthResponse,
    PredictResponse,
    TopKItem,
)
from galaxy_vit.serve.sdss import SDSSError, fetch_sdss_cutout

DEFAULT_CKPT_PATH = Path("runs/m1_zoobot_finetune/best.pt")
DEFAULT_FRONTEND_DIST = Path("frontend/dist")
ATTENTION_CACHE_MAX_SIZE = 128
TOP_K = 3

_state: dict[str, Any] = {}


def _resolve_ckpt_path() -> Path:
    return Path(os.environ.get("GALAXY_VIT_CKPT", str(DEFAULT_CKPT_PATH)))


def _resolve_device() -> str:
    return os.environ.get("GALAXY_VIT_DEVICE", "cpu")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the classifier + initialise per-app state at startup."""
    classifier = GalaxyClassifier(_resolve_ckpt_path(), device=_resolve_device())
    _state["classifier"] = classifier
    _state["start_time"] = time.time()
    _state["attention_cache"] = OrderedDict[str, bytes]()
    try:
        yield
    finally:
        _state.clear()


app = FastAPI(
    title="Galaxy-ViT",
    description="Live demo backend for the Galaxy-ViT M1 baseline.",
    lifespan=lifespan,
)


def _classifier() -> GalaxyClassifier:
    classifier = _state.get("classifier")
    if classifier is None:
        raise HTTPException(
            status_code=503, detail="model not loaded yet; lifespan hasn't completed"
        )
    assert isinstance(classifier, GalaxyClassifier)
    return classifier


def _top_k_items(top: list[tuple[int, float]]) -> list[TopKItem]:
    return [
        TopKItem(
            class_id=cid,
            class_name=GALAXY10_LABEL_NAMES[cid],
            probability=p,
        )
        for cid, p in top
    ]


def _cache_attention(overlay: PILImage_.Image) -> str:
    """Save the overlay PNG bytes to the LRU cache; return its UUID handle."""
    aid = uuid.uuid4().hex
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    cache: OrderedDict[str, bytes] = _state["attention_cache"]
    cache[aid] = buf.getvalue()
    while len(cache) > ATTENTION_CACHE_MAX_SIZE:
        cache.popitem(last=False)  # evict oldest
    return aid


def _decode_image(contents: bytes) -> PILImage_.Image:
    try:
        image = PILImage_.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"could not decode image: {exc}",
        ) from exc
    assert isinstance(image, PILImage_.Image)
    return image


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    classifier = _classifier()
    return HealthResponse(
        ok=True,
        model_run_id=classifier.cfg.run_id,
        model_kind=classifier.cfg.model.kind,
        uptime_s=time.time() - _state["start_time"],
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict(file: Annotated[UploadFile, File()]) -> PredictResponse:
    classifier = _classifier()
    contents = await file.read()
    image = _decode_image(contents)
    top = classifier.predict(image, top_k=TOP_K)
    overlay = classifier.gradcam_overlay(image)
    aid = _cache_attention(overlay)
    return PredictResponse(top_k=_top_k_items(top), attention_id=aid)


@app.get("/api/predict_sdss", response_model=PredictResponse)
def predict_sdss(ra: float, dec: float) -> PredictResponse:
    if not (0.0 <= ra <= 360.0):
        raise HTTPException(
            status_code=400, detail=f"ra must be in [0, 360]; got {ra}"
        )
    if not (-90.0 <= dec <= 90.0):
        raise HTTPException(
            status_code=400, detail=f"dec must be in [-90, 90]; got {dec}"
        )
    classifier = _classifier()
    try:
        image = fetch_sdss_cutout(ra, dec)
    except SDSSError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    top = classifier.predict(image, top_k=TOP_K)
    overlay = classifier.gradcam_overlay(image)
    aid = _cache_attention(overlay)
    return PredictResponse(top_k=_top_k_items(top), attention_id=aid)


@app.get("/api/attention/{aid}")
def attention(aid: str) -> Response:
    cache = _state.get("attention_cache")
    if cache is None or aid not in cache:
        raise HTTPException(
            status_code=404,
            detail=f"attention overlay {aid!r} not in cache (may have been evicted)",
        )
    return Response(content=cache[aid], media_type="image/png")


# ------------------------------------------------------------------------ #
# Static mounts — registered AFTER all /api routes so explicit routes still
# win. /static serves the run dir (curves.png, run_config.json, the
# stratified interpretability overlays) for the model card; / serves the
# Vite-built SPA bundle from frontend/dist when present.
# ------------------------------------------------------------------------ #


def _mount_static() -> None:
    """Conditionally mount run artefacts at /static and the SPA at /.

    Both mounts are silently skipped if the source directory doesn't exist
    (e.g. frontend hasn't been built, or running tests without a run dir),
    so the API stays usable headless.
    """
    run_dir = _resolve_ckpt_path().parent
    if run_dir.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(run_dir)),
            name="static",
        )

    spa_dir = Path(os.environ.get("GALAXY_VIT_FRONTEND", str(DEFAULT_FRONTEND_DIST)))
    if spa_dir.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(spa_dir), html=True),
            name="spa",
        )


_mount_static()
