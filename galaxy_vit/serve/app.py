"""FastAPI server for the Galaxy-ViT live demo (T1.7, extended at T4.3).

Endpoints (all under ``/api`` per ARCHITECTURE.md §2.2):

* ``GET  /api/health``                            -> HealthResponse
* ``POST /api/predict``                           -> PredictResponse  (Galaxy10 plurality + GradCAM)
* ``GET  /api/predict_sdss?ra&dec``               -> PredictResponse  (SDSS cutout)
* ``GET  /api/attention/{id}``                    -> image/png        (cached overlay)
* ``POST /api/posteriors``                        -> PosteriorResponse (T4.3 Dirichlet posteriors)
* ``GET  /api/demo_galaxies``                     -> DemoGalaxiesResponse
* ``GET  /api/demo_galaxies/{id}/posteriors``     -> DemoGalaxyPosteriorResponse
* ``GET  /api/demo_galaxies/{id}/thumbnail``      -> image/jpeg

The Galaxy10 classifier is loaded from ``$GALAXY_VIT_CKPT``
(default ``runs/m1_zoobot_finetune/best.pt``). The Dirichlet predictor
is loaded from ``$GALAXY_VIT_DIRICHLET_CKPT`` (default
``runs/m3_dirichlet/best.pt``) with optional temperature calibration
from ``$GALAXY_VIT_DIRICHLET_CAL`` (default the calibrated_metrics.json
next to the checkpoint). Both load lazily-once at lifespan startup;
their absence is logged but doesn't crash the server (the corresponding
endpoints 503 instead).
"""

from __future__ import annotations

import io
import json
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage_

from galaxy_vit.inference.dirichlet_predictor import (
    DirichletPosteriorPredictor,
    posterior_to_payload,
)
from galaxy_vit.inference.predict import GalaxyClassifier
from galaxy_vit.inference.similarity import (
    SimilarityIndex,
    encode_image_to_feature,
)
from galaxy_vit.serve.schemas import (
    GALAXY10_LABEL_NAMES,
    DemoGalaxiesResponse,
    DemoGalaxyItem,
    DemoGalaxyPosteriorResponse,
    HealthResponse,
    OutlierItem,
    OutliersResponse,
    PosteriorResponse,
    PredictResponse,
    SimilarGalaxiesResponse,
    SimilarGalaxyItem,
    SkyPoint,
    SkyPointsResponse,
    TopKItem,
    UMAPPoint,
    UMAPPointsResponse,
    VolunteerOverlayItem,
)
from galaxy_vit.serve.sdss import SDSSError, fetch_sdss_cutout

DEFAULT_CKPT_PATH = Path("runs/m1_zoobot_finetune/best.pt")
DEFAULT_DIRICHLET_CKPT = Path("runs/m3_dirichlet/best.pt")
DEFAULT_DIRICHLET_CAL = Path("runs/m3_dirichlet/calibrated_metrics.json")
DEFAULT_DEMO_GALAXIES_DIR = Path("artifacts/demo_galaxies")
DEFAULT_UMAP_COORDS = Path("artifacts/umap_coords.parquet")
DEFAULT_UMAP_3D_COORDS = Path("artifacts/umap_3d_coords.parquet")
DEFAULT_TEST_THUMBS = Path("artifacts/test_thumbs")
DEFAULT_TEST_SALIENCIES = Path("artifacts/test_saliencies")
DEFAULT_SIMILAR_FEATURES = Path("artifacts/test_thumb_features.parquet")
DEFAULT_OUTLIERS = Path("artifacts/outliers.json")
DEFAULT_SKY_POINTS = Path("artifacts/sky_points.parquet")
DEFAULT_FRONTEND_DIST = Path("frontend/dist")
ATTENTION_CACHE_MAX_SIZE = 128
TOP_K = 3
SIMILAR_DEFAULT_K = 20
SIMILAR_MAX_K = 60
OUTLIER_METRICS = ("entropy", "bald", "disagreement")
OUTLIER_DEFAULT_K = 20
OUTLIER_MAX_K = 100

_state: dict[str, Any] = {}


def _resolve_ckpt_path() -> Path:
    return Path(os.environ.get("GALAXY_VIT_CKPT", str(DEFAULT_CKPT_PATH)))


def _resolve_device() -> str:
    return os.environ.get("GALAXY_VIT_DEVICE", "cpu")


def _resolve_dirichlet_ckpt() -> Path:
    return Path(os.environ.get("GALAXY_VIT_DIRICHLET_CKPT", str(DEFAULT_DIRICHLET_CKPT)))


def _resolve_dirichlet_cal() -> Path:
    return Path(os.environ.get("GALAXY_VIT_DIRICHLET_CAL", str(DEFAULT_DIRICHLET_CAL)))


def _resolve_demo_galaxies_dir() -> Path:
    return Path(
        os.environ.get("GALAXY_VIT_DEMO_GALAXIES", str(DEFAULT_DEMO_GALAXIES_DIR))
    )


def _resolve_umap_coords() -> Path:
    return Path(os.environ.get("GALAXY_VIT_UMAP_COORDS", str(DEFAULT_UMAP_COORDS)))


def _resolve_umap_3d_coords() -> Path:
    return Path(
        os.environ.get("GALAXY_VIT_UMAP_3D_COORDS", str(DEFAULT_UMAP_3D_COORDS))
    )


def _resolve_test_thumbs() -> Path:
    return Path(os.environ.get("GALAXY_VIT_TEST_THUMBS", str(DEFAULT_TEST_THUMBS)))


def _resolve_test_saliencies() -> Path:
    return Path(
        os.environ.get(
            "GALAXY_VIT_TEST_SALIENCIES", str(DEFAULT_TEST_SALIENCIES)
        )
    )


def _resolve_similar_features() -> Path:
    return Path(
        os.environ.get(
            "GALAXY_VIT_SIMILAR_FEATURES", str(DEFAULT_SIMILAR_FEATURES)
        )
    )


def _resolve_outliers_path() -> Path:
    return Path(os.environ.get("GALAXY_VIT_OUTLIERS", str(DEFAULT_OUTLIERS)))


def _resolve_sky_points() -> Path:
    return Path(os.environ.get("GALAXY_VIT_SKY_POINTS", str(DEFAULT_SKY_POINTS)))


def _try_load_dirichlet() -> DirichletPosteriorPredictor | None:
    """Load the Dirichlet predictor if the checkpoint exists; otherwise None.

    Absence is fine -- the /api/posteriors endpoints will 503 with a
    clear message. Lets the original Galaxy10 endpoints keep working
    on hosts that haven't run T3.6 yet.

    Calibration is OPT-IN at serve time, even when the calibrated_metrics
    JSON is present. Reasoning: the T3.6 best.pt aims at coverage = 0.95
    via a relatively large temperature (T=60 on v1), which widens CIs to
    the point of losing per-image discriminability. For the live-demo
    posteriors tab we want CIs that visibly differ across galaxies, so
    we default to raw alpha. Set GALAXY_VIT_DIRICHLET_USE_CAL=1 to opt
    back into the calibrated regime (e.g. for the model-card coverage
    figures).
    """
    ckpt = _resolve_dirichlet_ckpt()
    if not ckpt.is_file():
        return None
    use_cal = os.environ.get("GALAXY_VIT_DIRICHLET_USE_CAL", "").lower() in (
        "1", "true", "yes",
    )
    cal_path: Path | None = None
    if use_cal:
        cal = _resolve_dirichlet_cal()
        cal_path = cal if cal.is_file() else None
    try:
        return DirichletPosteriorPredictor(
            ckpt, calibrated_metrics_path=cal_path, device=_resolve_device()
        )
    except Exception as exc:
        # Don't crash the whole server; log and let the endpoint 503.
        print(f"[serve] failed to load Dirichlet predictor: {exc}", flush=True)
        return None


def _try_load_demo_manifest() -> list[dict[str, Any]] | None:
    """Read artifacts/demo_galaxies/manifest.json if present."""
    path = _resolve_demo_galaxies_dir() / "manifest.json"
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(loaded, list)
        return loaded
    except Exception as exc:
        print(f"[serve] failed to load demo galaxies manifest: {exc}", flush=True)
        return None


def _try_load_umap_points() -> list[dict[str, Any]] | None:
    """Read artifacts/umap_coords.parquet into a list of {idx, x, y, label, label_name}."""
    path = _resolve_umap_coords()
    if not path.is_file():
        return None
    try:
        import pandas as pd

        df = pd.read_parquet(path)
        records: list[dict[str, Any]] = []
        for i, row in enumerate(df.itertuples(index=False)):
            records.append(
                {
                    "idx": i,
                    "x": float(row.umap_x),
                    "y": float(row.umap_y),
                    "z": None,
                    "label": int(row.smooth_or_featured_label),
                    "label_name": str(row.smooth_or_featured_name),
                }
            )
        return records
    except Exception as exc:
        print(f"[serve] failed to load umap_coords.parquet: {exc}", flush=True)
        return None


def _try_load_umap_3d_points() -> list[dict[str, Any]] | None:
    """A-6: read artifacts/umap_3d_coords.parquet into the same record shape.

    Adds a populated ``z`` field. Row idx is the parquet row order
    which mirrors the 2-D parquet (and ``artifacts/test_thumbs/<idx>.jpg``)
    by construction in ``scripts/extract_umap_3d.py``.
    """
    path = _resolve_umap_3d_coords()
    if not path.is_file():
        return None
    try:
        import pandas as pd

        df = pd.read_parquet(path)
        records: list[dict[str, Any]] = []
        for i, row in enumerate(df.itertuples(index=False)):
            records.append(
                {
                    "idx": i,
                    "x": float(row.umap_x),
                    "y": float(row.umap_y),
                    "z": float(row.umap_z),
                    "label": int(row.smooth_or_featured_label),
                    "label_name": str(row.smooth_or_featured_name),
                }
            )
        return records
    except Exception as exc:
        print(f"[serve] failed to load umap_3d_coords.parquet: {exc}", flush=True)
        return None


def _try_load_similarity_index() -> SimilarityIndex | None:
    """Load test_thumb_features.parquet into an in-memory kNN index.

    Absence is fine -- /api/similar/* endpoints will 503 with a clear
    "run scripts/cache_test_thumb_features.py" hint.
    """
    path = _resolve_similar_features()
    if not path.is_file():
        return None
    try:
        return SimilarityIndex.from_parquet(path)
    except Exception as exc:
        print(
            f"[serve] failed to load test_thumb_features.parquet: {exc}",
            flush=True,
        )
        return None


def _try_load_outliers() -> dict[str, Any] | None:
    """Load artifacts/outliers.json (S-3) into memory at startup.

    Absence is fine -- /api/outliers will 503 with a clear hint.
    """
    path = _resolve_outliers_path()
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(loaded, dict)
        return loaded
    except Exception as exc:
        print(f"[serve] failed to load outliers.json: {exc}", flush=True)
        return None


def _try_load_sky_points() -> list[dict[str, Any]] | None:
    """S-2: load artifacts/sky_points.parquet into memory at startup.

    The full 14k row payload is ~1 MB JSON-serialised and kept in
    memory after the first read; the Sky tab fetches it once on
    mount.
    """
    path = _resolve_sky_points()
    if not path.is_file():
        return None
    try:
        import pandas as pd

        df = pd.read_parquet(path)
        records: list[dict[str, Any]] = []
        for row in df.itertuples(index=False):
            records.append(
                {
                    "dr8_id": str(row.dr8_id),
                    "ra": float(row.ra),
                    "dec": float(row.dec),
                    "label": int(row.smooth_or_featured_label),
                    "label_name": str(row.smooth_or_featured_name),
                    "entropy": float(row.entropy),
                }
            )
        return records
    except Exception as exc:
        print(f"[serve] failed to load sky_points.parquet: {exc}", flush=True)
        return None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the classifier + initialise per-app state at startup."""
    classifier = GalaxyClassifier(_resolve_ckpt_path(), device=_resolve_device())
    _state["classifier"] = classifier
    _state["dirichlet"] = _try_load_dirichlet()
    _state["demo_manifest"] = _try_load_demo_manifest()
    _state["umap_points"] = _try_load_umap_points()
    _state["umap_3d_points"] = _try_load_umap_3d_points()
    _state["similarity_index"] = _try_load_similarity_index()
    _state["outliers"] = _try_load_outliers()
    _state["sky_points"] = _try_load_sky_points()
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
# T4.3 -- Multi-question Dirichlet posterior endpoints
# ------------------------------------------------------------------------ #


def _dirichlet() -> DirichletPosteriorPredictor:
    pred = _state.get("dirichlet")
    if pred is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Dirichlet predictor not loaded. Set $GALAXY_VIT_DIRICHLET_CKPT "
                "to a valid runs/<id>/best.pt produced by the T3.6 trainer."
            ),
        )
    assert isinstance(pred, DirichletPosteriorPredictor)
    return pred


def _demo_manifest() -> list[dict[str, Any]]:
    manifest = _state.get("demo_manifest")
    if manifest is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Demo-galaxies manifest not loaded. Run "
                "`python -m scripts.build_demo_galaxies` to produce "
                "artifacts/demo_galaxies/manifest.json."
            ),
        )
    assert isinstance(manifest, list)
    return manifest


def _posterior_response_from_image(image: PILImage_.Image) -> PosteriorResponse:
    pred = _dirichlet()
    posteriors = pred.predict_posterior(image)
    payload = posterior_to_payload(posteriors)
    return PosteriorResponse.model_validate(
        {
            "questions": payload,
            "calibration_regime": pred.calibration_regime,
            "temperature": pred.temperature,
        }
    )


@app.post("/api/posteriors", response_model=PosteriorResponse)
async def posteriors(file: Annotated[UploadFile, File()]) -> PosteriorResponse:
    contents = await file.read()
    image = _decode_image(contents)
    return _posterior_response_from_image(image)


@app.get("/api/demo_galaxies", response_model=DemoGalaxiesResponse)
def demo_galaxies() -> DemoGalaxiesResponse:
    manifest = _demo_manifest()
    items = [
        DemoGalaxyItem(
            id=g["id"],
            smooth_or_featured_plurality=g["smooth_or_featured_plurality"],
            thumbnail_url=f"/api/demo_galaxies/{g['id']}/thumbnail",
        )
        for g in manifest
    ]
    return DemoGalaxiesResponse(galaxies=items)


@app.get("/api/demo_galaxies/{galaxy_id}/thumbnail")
def demo_galaxy_thumbnail(galaxy_id: str) -> FileResponse:
    _demo_manifest()  # 503 if missing
    thumb_path = _resolve_demo_galaxies_dir() / "thumbs" / f"{galaxy_id}.jpg"
    if not thumb_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"demo galaxy {galaxy_id!r} thumbnail not found"
        )
    return FileResponse(thumb_path, media_type="image/jpeg")


def _volunteer_overlay_for(entry: dict[str, Any]) -> list[VolunteerOverlayItem]:
    """Build the per-question volunteer-overlay items from a manifest entry."""
    from galaxy_vit.data.schema import question_index_groups

    out: list[VolunteerOverlayItem] = []
    counts_flat = entry["counts"]
    valid_flat = entry["valid"]
    for q_idx, (q_name, start, end) in enumerate(question_index_groups()):
        counts_q = [int(c) for c in counts_flat[start:end]]
        total = sum(counts_q)
        fractions = (
            [c / total for c in counts_q] if total > 0 else [0.0] * len(counts_q)
        )
        out.append(
            VolunteerOverlayItem(
                question=q_name,
                valid=bool(valid_flat[q_idx]),
                fractions=fractions,
            )
        )
    return out


@app.get(
    "/api/demo_galaxies/{galaxy_id}/posteriors",
    response_model=DemoGalaxyPosteriorResponse,
)
def demo_galaxy_posteriors(galaxy_id: str) -> DemoGalaxyPosteriorResponse:
    manifest = _demo_manifest()
    entry = next((g for g in manifest if g["id"] == galaxy_id), None)
    if entry is None:
        raise HTTPException(
            status_code=404, detail=f"demo galaxy {galaxy_id!r} not in manifest"
        )
    thumb_path = _resolve_demo_galaxies_dir() / "thumbs" / f"{galaxy_id}.jpg"
    if not thumb_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"demo galaxy {galaxy_id!r} thumbnail missing"
        )
    image = PILImage_.open(thumb_path).convert("RGB")
    posterior = _posterior_response_from_image(image)
    volunteer = _volunteer_overlay_for(entry)
    return DemoGalaxyPosteriorResponse(posterior=posterior, volunteer=volunteer)


# ------------------------------------------------------------------------ #
# T4.2 -- Interactive UMAP Explorer endpoints
# ------------------------------------------------------------------------ #


def _umap_points(n_dims: int) -> list[dict[str, Any]]:
    if n_dims not in (2, 3):
        raise HTTPException(
            status_code=400, detail=f"n_dims must be 2 or 3; got {n_dims}"
        )
    state_key = "umap_points" if n_dims == 2 else "umap_3d_points"
    pts = _state.get(state_key)
    if pts is None:
        if n_dims == 2:
            hint = (
                "Generate artifacts/umap_coords.parquet by running "
                "`python -m scripts.extract_umap` (T2.4)."
            )
        else:
            hint = (
                "Generate artifacts/umap_3d_coords.parquet by running "
                "`python -m scripts.extract_umap_3d` (A-6)."
            )
        raise HTTPException(
            status_code=503,
            detail=f"UMAP coords (n_dims={n_dims}) not loaded. {hint}",
        )
    assert isinstance(pts, list)
    return pts


@app.get("/api/umap_points", response_model=UMAPPointsResponse)
def umap_points(n_dims: int = 2) -> UMAPPointsResponse:
    """A-6: return 2-D or 3-D UMAP coords for the test-thumb set.

    ``?n_dims=3`` returns the same idx layout (one row per
    ``test_thumbs/<idx>.jpg``) with an additional ``z`` field
    populated from the dedicated 3-D fit.
    """
    pts = _umap_points(n_dims)
    label_names: dict[int, str] = {}
    for p in pts:
        label_names[int(p["label"])] = str(p["label_name"])
    ordered = [label_names[i] for i in sorted(label_names)]
    return UMAPPointsResponse(
        points=[UMAPPoint.model_validate(p) for p in pts],
        label_names=ordered,
    )


@app.get("/api/test_thumbs/{idx}/thumbnail")
def test_thumb(idx: int) -> FileResponse:
    if idx < 0:
        raise HTTPException(status_code=404, detail=f"bad thumbnail index {idx}")
    thumb_path = _resolve_test_thumbs() / f"{idx:05d}.jpg"
    if not thumb_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"test thumbnail {idx} not found"
        )
    return FileResponse(thumb_path, media_type="image/jpeg")


@app.get("/api/test_thumbs/{idx}/saliency")
def test_thumb_saliency(idx: int) -> FileResponse:
    """S-4: GradCAM overlay JPEG for the Explorer tab's hover crossfade.

    Saliencies are precomputed once by ``scripts/build_test_saliencies.py``;
    serving them is a static file lookup. Returns 404 when the overlay
    is missing (the demo can fall back to the plain thumbnail).
    """
    if idx < 0:
        raise HTTPException(status_code=404, detail=f"bad saliency index {idx}")
    sal_path = _resolve_test_saliencies() / f"{idx:05d}.jpg"
    if not sal_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"saliency {idx} not found; run "
                "`python -m scripts.build_test_saliencies` to precompute them"
            ),
        )
    return FileResponse(sal_path, media_type="image/jpeg")


@app.post("/api/test_thumbs/{idx}/posteriors", response_model=PosteriorResponse)
def test_thumb_posteriors(idx: int) -> PosteriorResponse:
    """Compute the model's posterior for a UMAP-set thumbnail by index.

    Lets the Explorer tab show a full posterior breakdown for any
    clicked point in the scatter plot (substitutes the DEVPLAN
    'click-to-Aladin' feature -- the HF dataset doesn't ship RA/Dec
    per galaxy so click-to-Aladin isn't implementable here).
    """
    thumb_path = _resolve_test_thumbs() / f"{idx:05d}.jpg"
    if not thumb_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"test thumbnail {idx} not found"
        )
    image = PILImage_.open(thumb_path).convert("RGB")
    return _posterior_response_from_image(image)


# ------------------------------------------------------------------------ #
# S-1 -- Similar-galaxy kNN endpoints
# ------------------------------------------------------------------------ #


def _similarity_index() -> SimilarityIndex:
    idx = _state.get("similarity_index")
    if idx is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Similarity index not loaded. Generate "
                "artifacts/test_thumb_features.parquet by running "
                "`python -m scripts.cache_test_thumb_features` (S-1)."
            ),
        )
    assert isinstance(idx, SimilarityIndex)
    return idx


def _clamp_k(k: int) -> int:
    if k < 1:
        raise HTTPException(status_code=400, detail=f"k must be >= 1; got {k}")
    return min(k, SIMILAR_MAX_K)


def _build_similar_items(
    hits: list[Any],
) -> list[SimilarGalaxyItem]:
    """Project SimilarityHit dataclasses into Pydantic items with thumb URLs."""
    return [
        SimilarGalaxyItem(
            idx=h.idx,
            distance=h.distance,
            thumbnail_url=f"/api/test_thumbs/{h.idx}/thumbnail",
        )
        for h in hits
    ]


@app.get("/api/similar/{idx}", response_model=SimilarGalaxiesResponse)
def similar_by_index(idx: int, k: int = SIMILAR_DEFAULT_K) -> SimilarGalaxiesResponse:
    """kNN where the query is the cached feature at ``idx``.

    Returns ``idx`` itself as the first hit (distance ≈ 0); the remaining
    K-1 hits are the morphology-nearest test-set galaxies.
    """
    index = _similarity_index()
    k = _clamp_k(k)
    if idx < 0 or idx >= index.n_items:
        raise HTTPException(
            status_code=404,
            detail=f"idx {idx} out of range [0, {index.n_items})",
        )
    hits = index.topk_by_index(idx, k=k)
    return SimilarGalaxiesResponse(
        query_idx=idx, hits=_build_similar_items(hits)
    )


@app.post("/api/similar", response_model=SimilarGalaxiesResponse)
async def similar_upload(
    file: Annotated[UploadFile, File()],
    k: int = SIMILAR_DEFAULT_K,
) -> SimilarGalaxiesResponse:
    """kNN where the query is an uploaded image (encoded through Zoobot).

    Uses the same M3 Dirichlet encoder + eval transform that built the
    cache, so feature space is consistent.
    """
    index = _similarity_index()
    k = _clamp_k(k)
    predictor = _dirichlet()
    contents = await file.read()
    image = _decode_image(contents)
    feat = encode_image_to_feature(
        predictor.model, predictor.transform, image, device=predictor.device
    )
    hits = index.topk(feat, k=k)
    return SimilarGalaxiesResponse(
        query_idx=None, hits=_build_similar_items(hits)
    )


# ------------------------------------------------------------------------ #
# S-2 -- Sky map endpoint
# ------------------------------------------------------------------------ #


def _sky_points() -> list[dict[str, Any]]:
    pts = _state.get("sky_points")
    if pts is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Sky points not loaded. Generate "
                "artifacts/sky_points.parquet by running "
                "`python -m scripts.build_sky_points` (S-2)."
            ),
        )
    assert isinstance(pts, list)
    return pts


@app.get("/api/sky_points", response_model=SkyPointsResponse)
def sky_points() -> SkyPointsResponse:
    """Joined inference + volunteer catalog for the Sky tab.

    Each point carries ra/dec/dr8_id plus the model's predicted
    smooth-or-featured plurality and the per-galaxy predictive
    entropy (so the frontend can color by either uncertainty or class).

    Click handling is delegated to ``/api/predict_sdss?ra&dec`` for
    the Aladin sub-view and to a future ``/api/posteriors_by_dr8_id``
    endpoint for the scatter sub-view.
    """
    pts = _sky_points()
    label_names: dict[int, str] = {}
    for p in pts:
        label_names[int(p["label"])] = str(p["label_name"])
    ordered = [label_names[i] for i in sorted(label_names)]
    return SkyPointsResponse(
        points=[SkyPoint.model_validate(p) for p in pts],
        label_names=ordered,
    )


# ------------------------------------------------------------------------ #
# S-3 -- Outliers ("most interesting galaxies") endpoint
# ------------------------------------------------------------------------ #


def _outliers_payload() -> dict[str, Any]:
    payload = _state.get("outliers")
    if payload is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Outlier rankings not loaded. Generate "
                "artifacts/outliers.json by running "
                "`python -m scripts.build_outlier_indices` (S-3)."
            ),
        )
    assert isinstance(payload, dict)
    return payload


@app.get("/api/outliers", response_model=OutliersResponse)
def outliers(metric: str = "entropy", k: int = OUTLIER_DEFAULT_K) -> OutliersResponse:
    """Return the top-K most-outlier-y galaxies for a chosen metric.

    Metrics:

    * ``entropy``      -- predictive entropy summed across questions.
    * ``bald``         -- Houlsby+11 BALD; "confidently uncertain".
    * ``disagreement`` -- mean L1 vs volunteer fractions across valid
      questions; only galaxies with >=1 valid question are included.

    Top-K is sorted descending by metric value. The response also
    carries the population ``median`` so the frontend can show
    "outlier 7.2 vs median 1.4" for context.
    """
    if metric not in OUTLIER_METRICS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unknown metric {metric!r}; "
                f"valid options: {list(OUTLIER_METRICS)}"
            ),
        )
    if k < 1:
        raise HTTPException(status_code=400, detail=f"k must be >= 1; got {k}")
    k = min(k, OUTLIER_MAX_K)
    payload = _outliers_payload()
    metrics = payload.get("metrics", {})
    rows = list(metrics.get(metric, []))[:k]
    median_block = payload.get("median", {})
    items = [
        OutlierItem(
            idx=int(r["idx"]),
            value=float(r["value"]),
            thumbnail_url=f"/api/test_thumbs/{int(r['idx'])}/thumbnail",
        )
        for r in rows
    ]
    return OutliersResponse(
        metric=metric,
        median=float(median_block.get(metric, 0.0)),
        items=items,
    )


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
