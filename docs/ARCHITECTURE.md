# docs/ARCHITECTURE.md — Galaxy-ViT System Architecture

## 1. System Overview

Galaxy-ViT is a single-service web application fronted by a React SPA and backed by a FastAPI inference server. The trained model weights live on the Hugging Face Hub and are pulled at container startup. External astronomy services (SDSS SkyServer cutouts, DECaLS HiPS via Aladin Lite) are consumed on demand with in-memory caching.

```
┌──────────────────────────────────────────────────────────────┐
│  Hugging Face Space (Docker, free CPU Basic, port 7860)      │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  FastAPI (Uvicorn)                                     │  │
│  │  ├── / (static)  ── React SPA built bundle             │  │
│  │  ├── /api/health                                       │  │
│  │  ├── /api/predict              (upload JPG/PNG)        │  │
│  │  ├── /api/predict_sdss?ra&dec  (SDSS cutout lookup)    │  │
│  │  ├── /api/attention/{id}       (rollout / GradCAM PNG) │  │
│  │  ├── /api/posteriors/{id}      (Dirichlet CIs)         │  │
│  │  └── /api/umap                 (paged embedding JSON)  │  │
│  │                                                        │  │
│  │  Inference: bf16 on CPU (fallback ONNX/INT8 if needed) │  │
│  │  Weights pulled from HF Hub at startup → /tmp/model/   │  │
│  └────────────────────────────────────────────────────────┘  │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼   (on demand, LRU-cached)
       ┌──────────────────────────────────────┐
       │ SDSS SkyServer ImgCutout (DR18)      │
       │ DECaLS HiPS via Aladin Lite v3 (CDS) │
       │ HF Hub (weights)                     │
       └──────────────────────────────────────┘
```

## 2. Component Responsibilities

### 2.1 Frontend (`frontend/`)
- Vite + React 18 + TypeScript.
- TailwindCSS + shadcn/ui component library.
- Aladin Lite v3 embedded on the Live Sky / Classify tab (DECaLS DR10 HiPS layer; CDS logo retained).
- `react-plotly.js` (scattergl) for the UMAP Explorer.
- Recharts for top-k probability bars and confusion matrix.
- No direct third-party calls — all data flows through `/api/*`.

### 2.2 Backend (`galaxy_vit/serve/`)
- `app.py` — FastAPI app; mounts static frontend bundle at `/` and API at `/api`.
- `schemas.py` — Pydantic v2 request / response models (deterministic validation).
- `sdss.py` — SkyServer client with `functools.lru_cache` (maxsize=1024) and exponential backoff.
- Concurrency: single-worker Uvicorn (CPU Basic tier has 2 vCPU; model loading is ~2 GB RAM, so one worker is the safe default).

### 2.3 Model layer (`galaxy_vit/models/`, `inference/`)
- Encoder: `mwalmsley/zoobot-encoder-convnext_nano` or ViT-B/16 baseline.
- Head: `DirichletMultinomialHead` (Month 3+) or 10-way softmax (Month 1).
- Forward: bf16 on CPU; loss math (gammaln) always cast to fp32.
- Posteriors: analytic `Beta(α_i, Σα − α_i)` via `torch.distributions.Beta.icdf`.

### 2.4 Data layer (`galaxy_vit/data/`)
- `galaxy10.py` — HF Datasets loader + stratified split (Month 1).
- `gz_desi.py` — WebDataset / Parquet streaming loader (Month 2+).
- `schema.py` — GZ DESI decision-tree schema (imports `gz_desi_pairs`, `gz_desi_dependencies` from `galaxy-datasets`).
- `transforms.py` — rotation, flips, resized-crop, MixUp. No hue jitter.

## 3. Deployment Topology

| Environment | Frontend | Backend | Weights | Cost |
|---|---|---|---|---|
| Local dev | Vite dev server :5173 | uvicorn :8000 | Local checkpoint | $0 |
| Docker local | Static bundle in container | uvicorn :7860 | Local checkpoint | $0 |
| HF Space (prod) | Static bundle in container | uvicorn :7860 | HF Hub → `/tmp/model/` | $0 |
| Cloud GPU (Month 5) | N/A | Modal/Lambda batch job | Local checkpoint | $50–150 one-off |

Cold-start behavior: HF Spaces sleeps after 30 min idle; first request takes 30–60 s. The frontend shows a "warming up the model…" state. No keep-alive pinger (against HF Spaces ToS).

## 4. Data Flow Diagrams

### 4.1 Classify via upload
```
User → React (Classify tab) → POST /api/predict (multipart)
     → FastAPI decodes PIL → transforms → model forward (bf16)
     → softmax / Dirichlet posterior → attention rollout PNG
     → JSON { top_k, attention_url, posteriors? } → React renders
```

### 4.2 Classify via Aladin Lite click
```
User clicks sky → Aladin Lite emits (ra, dec) → React
     → GET /api/predict_sdss?ra=&dec=
     → sdss.py LRU → SDSS SkyServer cutout (PIL)
     → same inference pipeline as 4.1
```

### 4.3 UMAP Explorer
```
On mount → GET /api/umap?page=0&limit=5000 → paged Parquet read
     → scattergl renders; hover → thumbnail fetch; click → Aladin jump
     → lasso → POST /api/umap/select { ids[] } → sample grid of thumbnails
```

## 5. Secrets & Configuration

- Environment variables only (never committed): `HF_USER`, `HF_TOKEN`, `WANDB_API_KEY`, `WANDB_ENTITY`, `DATA_DIR`.
- `galaxy_vit.config.Settings` (Pydantic `BaseSettings`) loads + validates on startup.
- `.env.example` in repo root enumerates required keys; `.env` is git-ignored.
- `detect-secrets` pre-commit hook blocks accidental key commits.

## 6. Observability

- W&B training runs: `galaxy-vit` project; public reports linked from model card.
- Backend logs to stdout (HF Spaces captures); structured JSON logs via `loguru`.
- `/api/health` returns `{ok: true, model_sha: ..., uptime_s: ...}`.
- Playwright e2e on deploy: hits all tabs + one prediction per tab.

## 7. Failure Modes & Recovery

| Failure | Detection | Recovery |
|---|---|---|
| HF Hub download fails at startup | `/health` returns 503 | Container restart; fallback to last-cached weights in `/tmp/model/` if present |
| SDSS cutout 429/timeout | httpx raises → `/api/predict_sdss` returns 502 | LRU + exponential backoff; frontend shows retry toast |
| Aladin Lite CDN down | Frontend network error | Tab shows graceful fallback to upload-only |
| OOM on CPU Basic | Uvicorn worker killed | Auto-restart; downgrade to ONNX INT8 model in prod if recurrent |
| Model load > memory | Startup fails | CI smoke test catches before deploy |

## 8. Upgrade Paths

- **CPU Basic → GPU Small** ($9/mo HF upgrade): drop bf16/ONNX gymnastics, enable batched inference, latency <50 ms/image.
- **Modal / Replicate GPU inference endpoint**: keep the React SPA on HF Spaces, move `/api/*` to serverless GPU behind a reverse proxy. Useful only after user traffic justifies the cost.
- **Zenodo-hosted catalog** (Month 5): keep the Parquet release on HF Datasets but mirror on Zenodo for DOI citability.

## 9. Licensing Map

| Component | License | Note |
|---|---|---|
| Our code | MIT | `LICENSE` at repo root |
| Our weights | CC-BY 4.0 | Matches Zoobot upstream |
| Zoobot encoder | CC-BY 4.0 | Walmsley et al. |
| Galaxy Zoo labels | CC-BY-SA | Galaxy Zoo terms |
| SDSS data | Public domain | Cite SDSS collaboration |
| Aladin Lite | Free embed | CDS attribution must remain visible |
| DECaLS HiPS | Public | Cite Legacy Surveys |
