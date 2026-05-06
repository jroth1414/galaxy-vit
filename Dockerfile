# syntax=docker/dockerfile:1.7
#
# Galaxy-ViT — multi-stage runtime image (T1.8).
#   Stage 1 (frontend): Vite + React + TS + Tailwind v4 build → static SPA.
#   Stage 2 (runtime):  Python 3.11 + FastAPI + Uvicorn, model + SPA served
#                       on port 7860 (the HF Spaces convention).
#
# Build:  docker build -t galaxy-vit .
# Run:    docker run -p 7860:7860 \
#           -v $HOME/galaxy-vit-runs:/app/runs:ro \
#           -e GALAXY_VIT_CKPT=/app/runs/m1_zoobot_finetune/best.pt \
#           galaxy-vit
#
# The image deliberately does NOT bake model weights — best.pt is 343 MB
# and lives in a host-mounted /app/runs (or in the future, downloaded from
# the HF Hub at startup once T6.2 publishes weights).

# ---------- Stage 1: frontend build ----------
FROM node:22-bookworm-slim AS frontend
WORKDIR /frontend

# Copy lockfile first for cache-friendly npm ci.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

# Build the SPA bundle into /frontend/dist.
COPY frontend/ ./
RUN npm run build

# ---------- Stage 2: Python runtime ----------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# System deps:
#   - ca-certificates: TLS for HF Hub + SDSS SkyServer
#   - libgl1 + libglib2.0-0: Pillow / OpenCV image decoding for the SPA path
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install the package with the [m1] (PIL + datasets) and [m1-serve] extras.
# Torch and torchvision are CPU-only (HF Spaces free tier has no GPU);
# pinned via the dedicated cpu index for image size reasons.
COPY pyproject.toml README.md ./
COPY galaxy_vit ./galaxy_vit
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ".[m1,m1-train,m1-serve]"

# Bring in the built SPA from stage 1.
COPY --from=frontend /frontend/dist ./frontend/dist

# Bring in the small reproducibility artefacts (curves.png, normalization,
# splits) so the model card and trainer-side scripts can find them.
# best.pt and the TensorBoard event files stay out of the image and are
# expected to be host-mounted at /app/runs at run time.
COPY configs ./configs
COPY data ./data

EXPOSE 7860

# uvicorn single-worker on :7860; HF Spaces convention.
CMD ["uvicorn", "galaxy_vit.serve.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
