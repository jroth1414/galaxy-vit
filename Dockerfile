# syntax=docker/dockerfile:1.7
#
# Galaxy-ViT — multi-stage placeholder image.
#   Stage 1 (frontend): Vite/React build, real implementation lands in T1.8.
#   Stage 2 (runtime):  FastAPI + Uvicorn on Python 3.11, real entrypoint
#                       lands in T1.7.
#
# Until then this image installs the package and exits cleanly so that the
# CI "docker build check" job has something to verify.

# ---------- Stage 1: frontend build ----------
FROM node:20-bookworm-slim AS frontend
WORKDIR /frontend

# Copy whatever exists in frontend/ (likely just .gitkeep at T0.1).
COPY frontend/ ./

# Build if a package.json is present; otherwise emit a stub bundle so
# downstream COPY --from=frontend doesn't fail.
RUN set -eux; \
    if [ -f package.json ]; then \
        npm ci --no-audit --no-fund && npm run build; \
    else \
        mkdir -p dist && \
        printf '<!doctype html><meta charset="utf-8"><title>galaxy-vit</title>\n<p>Frontend bundle lands in T1.8.</p>\n' > dist/index.html; \
    fi

# ---------- Stage 2: Python runtime ----------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps minimal at T0.1; FastAPI/Pillow/etc. land with T1.7.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install the package (deps come from pyproject.toml).
COPY pyproject.toml README.md ./
COPY galaxy_vit ./galaxy_vit
RUN pip install --no-cache-dir .

# Pull in the built (or stub) frontend.
COPY --from=frontend /frontend/dist ./frontend/dist

# HF Spaces serves on 7860; same port locally for parity.
EXPOSE 7860

# Placeholder entrypoint — replaced in T1.7 with `uvicorn galaxy_vit.serve.app:app`.
CMD ["python", "-c", "print('galaxy-vit placeholder runtime; FastAPI lands in T1.7')"]
