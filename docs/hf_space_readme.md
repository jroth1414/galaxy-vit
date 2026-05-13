---
title: Galaxy-ViT
emoji: 🌌
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Bayesian galaxy morphology — Zoobot + Dirichlet-Multinomial
tags:
  - astronomy
  - galaxy-morphology
  - galaxy-zoo
  - bayesian
  - dirichlet-multinomial
  - zoobot
---

# Galaxy-ViT — live demo

A Bayesian multi-question galaxy morphology classifier built on a
galaxy-pretrained [Zoobot](https://huggingface.co/mwalmsley/zoobot-encoder-convnext_nano)
ConvNeXt-nano encoder with a Dirichlet–Multinomial output head over
the Galaxy Zoo DESI 10-question / 34-answer decision tree. Produces
analytic per-answer credible intervals at ≥ 93 % coverage at the 95 %
level.

## What this Space demonstrates

Seven tabs over the same backend:

| Tab | What it does |
|---|---|
| **Classify** | (M1 not bundled on the Space — gracefully 503s; full M3 posterior is exposed in the Posteriors tab) |
| **Posteriors** | per-question Dirichlet bars with 95 % CI whiskers · Sankey question-tree view · per-question GradCAM · "most interesting galaxies" outlier gallery |
| **Explorer** | 2-D / 3-D UMAP over 2,462 DR8 test galaxies · lasso · hover thumbnails crossfade to precomputed GradCAM · click → posterior |
| **Sky** | ~14k DR8 galaxies on an RA/Dec scatter · Aladin Lite DR10 embed · object-name resolver (M31, NGC 1300, …) via CDS Sesame |
| **Similar** | cosine-kNN over cached ConvNeXt features; query by upload or cache idx |
| **Training** | per-epoch animation of 24 demo galaxies in Zoobot feature space across all training epochs |
| **Model Card** | comparison metrics, curves, interpretability gallery |

## How it works

The container boots a FastAPI backend (Uvicorn, port 7860) that
serves a Vite/React 19 SPA. The M3 (Dirichlet) checkpoint is
downloaded from the published model repo
[`roth1414/galaxy-vit-zoobot-dirichlet`](https://huggingface.co/roth1414/galaxy-vit-zoobot-dirichlet)
at startup via `huggingface_hub`. All demo data (UMAP coordinates,
test-set thumbnails, precomputed GradCAMs, sky points, training-movie
frames) ships in the image — ~40 MB total.

## Source + datasets

- **Code**: <https://github.com/jroth1414/galaxy-vit>
- **Predictions dataset**: <https://huggingface.co/datasets/roth1414/galaxy-vit-gz-desi-dirichlet-predictions>
  (61,440-row per-galaxy α parquet)
- **Model weights**: <https://huggingface.co/roth1414/galaxy-vit-zoobot-dirichlet>

## Numbers

Held-out DR8 test split:

| Metric | Value |
|---|---:|
| Macro vote-fraction MAE (Dirichlet head) | **0.0883** |
| Macro coverage @ 95 % CI (T-calibrated) | **0.93** |
| Macro top-1 (cross-entropy baseline) | 0.8445 |
| Bar fraction vs bulge size | model and volunteer Spearman ρ both positive, p < 10⁻²² |

## License

Model weights: CC-BY-4.0 (matches Zoobot upstream). Predictions
parquet (companion release): CC-BY-4.0. App code: MIT. Galaxy Zoo and
DECaLS upstream data retain their original licenses.
