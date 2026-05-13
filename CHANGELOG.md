# Changelog

## v2 — Visualization features (2026-05)

A focused expansion of the live demo across ten features, planned in
[`docs/v2_features_plan.md`](docs/v2_features_plan.md). Test suite grew
from 189 → 324 passing tests (+135) and the SPA grew from 4 tabs to 7.

### New tabs

| Tab | Features inside |
|---|---|
| **Sky** | RA/Dec scatter of ~14k DR8 galaxies (S-2) · Aladin Lite DR10 embed (S-2) · "Predict at coords" via /api/predict_sdss · Object-name + RA/Dec resolver via CDS Sesame (A-8) |
| **Similar** | Cosine-kNN over 2,462 cached ConvNeXt features (S-1); query by upload or cache idx; deep-link from Classify / Posteriors / Explorer |
| **Training** | Per-epoch animation of 24 demo galaxies in Zoobot feature space (C-15), all epochs projected through the final-epoch UMAP so the cloud doesn't jitter |

### Inside existing tabs

| Tab | New capabilities |
|---|---|
| **Classify** | Compare panel — same image through M1 (Galaxy10) and M3 (Dirichlet) side-by-side (C-16); informal mapping table in [`docs/m1_to_m3_mapping.md`](docs/m1_to_m3_mapping.md) |
| **Posteriors** | "Most interesting galaxies" outlier panel: entropy / BALD / disagreement (S-3) · Bar-view ↔ Sankey question-tree toggle (A-5) · Per-question GradCAM dropdown (A-7) · "Find similar" deep-link · Outlier-clicked test thumb loads its full posterior |
| **Explorer** | 2-D / 3-D UMAP toggle (A-6) · Hover crossfades to precomputed GradCAM (S-4) · "Find similar" deep-link |

### New backend endpoints

```
GET  /api/similar/{idx}?k=N             S-1
POST /api/similar                       S-1
GET  /api/outliers?metric=...&k=N       S-3
GET  /api/test_thumbs/{idx}/saliency    S-4
GET  /api/umap_points?n_dims=2|3        A-6
GET  /api/sky_points                    S-2
POST /api/tree_flow                     A-5
GET  /api/tree_flow/test_thumbs/{idx}   A-5
POST /api/per_question_gradcam          A-7
GET  /api/per_question_gradcam/test_thumbs/{idx}   A-7
GET  /api/resolve_name?name=...         A-8
POST /api/compare                       C-16
GET  /api/training_movie                C-15
```

### New artifacts (all committed; total ~28 MB)

| Path | Size | Producer |
|---|---:|---|
| `artifacts/test_thumb_features.parquet` | ~10 MB | `scripts/cache_test_thumb_features.py` (S-1) |
| `artifacts/outliers.json` | ~25 KB | `scripts/build_outlier_indices.py` (S-3) |
| `artifacts/test_saliencies/*.jpg` | ~13 MB (2,462 files) | `scripts/build_test_saliencies.py` (S-4) |
| `artifacts/umap_3d_coords.parquet` + `_metrics.json` | ~50 KB | `scripts/extract_umap_3d.py` (A-6) |
| `artifacts/sky_points.parquet` | ~600 KB | `scripts/build_sky_points.py` (S-2) |
| `artifacts/training_movie.parquet` | ~150 KB | `scripts/build_training_movie.py` (C-15) |

### Engineering

- New `galaxy_vit/inference/{similarity,outliers,tree_flow}.py` modules.
- `galaxy_vit/training/dirichlet_trainer.py` gained an opt-in per-epoch
  demo-feature dump hook (`logging.per_epoch_features_path`).
- All v2 features ship behind 503-with-hint fallbacks when their
  precomputed artifacts are absent, so the demo degrades gracefully on
  fresh checkouts.

## v1 — Initial release (2026-04)

Initial Galaxy-ViT release covering DEVPLAN T0.1 – T6.3 (skipping T6.1
paper draft):

- **M1** — Galaxy10 ViT-B/16 baseline + Zoobot ConvNeXt-nano finetune
- **M2** — Walmsley+23 reproduction baseline (per-question cross-entropy)
- **M3** — Dirichlet-Multinomial head with parent-dependency masking,
  closed-form Beta credible intervals, and post-hoc temperature
  calibration
- Active learning with closed-form Dirichlet BALD acquisition
- Full-pass inference pipeline + Zenodo-ready release parquet
- Live demo: Classify / Posteriors / Explorer / Model Card
- HF Hub releases for both the predictions dataset and the model

Test suite: 189 passed, 1 skipped.
