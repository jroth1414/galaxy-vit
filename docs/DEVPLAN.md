# Galaxy-ViT Development Plan v5 (Agent-Executable)

Source of truth for the coding agent. Every task below has a machine-verifiable acceptance test. The agent works top-down, one PR per task, pausing for human review only at the five HITL checkpoints.

---

## 1. Principles for Agent Execution

1. **Every task has a machine-verifiable acceptance test.** Stop when `pytest -k <task_id>` exits 0.
2. **No ambiguous pronouns.** Exact file paths, variable names, HF Hub IDs only.
3. **Deterministic by default.** Every script takes `--seed` (default 42) and writes `run_config.json`.
4. **One task = one PR** (<400 LOC diff).
5. **Fail loudly, early.** Validate all inputs with Pydantic or `assert` before compute.
6. **No network calls in tests.** Offline pytest only; fixtures in `tests/fixtures/`.
7. **Never invent** file paths, HF repo IDs, hyperparameters, or schema names.

---

## 2. Canonical Names (Verbatim)

```
Repo root:                galaxy-vit/
Python pkg:               galaxy_vit/
Config dir:               configs/
Data cache:               ~/.cache/galaxy_vit/
HF model repo:            <HF_USER>/galaxy-vit-zoobot-dirichlet
HF space:                 <HF_USER>/galaxy-vit-demo
W&B project:              galaxy-vit
W&B entity:               <WANDB_ENTITY>

External IDs (do not substitute):
  Zoobot encoder:         mwalmsley/zoobot-encoder-convnext_nano
  ViT baseline:           google/vit-base-patch16-224
  Galaxy10 dataset:       matthieulel/galaxy10_decals
  GZ DESI catalog:        Zenodo record 8331338
  Aladin Lite v3:         https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js
  SDSS cutouts:           https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg

Env vars (required):      HF_USER, HF_TOKEN, WANDB_API_KEY, WANDB_ENTITY, DATA_DIR
```

---

## 3. Repository Layout

```
galaxy-vit/
├── galaxy_vit/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── galaxy10.py
│   │   ├── gz_desi.py
│   │   ├── schema.py
│   │   ├── splits.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── vit_baseline.py
│   │   ├── zoobot_encoder.py
│   │   └── dirichlet_head.py
│   ├── losses/
│   │   └── dirichlet_mn.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── metrics.py
│   ├── inference/
│   │   ├── predict.py
│   │   ├── attention.py
│   │   └── posterior.py
│   ├── serve/
│   │   ├── app.py
│   │   ├── schemas.py
│   │   └── sdss.py
│   └── viz/
│       ├── umap_embed.py
│       └── failure_gallery.py
├── frontend/
├── configs/
├── scripts/
├── tests/
│   ├── fixtures/
│   ├── test_schema.py
│   ├── test_masking.py
│   ├── test_loss.py
│   ├── test_head.py
│   ├── test_dataloader.py
│   └── test_serve.py
├── docs/
│   ├── DEVPLAN.md
│   ├── ARCHITECTURE.md
│   └── SCHEMA.md
├── Dockerfile
├── .github/workflows/
├── pyproject.toml
└── README.md
```

---

## 4. Task Graph

### Phase 0 — Scaffolding

**T0.1 — Repo scaffold**
- Create layout above, `pyproject.toml` (uv/hatch), pre-commit (ruff + mypy + detect-secrets), `Dockerfile`, `ci.yml`.
- Accept: `ruff check . && mypy galaxy_vit && pytest -q` exits 0.

**T0.2 — Settings + env validation**
- `galaxy_vit/config.py` with Pydantic `Settings` for required env vars; fails loudly if missing.
- Accept: `tests/test_config.py::test_missing_env_raises`.

### Phase 1 — Month 1 (Foundation)

**T1.1 — Galaxy10 loader + stratified split**
- Output: `data/splits/galaxy10_split.csv`, `galaxy_vit/data/galaxy10.py`.
- Accept: `test_galaxy10_split_sizes` — 70/15/15 within ±5; class ratios within 1% across splits.

**T1.2 — Compute DECaLS normalization**
- Output: `configs/normalization.json` (per-channel mean/std on train split only).
- Accept: `test_normalization_matches_cached` within 1e-4.

**T1.3 — Augmentation pipeline**
- Rotation [-180,180] bilinear + reflect pad + center-crop 224; H/V flip; resized-crop 0.85–1.0; MixUp α=0.2; no hue jitter.
- Accept: `test_transforms_shape_dtype`, `test_no_corner_zeros_after_rotation`.

**T1.4 — ViT-B/16 baseline**
- Command: `python -m galaxy_vit.training.trainer --config configs/m1_vit_baseline.yaml`.
- Accept: val top-1 ≥ 0.82 AND val macro-F1 ≥ 0.78; `runs/m1_vit/metrics.json` written; W&B URL printed.

**T1.5 — Zoobot ConvNeXt-nano finetune + class-balanced loss** *(HITL gate #1)*
- Class-balanced loss (Cui 2019, β=0.9999); two-stage head-then-full.
- Accept: val macro-F1 strictly exceeds T1.4 by ≥1.5 absolute points on identical split.

**T1.6 — Interpretability (attention rollout + GradCAM)**
- Output: 30 stratified test-image overlays to `runs/m1_zoobot/interpretability/`.
- Accept: `test_attention_shape`, `test_gradcam_nonzero`.

**T1.7 — FastAPI backend**
- Endpoints: `/health`, `/predict`, `/predict_sdss?ra&dec`, `/attention/{id}`.
- Accept: `pytest tests/test_serve.py`; p95 CPU latency < 800 ms for `/predict`.

**T1.8 — React + Aladin Lite UI + HF Space deploy**
- Tabs: Classify, Model Card.
- Accept: Playwright smoke test on live URL; GitHub Actions deploy green.

### Phase 2 — Month 2 (GZ DESI Scale)

**T2.1 — GZ DESI catalog ingestion**
- Zenodo 8331338 → `data/gz_desi_500k.parquet` (DECaLS-only, ≥5 votes/question).
- Accept: 400k–600k rows; schema validator confirms all vote-count columns.

**T2.2 — Streaming WebDataset pipeline**
- Accept: 10k-sample pass in <120 s on one 5070 Ti.

**T2.3 — Reproduce W+23 10-way baseline**
- Accept: top-1 within 2% of published value.

**T2.4 — UMAP embedding extractor + static figure**
- Output: `artifacts/umap_penultimate.png`, `artifacts/umap_coords.parquet`.
- Accept: silhouette score ≥ 0.15 across coarse classes.

### Phase 3 — Month 3 (Dirichlet Head, Agent-Critical)

**T3.1 — Schema integration**
- Import `gz_desi_pairs`, `gz_desi_dependencies` from `galaxy-datasets`; wrap in `galaxy_vit/data/schema.py`.
- Accept: `tests/test_schema.py` — total answers = sum of per-question counts; every dependency key exists.

**T3.2 — Masking unit tests (TDD gate; HITL #2)**
- Agent writes 5+ synthetic-galaxy test cases FIRST; tests must FAIL because code not yet written.
- Cases: zero votes on dependent, parent lost, below min_votes, all full, parent tie strict.

**T3.3 — `DirichletMultinomialHead` module**
- `softplus(logits) + alpha_floor=1.0`.
- Accept: `test_forward_positive_alpha`, `test_backward_finite_grad`.

**T3.4 — Dirichlet-Multinomial loss with masking**
- `torch.special.gammaln` in fp32; per-question normalization.
- Accept: T3.2 tests all pass; `test_overfit_100_samples` reaches MAE<5% in <200 steps.

**T3.5 — Reference comparison vs. Zoobot 2.0**
- Accept: loss within 1% of Zoobot's native `define_model` + loss on identical inputs.

**T3.6 — Full training run on 500k subset** *(HITL gate #3)*
- Config: `configs/m3_dirichlet.yaml`; 50 epochs bf16.
- Accept gates (all required): MAE on smooth-or-featured ≤ 15%; coverage @ 95% CI ≥ 85%; monotone loss first 5 epochs; no NaN α.

**T3.7 — Analytic posterior module**
- `Beta(α_i, Σα − α_i)` CIs via `Beta.icdf`.
- Accept: `test_beta_ci_matches_scipy` within 1e-4.

### Phase 4 — Month 4 (AL + UMAP Explorer + Posteriors Tab)

**T4.1 — Active learning loop (entropy / BALD)**
- Accept: entropy acquisition reaches 90% of full-data MAE in ≤60% of labels across 3 seeds.

**T4.2 — Interactive UMAP Explorer tab**
- `react-plotly.js` scattergl; hover thumbnail, click-to-Aladin, color-by selector, lasso.
- Accept: Playwright lassos 100 points; sample grid renders.

**T4.3 — Multi-Question Posteriors tab**
- Per-question bars with 95% CI whiskers; parent-dependency greyout; compare-to-volunteers overlay.
- Accept: Playwright selects 3 galaxies; verifies non-overlapping CIs on ≥1 question.

### Phase 5 — Month 5 (Science + Release)

**T5.1 — Full 8.67M inference pass** *(HITL gate #4)*
- Rented A100/H100; writes `releases/gz_desi_dirichlet_v1.parquet`.
- Accept: row count matches ±0.1%; α columns positive; checksum logged.

**T5.2 — Science case (bar fraction vs. z)**
- Output: `artifacts/bar_fraction_vs_z.png` + 400-word `science_note.md`.
- Accept: qualitative trend matches W+23 direction.

**T5.3 — Release HF Dataset**
- Accept: dataset card renders; Zenodo DOI reserved.

### Phase 6 — Month 6 (Paper + Release)

**T6.1 — Paper draft** *(HITL gate #5)*
- LaTeX (ml4ps or mnras template).
- Accept: `pdflatex` compiles; `check_citations.py` flags no missing refs.

**T6.2 — HF Hub model release + model card**
- Accept: `huggingface_hub` loads weights end-to-end in fresh venv.

**T6.3 — Live demo v2 (4 tabs) + 60-s Loom**
- Accept: uptime monitor green for 72 h; Playwright e2e on all 4 tabs.

---

## 5. Canonical YAML Config (Agent Cannot Deviate)

```yaml
run_id: m3_dirichlet_v1
seed: 42
data:
  source: gz_desi_500k
  parquet: data/gz_desi_500k.parquet
  split_csv: data/splits/gz_desi_500k_split.csv
  image_size: 224
  normalization: configs/normalization.json
model:
  encoder: mwalmsley/zoobot-encoder-convnext_nano
  head: dirichlet_multinomial
  alpha_floor: 1.0
loss:
  name: dirichlet_mn
  min_votes: 3
optim:
  name: adamw
  encoder_lr: 1.0e-5
  head_lr: 1.0e-3
  weight_decay: 0.05
  warmup_epochs: 3
  schedule: cosine
train:
  epochs: 50
  batch_size: 64
  precision: bf16
  grad_clip: 1.0
  early_stop_metric: val/vote_mae_macro
  early_stop_patience: 5
logging:
  wandb_project: galaxy-vit
  wandb_tags: [m3, dirichlet, zoobot, convnext_nano]
  save_dir: runs/m3_dirichlet_v1
```

---

## 6. CI / CD

`.github/workflows/ci.yml` on every PR:
1. `ruff check`, `mypy galaxy_vit`.
2. `pytest -q` (offline).
3. 200-step smoke-training on 50-galaxy fixture (`make smoke`); loss must decrease.
4. Docker build check (no push).

`.github/workflows/deploy.yml` on push to `main`:
1. Multi-stage Docker image (frontend → FastAPI).
2. Push to HF Space.
3. Ping `/health` for 60 s warm-up.

---

## 7. Guardrails

- Never commit: W&B keys, HF tokens, scraped CSVs >10 MB. `detect-secrets` pre-commit.
- Never `pip install` without updating `pyproject.toml`.
- Never edit `configs/normalization.json` after T1.2 without a new run ID.
- Never change the Dirichlet schema without re-running T3.1 tests.
- Never skip a failing acceptance test; if it fails twice, pause for human review.

---

## 8. Per-Task PR Template

```
## Task ID
T3.4

## Goal (copied from plan)
Masked Dirichlet-Multinomial NLL in galaxy_vit/losses/dirichlet_mn.py.

## Acceptance tests run
pytest tests/test_masking.py tests/test_loss.py -q  ->  PASS (12/12)

## New / changed files
- galaxy_vit/losses/dirichlet_mn.py  (new, 54 LOC)
- tests/test_loss.py                 (new, 110 LOC)

## Metrics / artifacts produced
- artifacts/overfit_100_curve.png (loss -> 0.03 in 180 steps)

## Open questions for human
- None.

## Next task
T3.5
```

---

## 9. HITL Gates (Five Only)

1. End of T1.5 — does Zoobot beat ViT by ≥1.5 macro-F1 points?
2. End of T3.2 — do synthetic masking tests capture intended semantics?
3. End of T3.6 — do Dirichlet training gates all pass?
4. End of T5.1 — confirm config before spending cloud GPU ($50–150).
5. T6.1 — paper draft review before arXiv.
