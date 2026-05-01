# Galaxy-ViT: Bayesian Multi-Question Galaxy Morphology with Zoobot + Dirichlet

> **Status**: Development plan v5 (agent-executable). Month 1 target: shippable live demo. Months 2–6: research extension toward a Bayesian multi-question model on Galaxy Zoo DESI.

**Primary narrative**: Finetune a galaxy-pretrained Zoobot encoder on Galaxy10 DECaLS, then scale to Galaxy Zoo DESI (8.67M galaxies) with a Dirichlet-Multinomial multi-question head that produces calibrated vote-fraction posteriors per question of the Galaxy Zoo decision tree.

**Live demo**: https://huggingface.co/spaces/<HF_USER>/galaxy-vit-demo
**Model weights**: https://huggingface.co/<HF_USER>/galaxy-vit-zoobot-dirichlet
**W&B report**: https://wandb.ai/<WANDB_ENTITY>/galaxy-vit
**Paper (preprint)**: *populated after T6.1*

---

## For the Coding Agent: Read This First

You are executing the plan in `docs/DEVPLAN.md` (v5). Your rules of engagement:

1. **Work task-by-task, top-down.** Tasks are identified `T0.1 … T6.3`. Do not skip ahead.
2. **Every task has a machine-verifiable acceptance test.** Stop when `pytest -k <task_id>` passes.
3. **One task = one PR.** Use `.github/PULL_REQUEST_TEMPLATE.md`. Diff budget: <400 LOC.
4. **If an acceptance gate fails twice, STOP and request human review.** Do not weaken the test.
5. **Never invent** file paths, HF repo IDs, hyperparameters, or schema names. They are enumerated in §Canonical Names below.
6. **Never commit secrets.** `detect-secrets` runs pre-commit. Read env vars via `galaxy_vit.config.Settings`.
7. **Pause for human review** at these five checkpoints: end of T1.5, T3.2, T3.6, T5.1, T6.1.
8. **Reproducibility is non-negotiable.** Every script takes `--seed` (default 42) and writes `run_config.json` next to its outputs.

If you cannot proceed without making an assumption, open an issue titled `CLARIFY: <task_id>` and stop.

---

## For the Human Reviewer

- Full plan: `docs/DEVPLAN.md`
- Task board: GitHub Projects "Galaxy-ViT v5" (auto-populated from plan)
- HITL checkpoints: see §Human-in-the-Loop Gates below
- Expected timeline: Month 1 (4 weekends, shippable), Months 2–6 (research extension, ~6 months calendar)

---

## What This Project Is

A Bayesian galaxy morphology classifier that:

- **Ingests** images from DECaLS / DES / BASS / MzLS via the Galaxy Zoo DESI catalog.
- **Predicts** per-question Dirichlet concentrations over the Galaxy Zoo decision tree.
- **Produces** analytic posterior credible intervals per answer per galaxy — no MC-Dropout hack required.
- **Ships** as a live web demo with four tabs: Classify, Multi-Question Posteriors, UMAP Explorer, Label Priority Dashboard.
- **Releases** a predicted vote-fraction catalog for 8.67M galaxies as a public HF Dataset.

It is designed to be executed end-to-end by a coding agent in ~6 months of calendar time, with Month 1 producing a polished portfolio artifact even if later months slip.

---

## Canonical Names (Agent: Use Verbatim)

```
Repo root:                galaxy-vit/
Python pkg:               galaxy_vit/
HF model repo:            <HF_USER>/galaxy-vit-zoobot-dirichlet
HF space:                 <HF_USER>/galaxy-vit-demo
W&B project:              galaxy-vit

External IDs (do not substitute):
  Zoobot encoder:         mwalmsley/zoobot-encoder-convnext_nano
  ViT baseline:           google/vit-base-patch16-224
  Galaxy10 dataset:       matthieulel/galaxy10_decals
  GZ DESI catalog:        Zenodo record 8331338
  Aladin Lite v3:         https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js
  SDSS cutouts:           https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg

Environment variables (required; Settings fails if missing):
  HF_USER, HF_TOKEN, WANDB_API_KEY, WANDB_ENTITY, DATA_DIR
```

---

## Repository Layout

```
galaxy-vit/
├── galaxy_vit/         # Python package (data, models, losses, training, inference, serve, viz)
├── frontend/           # Vite + React + TS (shadcn/ui, Tailwind, Aladin Lite v3, Plotly)
├── configs/            # One YAML per training run
├── scripts/            # Thin CLI entry points
├── tests/              # Offline pytest suite; fixtures in tests/fixtures/
├── docs/
│   ├── DEVPLAN.md      # Full v5 plan (source of truth for the agent)
│   ├── ARCHITECTURE.md # System diagram + deployment topology
│   └── SCHEMA.md       # GZ DESI decision-tree schema reference
├── Dockerfile          # Multi-stage: frontend build → FastAPI runtime
├── .github/workflows/  # ci.yml (PR gate), deploy.yml (push to main → HF Space)
└── pyproject.toml      # uv/hatch; pins + dev extras
```

---

## Quickstart

### 1. Install
```bash
git clone https://github.com/<HF_USER>/galaxy-vit && cd galaxy-vit
cp .env.example .env  # fill in HF_USER, HF_TOKEN, WANDB_API_KEY, WANDB_ENTITY, DATA_DIR
uv sync --all-extras
pre-commit install
```

### 2. Verify env
```bash
python -m galaxy_vit.config         # prints redacted settings; exits nonzero if missing vars
pytest -q                           # all offline tests must pass before any training
```

### 3. Run Month 1 baseline (smoke)
```bash
python -m galaxy_vit.training.trainer --config configs/m1_vit_baseline.yaml
python -m galaxy_vit.training.trainer --config configs/m1_zoobot_finetune.yaml
```

### 4. Run FastAPI + frontend locally
```bash
docker build -t galaxy-vit . && docker run -p 7860:7860 --env-file .env galaxy-vit
# open http://localhost:7860
```

---

## Task Graph (Summary)

| Phase | Tasks | Goal | Deliverable |
|---|---|---|---|
| 0 | T0.1–T0.2 | Scaffolding, Settings, CI | Green `pytest`, `ruff`, `mypy` |
| 1 (Month 1) | T1.1–T1.8 | Galaxy10 + ViT + Zoobot + demo | Live HF Space v1 |
| 2 (Month 2) | T2.1–T2.4 | GZ DESI pipeline + reproduce W+23 | 500k Parquet + UMAP fig |
| 3 (Month 3) | T3.1–T3.7 | Dirichlet-Multinomial head | Calibrated posteriors, 5 HITL gates |
| 4 (Month 4) | T4.1–T4.3 | Active learning + UMAP Explorer + Posteriors tab | 2 new UI tabs + AL curves |
| 5 (Month 5) | T5.1–T5.3 | 8.67M inference + science case + release | HF Dataset catalog |
| 6 (Month 6) | T6.1–T6.3 | Paper + HF weights + demo v2 | arXiv preprint, live 4-tab demo |

See `docs/DEVPLAN.md` for the full acceptance tests per task.

---

## Acceptance Test Philosophy

Every task terminates when a specific `pytest` command exits 0. Example from T3.4 (Dirichlet-Multinomial loss):

```bash
pytest tests/test_masking.py tests/test_loss.py -q
# PASS required: 12/12 including:
#   test_mask_zero_votes_on_dependent_question
#   test_mask_parent_answer_lost
#   test_mask_below_min_votes
#   test_mask_all_questions_full
#   test_mask_parent_tie_is_strict
#   test_overfit_100_samples
#   test_matches_zoobot_reference
```

Visual artifacts (UMAP embeddings, confusion matrices, posterior CIs) are also gated — either by a scalar metric threshold or a Playwright end-to-end test on the deployed demo.

---

## Human-in-the-Loop Gates

The agent pauses and requests review exactly here:

1. **End of T1.5** — Zoobot beats ViT by ≥1.5 macro-F1 points? If not, diagnose before GZ DESI.
2. **End of T3.2** — Do the 5 synthetic masking tests capture the semantics the human wants?
3. **End of T3.6** — Dirichlet training gates all pass? (MAE ≤ 15% on smooth-or-featured, coverage ≥ 85% at 95% CI, no NaN α.)
4. **End of T5.1** — Before spending cloud GPU ($50–150), confirm the final config.
5. **T6.1** — Paper-draft review before arXiv submission.

Everywhere else the agent runs autonomously.

---

## Tech Stack

**Training**: Python 3.11 · PyTorch nightly (cu128/cu130, sm_120) · `timm` · `transformers` · `datasets` · `zoobot` ≥ 2.0 · `wandb` · `torch.special.gammaln` (fp32)

**Backend**: FastAPI · Uvicorn · Pydantic v2 · `httpx` test client · in-memory LRU cache for SDSS cutouts

**Frontend**: Vite · React 18 · TypeScript · TailwindCSS · shadcn/ui · Recharts · `react-plotly.js` (scattergl for UMAP) · Aladin Lite v3 (with CDS attribution retained)

**Deploy**: Multi-stage Docker → Hugging Face Docker Space (free CPU Basic, 16 GB RAM, port 7860) · GitHub Actions on push to `main` · Playwright e2e after deploy

**CI gates**: `ruff`, `mypy`, `pytest` (offline), 200-step smoke training, `detect-secrets`

---

## Configuration

All training runs are launched by one command:

```bash
python -m galaxy_vit.training.trainer --config configs/<run>.yaml
```

The YAML schema is fixed; the agent must not introduce new keys without updating the Pydantic config model in `galaxy_vit/training/trainer.py`. See `configs/m3_dirichlet.yaml` for the canonical example.

---

## Model and Math Summary

- **Encoder**: `mwalmsley/zoobot-encoder-convnext_nano` (galaxy-pretrained on GZ volunteer answers). Baseline: `google/vit-base-patch16-224`.
- **Head** (Month 3+): `DirichletMultinomialHead` with `softplus(logits) + alpha_floor=1.0` to keep α strictly positive.
- **Loss**: masked Dirichlet-Multinomial NLL, normalized per question; dependency mask enforces GZ decision-tree constraints.
- **Posteriors**: analytic — per-answer marginal is `Beta(α_i, Σα − α_i)`; 95% CI via `Beta.icdf`.
- **Evaluation metrics**: mean vote-fraction MAE per question, Brier score, coverage at 50/80/95% CI, argmax agreement, ECE.

Full math in `docs/DEVPLAN.md` §Month 3.

---

## Live Demo Tabs

| Tab | Shipped after | What it does |
|---|---|---|
| **Classify** | T1.8 | Upload image or click on Aladin Lite sky → top-3 classes + attention overlay |
| **Model Card** | T1.8 | ViT-vs-Zoobot comparison table, confusion matrix, failure gallery |
| **UMAP Explorer** | T4.2 | Interactive embedding; hover preview, click-to-Aladin, color-by, lasso |
| **Multi-Question Posteriors** | T4.3 | Per-question Dirichlet bars with 95% CI whiskers, parent-dependency greyout |

---

## Reproducibility Guarantees

- Every run writes `run_config.json` (resolved YAML + git SHA + `pip freeze`).
- Splits live in `data/splits/*.csv` and are committed to the repo.
- DECaLS normalization stats are computed once in T1.2 and frozen in `configs/normalization.json`.
- All random seeds propagate: `--seed 42` default; NumPy, PyTorch (CPU + CUDA), Python `random`.
- W&B runs are public and linked from the model card.

---

## Constraints and Known Landmines

- **`sm_120` on PyTorch**: verify `torch.cuda.get_arch_list()` before assuming friction; nightly wheels have matured since late 2025.
- **`lgamma(0) = +∞`**: the `alpha_floor=1.0` in the head is load-bearing. Do not lower below 0.01.
- **Zero-vote answers must be `0`, not `NaN`** in the catalog.
- **bf16 + `gammaln`**: cast α to fp32 before `gammaln` calls; loss in fp32, forward pass in bf16.
- **HF Spaces**: only `/tmp` is writable; 30-min idle sleep (we accept cold starts, no pinger).
- **SDSS cutout API**: ~100 req/min informal rate limit; the `sdss.py` client uses LRU cache.
- **Aladin Lite**: free to embed, but CDS/Aladin logo must remain visible (licensing).
- **Galaxy Zoo labels are noisy** (~55% volunteer agreement threshold); practical accuracy ceiling ~92–95%.
- **GZ DESI (8.67M)** cannot fit on a single workstation — streaming pipeline is required from T2.2 onward.
- **Solo 6-month ML+astro projects routinely slip to 9–12 months.** Month 1 must be shippable standalone.

---

## Contributing (If the Agent Becomes a Team)

- One PR per task; use `.github/PULL_REQUEST_TEMPLATE.md`.
- Pre-commit: `ruff`, `mypy`, `detect-secrets`.
- All new dependencies go through `pyproject.toml`; no naked `pip install`.
- Breaking schema changes (e.g., GZ decision tree) require re-running T3.1 tests and bumping a run-ID suffix.

---

## Citing This Work

Once T6.1–T6.2 are complete:

```bibtex
@misc{roth2026galaxyvit,
  author       = {Roth, John},
  title        = {Galaxy-ViT: Bayesian Multi-Question Galaxy Morphology via Zoobot + Dirichlet},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/<HF_USER>/galaxy-vit-zoobot-dirichlet}},
  note         = {arXiv:XXXX.XXXXX}
}
```

And cite the upstream work this depends on: Walmsley+22 (Zoobot), Walmsley+23 (GZ DESI), Walmsley+20 (Bayesian CNNs + active learning), Dagli 2023 (Astroformer, Galaxy10 SOTA context), Abnar & Zuidema 2020 (attention rollout), Cui+19 (class-balanced loss).

---

## License

Code: MIT. Model weights: CC-BY 4.0 (matches Zoobot upstream). Galaxy Zoo data: CC-BY-SA per Galaxy Zoo terms. Aladin Lite: embed-only, CDS attribution retained.
