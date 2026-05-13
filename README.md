# Galaxy-ViT

A Bayesian multi-question galaxy morphology classifier built on a galaxy-pretrained
Zoobot ConvNeXt-nano encoder with a Dirichlet–Multinomial output head. Models the
Galaxy Zoo DESI 10-question / 34-answer decision tree end-to-end and produces
analytic per-answer credible intervals.

| | |
|---|---|
| Macro top-1 (cross-entropy baseline) | **0.8445** on the DR8 test split |
| Macro vote-fraction MAE (Dirichlet head) | **0.0883** |
| Macro coverage @ 95% CI (T-scaled Dirichlet) | **0.93** |
| Active-learning entropy acquisition | reaches 90% of full-data MAE on **16–26%** of labels (3 seeds) |
| Bar fraction vs bulge size | model and volunteer Spearman ρ both positive, p < 10⁻²² |

**Release dataset**: `releases/gz_desi_dirichlet_v1.parquet` — per-galaxy 34-dim Dirichlet
concentration vectors for 61,440 GZ DESI DR8 galaxies (also published as the HF dataset
[`jroth1414/galaxy-vit-gz-desi-dirichlet-predictions`](https://huggingface.co/datasets/jroth1414/galaxy-vit-gz-desi-dirichlet-predictions)).

---

## What this repo contains

### Model
- `galaxy_vit/models/zoobot_encoder.py` — Zoobot ConvNeXt-nano encoder loader
- `galaxy_vit/models/vit_baseline.py` — ViT-B/16 baseline for comparison
- `galaxy_vit/models/dirichlet_head.py` — `softplus(linear) + α_floor` head producing strictly positive 34-D concentration vectors

### Loss & inference
- `galaxy_vit/losses/dirichlet_mn.py` — masked Dirichlet–Multinomial NLL (fp32 `gammaln`), per-question normalized
- `galaxy_vit/inference/posterior.py` — analytic per-answer marginals `Beta(α_i, Σα − α_i)`, credible intervals, coverage
- `galaxy_vit/inference/dirichlet_predictor.py` — single-image posterior with parent-dependency greyout
- `galaxy_vit/inference/attention.py` — GradCAM and attention rollout overlays
- `galaxy_vit/training/calibration.py` — binned reliability, ECE, MCE, Brier
- `galaxy_vit/training/active_learning.py` — predictive entropy + closed-form Dirichlet BALD acquisitions

### Data
- `galaxy_vit/data/schema.py` — GZ DESI question / answer / dependency-tree schema pinned to upstream `galaxy-datasets`
- `galaxy_vit/data/gz_desi_hf_dataset.py` — streaming `IterableDataset` over HF `mwalmsley/gz_desi_wds` tar shards
- `galaxy_vit/data/masking.py` — per-question validity mask honoring the decision tree (with `tie_policy` for plurality-tie semantics)

### Trainers
- `galaxy_vit/training/trainer.py` — Galaxy10 baseline (M1)
- `galaxy_vit/training/multi_question_trainer.py` — per-question cross-entropy on GZ DESI
- `galaxy_vit/training/dirichlet_trainer.py` — Dirichlet–Multinomial training with coverage gates

### Demo
- `galaxy_vit/serve/` — FastAPI backend (`/predict`, `/predict_sdss`, `/posteriors`, `/demo_galaxies`, `/umap_points`, `/test_thumbs/{idx}`, plus v2: `/similar`, `/outliers`, `/sky_points`, `/tree_flow`, `/per_question_gradcam`, `/resolve_name`, `/compare`, `/training_movie`)
- `frontend/` — React + Vite SPA, seven tabs:
  - **Classify** — image upload → top-3 + GradCAM; "Compare with M3" runs the Dirichlet head on the same image side-by-side
  - **Posteriors** — per-question Dirichlet bars with 95% CI whiskers, plus a Sankey tree-flow view and a per-question GradCAM selector. Embeds the "most interesting galaxies" outlier panel (entropy / BALD / disagreement)
  - **Explorer** — plotly scattergl over 2,462 UMAP points; 2-D / 3-D toggle, lasso, hover thumbnails crossfade to precomputed GradCAM, click-to-posterior
  - **Sky** — 14k DR8 galaxies on RA/Dec scatter + Aladin Lite (DECaLS DR10) embed; name resolver (M31, NGC 1300, …) via CDS Sesame
  - **Similar** — cosine-kNN against the cached 2,462 ConvNeXt features; query by upload or by cache idx. "Find similar →" buttons on Classify / Posteriors / Explorer deep-link in
  - **Training** — animation of the 24 demo galaxies' positions in feature space across all training epochs (final-epoch UMAP fit projected onto every earlier epoch)
  - **Model Card** — comparison table, curves, interpretability gallery

---

## Quickstart

### Install

```bash
git clone https://github.com/jroth1414/galaxy-vit && cd galaxy-vit
python -m venv .venv && .venv/Scripts/activate   # Linux/Mac: source .venv/bin/activate
pip install -e ".[dev,m1,m1-train,m1-serve]"

# Pick one PyTorch wheel (RTX 50-series needs Blackwell sm_120 from a nightly):
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
# CPU-only (CI, HF Spaces):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

cp .env.example .env   # fill in HF_USER, HF_TOKEN, DATA_DIR
```

### Verify the install

```bash
python -m galaxy_vit.config           # prints redacted settings; exits nonzero on missing vars
pytest -q                             # 324 passed, 2 skipped on a clean checkout (v2 features included)
```

### Run training

```bash
# Galaxy10 ViT baseline
python -m galaxy_vit.training.trainer --config configs/m1_vit_baseline.yaml

# Zoobot ConvNeXt-nano finetune (Galaxy10)
python -m galaxy_vit.training.trainer --config configs/m1_zoobot_finetune.yaml

# GZ DESI per-question cross-entropy (Walmsley+23 reproduction baseline)
python -m galaxy_vit.training.multi_question_trainer --config configs/m2_w23_reproduction.yaml

# GZ DESI Dirichlet–Multinomial head (the main result)
python -m galaxy_vit.training.dirichlet_trainer --config configs/m3_dirichlet.yaml
```

### Run inference and downstream analyses

```bash
# Full-pass inference on the labeled DR8 subset (~2 min on RTX 5070 Ti)
python -m scripts.run_inference_pass --mode full

# Post-hoc temperature calibration sweep
python -m scripts.calibrate_dirichlet \
    --config configs/m3_dirichlet.yaml \
    --checkpoint runs/m3_dirichlet/best.pt \
    --out runs/m3_dirichlet/calibrated_metrics.json

# Active-learning experiment (3 seeds × 3 acquisitions × 10 rounds, ~10 min)
python -m scripts.run_active_learning

# Science case: bar fraction vs bulge prominence
python -m scripts.science_bar_vs_bulge
```

### Run the demo locally

```bash
# Backend (loads M1 + M3 checkpoints from runs/, demo galaxies + UMAP coords from artifacts/)
uvicorn galaxy_vit.serve.app:app --host 0.0.0.0 --port 7860

# Frontend
npm --prefix frontend install
npm --prefix frontend run dev      # http://localhost:5173
# or build + serve via the backend:
npm --prefix frontend run build    # writes frontend/dist; the backend mounts it at /
```

---

## Model & math summary

**Head.** A linear projection of the 640-D ConvNeXt-nano penultimate feature followed
by `softplus(·) + α_floor` (default 1.0) produces a 34-D concentration vector α. Per
question *q* the slice α_q parameterizes a Dirichlet posterior over per-answer
probabilities p_q.

**Loss.** Per-galaxy, per-question Dirichlet–Multinomial NLL on the observed vote
counts:

```
−log P(c | α)
  = log Γ(N + A) − log Γ(A) + Σ_k [log Γ(α_k) − log Γ(α_k + c_k)]
```

with A = Σα_k, N = Σc_k. `torch.special.gammaln` is forced to fp32 even when the
forward pass runs in bf16 — bf16 `gammaln` is unreliable near small α. Per-galaxy NLL
is averaged within each question (over the valid subset), then summed across the 10
questions. Loss-parity vs Zoobot 2.0's reference implementation is verified in
`tests/test_loss_parity.py` (gradient on α matches to < 10⁻³ relative across answer
counts 3–6).

**Per-question masking.** A galaxy's vote on question *q* is valid iff (a) it received
≥ `min_votes` total votes on *q* AND (b) the parent question's plurality answer matches
the gating answer for *q*, recursively. Tie semantics on the parent plurality are
configurable (`tie_policy="argmax"` matches `torch.argmax`; `tie_policy="drop"`
disqualifies all descendants of a tied parent).

**Posteriors.** For each per-answer probability p_i of question *q*, the marginal
posterior is `Beta(α_i, A_q − α_i)` (closed form). 95% credible intervals come from
`scipy.stats.beta.ppf(0.025, ·)` and `ppf(0.975, ·)`.

**Calibration.** The raw head is well-calibrated on cross-entropy training
(macro-ECE = 0.034 across questions) but the Dirichlet head over-concentrates at
training time. Post-hoc temperature scaling (`scripts/calibrate_dirichlet.py`)
recovers coverage = 0.93 at 95% CI while preserving the posterior mean exactly
(`α' = α / T` is scale-invariant on `α_i / Σα`).

**Active learning.** Two acquisition functions over the 10 questions:

- *Predictive entropy*: `−Σ_q Σ_i p_i log p_i` where p is the Dirichlet mean.
- *BALD*: `H[p_pred] − E_α[H[p|α]]` with the closed-form expected-entropy term
  `Σ_i (α_i / A) · (ψ(A+1) − ψ(α_i+1))` (digamma).

On a 2,496-galaxy pool with head-only retraining per round, entropy reaches 90% of
the full-data MAE in 16–26% of labels across 3 seeds; BALD trails entropy in this
small-pool regime.

---

## Reproducibility

- Every training script writes `runs/<id>/run_config.json` with the resolved YAML
  config, git SHA, `pip freeze`, Python and torch versions, and per-shard counts.
- Inference outputs ship with a SHA-256 sidecar in `releases/<id>.meta.json`.
- All scripts accept `--seed` (default 42); seeds propagate to NumPy, PyTorch CPU,
  PyTorch CUDA, and Python's `random`.
- Splits live in `data/splits/*.csv`, committed alongside the code that produced them.
- DECaLS per-channel normalization stats are computed once and frozen in
  `configs/normalization.json`.
- Training curves, metrics JSONs, and figures land under `runs/<id>/` and
  `artifacts/` and are committed; large checkpoints are gitignored but
  regenerable.

---

## Data sources

- **Galaxy10 DECaLS** ([`matthieulel/galaxy10_decals`](https://huggingface.co/datasets/matthieulel/galaxy10_decals)) — 17,736 images, 10-way coarse classes, used for the Month 1 baselines.
- **GZ DESI WebDataset** ([`mwalmsley/gz_desi_wds`](https://huggingface.co/datasets/mwalmsley/gz_desi_wds)) — labeled subset of the Walmsley+23 catalog as tar shards; 80 train / 20 val / 20 test split.
- **GZ DESI volunteer catalog** (Zenodo 8331338) — vote counts + RA/Dec used for `dr8_id` cross-match in the T5.2 science case.

---

## Tech stack

**Training** Python 3.11 · PyTorch nightly (cu128 / sm_120) · `timm` · `transformers` · `webdataset` · `umap-learn` · `scikit-learn` · `galaxy-datasets`

**Backend** FastAPI · Uvicorn · Pydantic v2 · `httpx` · `python-multipart` · `pandas` / `pyarrow`

**Frontend** Vite · React 19 · TypeScript · Tailwind v4 · `react-plotly.js`

**Dev** ruff · mypy (strict on first-party code) · pytest · pre-commit · `detect-secrets`

---

## Numerical landmines worth knowing about

- `lgamma(0) = +∞`. The `α_floor = 1.0` in the head is load-bearing — don't lower it without revisiting `tests/test_head.py`.
- bf16 `gammaln` is unreliable near small α; cast α to fp32 before the loss. The Dirichlet trainer enforces this; `tests/test_loss_parity.py` proves the result matches Zoobot 2.0's fp32 reference.
- `torch.distributions.Beta.icdf` raises `NotImplementedError` in current PyTorch — `inference/posterior.py` uses `scipy.stats.beta.ppf` and converts back. No gradient flows through that path because credible intervals are an inference summary.
- The `__tar_key__` field is injected by `_iter_samples_from_shard` so the inference pass can preserve the `<brick_id>_<object_id>` identifier for downstream cross-match against the volunteer catalog. Don't strip it.
- HF Spaces only allows writes to `/tmp` and idles after ~30 min; cold-start latency is accepted for the demo.

---

## License

Code: MIT. Predictions parquet (release): CC-BY-4.0. Galaxy Zoo and DECaLS upstream data retain their original licenses.

---

## Citation

```bibtex
@misc{galaxy_vit_dirichlet_v1,
  author = {Roth, John},
  title  = {Galaxy-ViT: Bayesian multi-question galaxy morphology with Zoobot + Dirichlet},
  year   = {2026},
  url    = {https://github.com/jroth1414/galaxy-vit},
}
```

Builds on Walmsley+22 (Zoobot foundation model), Walmsley+23 (Galaxy Zoo DESI), and
the `galaxy-datasets` canonical schema. Class-balanced cross-entropy follows
Cui+19. Attention rollout follows Abnar & Zuidema 2020.
