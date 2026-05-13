---
license: cc-by-4.0
language:
  - en
pipeline_tag: image-classification
tags:
  - astronomy
  - galaxy-morphology
  - galaxy-zoo
  - bayesian
  - dirichlet-multinomial
  - zoobot
  - convnext
base_model: mwalmsley/zoobot-encoder-convnext_nano
library_name: pytorch
---

# Galaxy-ViT — Zoobot ConvNeXt-nano + Dirichlet-Multinomial head

A Bayesian galaxy morphology classifier: a galaxy-pretrained ConvNeXt-nano encoder
([Zoobot](https://huggingface.co/mwalmsley/zoobot-encoder-convnext_nano)) finetuned
with a Dirichlet-Multinomial output head over the 10-question / 34-answer Galaxy Zoo
DESI decision tree. Produces analytic per-answer credible intervals; tested at
≥ 0.93 coverage at the 95% level after post-hoc temperature scaling.

## Live demo

Hosted at <https://<HF_USER>-galaxy-vit-demo.hf.space> (7 tabs):

- **Classify** — upload → top-3 + GradCAM; "Compare with M3" runs the
  Dirichlet head on the same image side-by-side.
- **Posteriors** — per-question Dirichlet bars with 95% CI whiskers,
  Sankey question-tree view, per-question GradCAM dropdown, and the
  "most interesting galaxies" outlier gallery (entropy / BALD /
  model-vs-volunteer disagreement).
- **Explorer** — 2-D / 3-D UMAP toggle over 2,462 DR8 test galaxies;
  lasso, hover thumbnails crossfade to precomputed GradCAM, click for
  posterior.
- **Sky** — RA/Dec scatter of ~14k joined-catalog galaxies, Aladin
  Lite DR10 embed, and an object-name resolver (M31, NGC 1300, …) via
  CDS Sesame.
- **Similar** — cosine-kNN against the cached ConvNeXt features;
  query by upload or cache idx.
- **Training** — animation of 24 demo galaxies' positions in feature
  space across all training epochs (final-epoch UMAP fit projected onto
  every earlier epoch so the cloud doesn't jitter).
- **Model Card** — this content, plus curves, sample failure cases, and
  the interpretability gallery.

A 90-second walkthrough is in `docs/loom_shotlist.md` in the source
repo. The complete v2 changelog is at
[`CHANGELOG.md`](https://github.com/jroth1414/galaxy-vit/blob/main/CHANGELOG.md).

## Model summary

| | |
|---|---|
| Architecture | ConvNeXt-nano (640-D feature) + `Linear(640, 34)` + `softplus + α_floor` |
| Parameters | ~16M (Zoobot encoder) + 21,794 (head) |
| Input | RGB DECaLS cutout, 224×224 |
| Output | 34-D concentration vector α (positive); per-question Dirichlet posterior |
| Training data | 80 train + 20 val tar shards of [`mwalmsley/gz_desi_wds`](https://huggingface.co/datasets/mwalmsley/gz_desi_wds), DR8 subset only |
| Optimizer | AdamW; encoder LR 1e-5, head LR 1e-3, weight decay 0.05 |
| Schedule | Cosine; 3-epoch warmup; 2 head-only epochs before encoder unfreeze |
| Precision | bf16 forward; `gammaln` and loss in fp32 |
| Early stop | `val/vote_mae_macro` patience 5; converged at epoch 25 of 50 |

## Metrics (held-out DR8 test split)

| Metric | Raw | T-calibrated (T=60) |
|---|---:|---:|
| Macro vote-fraction MAE | 0.0883 | 0.0883 (preserved by design) |
| Macro coverage at 95% CI | 0.49 | **0.93** |
| `all_finite` α throughout training | Yes | — |
| Monotone train loss, epochs 1–5 | Yes | — |

For the cross-entropy *baseline* trained on the same data
(`configs/m2_w23_reproduction.yaml`), macro top-1 = 0.8445 at the per-question
plurality, comparable to Walmsley+23's published numbers.

### Per-question breakdown (Dirichlet, calibrated test set)

| Question | MAE | Coverage |
|---|---:|---:|
| smooth-or-featured | 0.064 | ≥ 0.90 |
| disk-edge-on | 0.092 | ≥ 0.90 |
| has-spiral-arms | 0.124 | ≥ 0.85 |
| bar | 0.102 | ≥ 0.90 |
| bulge-size | 0.089 | ≥ 0.85 |
| how-rounded | 0.067 | ≥ 0.90 |
| edge-on-bulge | 0.099 | ≥ 0.85 |
| spiral-winding | 0.119 | ≥ 0.85 |
| spiral-arm-count | 0.077 | ≥ 0.85 |
| merging | 0.051 | ≥ 0.95 |

Lower-data questions (`spiral-winding`, `spiral-arm-count`, `edge-on-bulge`) have
the wider per-question CIs and the highest MAE; this is the expected pattern when
volunteer disagreement is large per-galaxy.

### Science validation (T5.2)

On the volunteer-cross-matched subset (~4,400 galaxies), model-predicted any-bar
fraction recovers the canonical bar–bulge anti-correlation:

* Spearman ρ vs bulge ordinal: **model +0.198**, **volunteer +0.146**, both
  p < 10⁻²².
* Same sign, same monotone direction, model slightly stronger than the
  noisier volunteer estimator.

## How to use

```python
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

# Download the weights.
ckpt_path = hf_hub_download(
    repo_id="jroth1414/galaxy-vit-zoobot-dirichlet",
    filename="best.pt",
)

# Construct the model (needs the project source for the head wrapper).
from galaxy_vit.models.dirichlet_head import build_zoobot_dirichlet
from galaxy_vit.data.transforms import build_eval_transform, load_normalization

model, _, _ = build_zoobot_dirichlet(num_answers=34, alpha_floor=1.0)
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Predict on a single image.
mean, std = load_normalization("configs/normalization.json")
tf = build_eval_transform(image_size=224, mean=mean, std=std)
img = Image.open("your_galaxy.jpg").convert("RGB")
with torch.no_grad():
    alpha = model(pixel_values=tf(img).unsqueeze(0)).alpha

# alpha has shape (1, 34) in canonical question_index_groups order.
# Posterior mean for the smooth-or-featured 3-answer block:
print((alpha[0, :3] / alpha[0, :3].sum()).tolist())
```

For analytic 95% credible intervals on every answer, see
`galaxy_vit.inference.posterior.credible_interval` in the source repo.

## Calibration regime

The released `best.pt` is the **raw** Dirichlet checkpoint (no temperature applied at
save time). The temperature scalar that recovers coverage = 0.93 at 95% CI is shipped
alongside in `calibrated_metrics.json` (single-T value, default 60.0). Two ways to
use it:

- For **per-image discriminability** (live demo, model card visualizations): use raw
  α. Per-galaxy CIs visibly differ between galaxies.
- For **calibrated coverage** (population statistics, science cases): divide α by T
  before computing posteriors. The posterior mean is unchanged (scale invariance).

The release script's default is raw; opt in to calibrated with
`scripts/calibrate_dirichlet.py`.

## Intended use

Research-grade galaxy morphology classification on DECaLS-style cutouts. The model
was trained only on the Walmsley+23 DR8 subset; predictions on substantially
out-of-distribution imagery (HSC, JWST, deep-field) should be re-validated.

## Limitations and biases

- **Distribution match**: training images are DECaLS DR8 at ~3-color compositing.
  Models exhibit known sensitivity to background subtraction, PSF, and pixel scale.
- **Galaxy Zoo volunteer labels are noisy**: typical inter-volunteer agreement on
  fine questions (`spiral-winding`, `spiral-arm-count`) is well below 80%; the
  practical accuracy ceiling on those questions is similar.
- **Imbalanced class support**: the labeled training subset is dominated by smooth
  galaxies (~80%). The model is well-calibrated on this prior; deploying on a
  population with a different morphology prior will need re-calibration.
- **Decision-tree gating**: per-question predictions for `disk-edge-on`,
  `has-spiral-arms`, `bar`, `bulge-size`, `how-rounded`, `edge-on-bulge`,
  `spiral-winding`, `spiral-arm-count` are only meaningful for galaxies whose
  upstream classifications match the gating answers. The `active` flag in the
  serving response handles this for downstream consumers.
- **bf16 numerical caveat**: the model trains in bf16 but `gammaln` and the loss
  run in fp32. Don't try to inference the head's loss in pure bf16; the head's
  forward in bf16 is fine.

## Training data

The labeled DR8 subset of the GZ DESI WebDataset
([`mwalmsley/gz_desi_wds`](https://huggingface.co/datasets/mwalmsley/gz_desi_wds)):
80 train shards, 20 val shards, 20 held-out test shards, totaling ~80k labeled
galaxies after the `has_any_dr8_votes` filter. Vote counts follow the canonical
Walmsley+23 schema; the project schema module
(`galaxy_vit.data.schema`) pins answer order against upstream
[`galaxy-datasets`](https://pypi.org/project/galaxy-datasets/) and asserts
canonical-ordering parity at every checkpoint load.

## Companion release

Per-galaxy Dirichlet predictions for the 61,440-row DR8 inference set are
published as the dataset
[`jroth1414/galaxy-vit-gz-desi-dirichlet-predictions`](https://huggingface.co/datasets/jroth1414/galaxy-vit-gz-desi-dirichlet-predictions).
That dataset's `key` and `dr8_id` columns let downstream users cross-match the
predictions against the Walmsley+23 volunteer catalog.

## Source code

[`github.com/jroth1414/galaxy-vit`](https://github.com/jroth1414/galaxy-vit) — full
training, inference, serving, and frontend code. The exact commit that produced
`best.pt` is recorded in `runs/m3_dirichlet/run_config.json`.

## Citation

```bibtex
@misc{galaxy_vit_zoobot_dirichlet_v1,
  author = {Roth, John},
  title  = {Galaxy-ViT: Bayesian multi-question galaxy morphology with Zoobot + Dirichlet},
  year   = {2026},
  url    = {https://huggingface.co/jroth1414/galaxy-vit-zoobot-dirichlet},
}
```

Built on Walmsley+22 (Zoobot foundation model), Walmsley+23 (Galaxy Zoo DESI), and
the [`galaxy-datasets`](https://pypi.org/project/galaxy-datasets/) canonical
schema.

## License

Model weights: CC-BY-4.0 (matches Zoobot upstream). Underlying training data
retains its original Galaxy Zoo / DECaLS licenses.
