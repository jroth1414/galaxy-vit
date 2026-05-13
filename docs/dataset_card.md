---
license: cc-by-4.0
language:
  - en
pretty_name: "Galaxy-ViT GZ DESI Dirichlet predictions"
task_categories:
  - tabular-classification
  - image-classification
tags:
  - astronomy
  - galaxy-morphology
  - galaxy-zoo
  - bayesian
  - dirichlet-multinomial
  - zoobot
size_categories:
  - 10K<n<100K
---

# Galaxy-ViT — GZ DESI Dirichlet predictions

Per-galaxy Dirichlet-Multinomial concentration parameters (α) for the
10-question / 34-answer Galaxy Zoo DESI decision tree, predicted by a
Zoobot ConvNeXt-nano encoder finetuned with a Dirichlet-Multinomial
head on the DR8 subset of the
[`mwalmsley/gz_desi_wds`](https://huggingface.co/datasets/mwalmsley/gz_desi_wds)
labeled split.

## Dataset summary

| | |
|---|---|
| Rows | 61,440 |
| Columns | 36 (`key`, `dr8_id`, `alpha_0` … `alpha_33`) |
| Format | Apache Parquet |
| File size | ~16 MB |
| Source images | DECaLS DR8 cutouts at 224×224 |
| Inference time | 128 s on RTX 5070 Ti (479 galaxies/s) |
| SHA-256 | recorded in the companion `.meta.json` sidecar |

Each row carries the model's per-answer Dirichlet concentration. The
posterior mean for answer *i* of question *q* is
`α_i / sum(α_q)`; the marginal posterior on the answer probability is
`Beta(α_i, sum(α_q) − α_i)` (used by the analytic credible-interval
module in `galaxy_vit.inference.posterior`).

## Live demo

These predictions back a 7-tab interactive demo at
<https://<HF_USER>-galaxy-vit-demo.hf.space>: per-question posterior
bars, decision-tree Sankey view, 2-D / 3-D UMAP, RA/Dec sky scatter
with an Aladin Lite DR10 embed, cosine-kNN similarity search, and a
training-progress animation. The v2 changelog is at
[`CHANGELOG.md`](https://github.com/jroth1414/galaxy-vit/blob/main/CHANGELOG.md);
the 90-second walkthrough script is in `docs/loom_shotlist.md`.

## Canonical column order (matches the head's 34-D layout)

| Index | Question | Answer |
|---:|---|---|
| 0 | smooth-or-featured | smooth |
| 1 | smooth-or-featured | featured-or-disk |
| 2 | smooth-or-featured | artifact |
| 3 | disk-edge-on | yes |
| 4 | disk-edge-on | no |
| 5 | has-spiral-arms | yes |
| 6 | has-spiral-arms | no |
| 7 | bar | strong |
| 8 | bar | weak |
| 9 | bar | no |
| 10 | bulge-size | dominant |
| 11 | bulge-size | large |
| 12 | bulge-size | moderate |
| 13 | bulge-size | small |
| 14 | bulge-size | none |
| 15 | how-rounded | round |
| 16 | how-rounded | in-between |
| 17 | how-rounded | cigar-shaped |
| 18 | edge-on-bulge | boxy |
| 19 | edge-on-bulge | none |
| 20 | edge-on-bulge | rounded |
| 21 | spiral-winding | tight |
| 22 | spiral-winding | medium |
| 23 | spiral-winding | loose |
| 24-29 | spiral-arm-count | 1, 2, 3, 4, more-than-4, can't-tell |
| 30 | merging | none |
| 31 | merging | minor-disturbance |
| 32 | merging | major-disturbance |
| 33 | merging | merger |

This ordering is the `galaxy_vit.data.schema.question_index_groups()`
output, derived from upstream
[`galaxy-datasets`](https://pypi.org/project/galaxy-datasets/) and
verified against the legacy hardcoded constants at every checkpoint
load (T3.1).

## Intended use

Research-grade. The model recovers known morphology relations on the
labeled validation subset:

* **Macro top-1 (T2.3 baseline, cross-entropy):** 0.8445 on the DR8
  test split, comparable to Walmsley+23's published per-question
  numbers.
* **Macro vote MAE (T3.6 Dirichlet):** 0.0883 on the DR8 test split.
* **Macro coverage at 95% CI (T3.6 Dirichlet, T-calibrated):** 0.93.
* **Science direction agreement (T5.2):** Spearman bar-vs-bulge
  correlation matches volunteer-observed sign (model ρ = +0.198,
  volunteer ρ = +0.146, both p < 10⁻²²).

Not appropriate for downstream scientific claims without
re-validation against the volunteer-vote ground truth on your sample
of interest.

## How to use

```python
from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download(
    repo_id="<HF_USER>/galaxy-vit-gz-desi-dirichlet-predictions",
    filename="gz_desi_dirichlet_v1.parquet",
    repo_type="dataset",
)
df = pd.read_parquet(path)

# Posterior-mean smooth-or-featured probability (3 answers, indices 0..2):
import numpy as np
sof = df[[f"alpha_{i}" for i in range(3)]].to_numpy()
sof_mean = sof / sof.sum(axis=1, keepdims=True)
df["p_smooth"]           = sof_mean[:, 0]
df["p_featured_or_disk"] = sof_mean[:, 1]
df["p_artifact"]         = sof_mean[:, 2]
```

For analytic 95% credible intervals, see
`galaxy_vit.inference.posterior.credible_interval` in the source
repo.

## Training data

The model was finetuned on the DR8 split of
[`mwalmsley/gz_desi_wds`](https://huggingface.co/datasets/mwalmsley/gz_desi_wds)
under the same canonical question / answer schema published by the
[`galaxy-datasets`](https://pypi.org/project/galaxy-datasets/)
package. Inference predictions in this release cover the same labeled
subset (DR8 train + val + test); the unlabeled 8.67M DESI catalog is
not included.

## Source code

Repository: `github.com/<HF_USER>/galaxy-vit` (see commit
hashes in the companion `.meta.json` for the exact code state that
produced this parquet).

## Citation

```bibtex
@misc{galaxy_vit_dirichlet_v1,
  author = {{Galaxy-ViT contributors}},
  title  = {Galaxy-ViT GZ DESI Dirichlet predictions, v1},
  year   = {2026},
  url    = {https://huggingface.co/datasets/<HF_USER>/galaxy-vit-gz-desi-dirichlet-predictions},
  doi    = {<ZENODO_DOI>},
}
```

Builds on the Galaxy Zoo DESI volunteer effort
([Walmsley et al. 2023](https://arxiv.org/abs/2306.01617)) and the
Zoobot Foundation-Model architecture
([Walmsley et al. 2022](https://arxiv.org/abs/2206.11927)).

## License

Predictions: CC-BY-4.0. Underlying image data and volunteer votes
retain their original licenses (Galaxy Zoo DESI / DECaLS imagery
CC-BY-4.0; vote data Walmsley+23 / Zenodo 8331338).
