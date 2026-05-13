# Galaxy10 (M1) ↔ GZ DESI Dirichlet (M3) — informal class mapping

The two models answer related but **non-isomorphic** questions.

* **M1** (`runs/m1_zoobot_finetune/best.pt`) is a 10-way softmax
  trained against the [Galaxy10 DECaLS][galaxy10] dataset's
  consensus-plurality labels. One scalar prediction per image.
* **M3** (`runs/m3_dirichlet/best.pt`) is the Dirichlet-Multinomial
  head trained against the full [Galaxy Zoo DESI][gzdesi] decision
  tree (10 questions, 34 answers, parent gating). One Dirichlet vector
  per question per image.

The Galaxy10 classes were curated by collapsing the GZ DESI vote
distribution into 10 plurality buckets. The mapping below is the
**closest correspondence** for each M1 class — but the M3 posterior
is multi-question and probabilistic, so any mapping is necessarily
loose.

## Mapping table

| M1 Galaxy10 class | M3 GZ DESI questions / answers (loose) |
|---|---|
| `disturbed` | `merging` ∈ {minor-disturbance, major-disturbance} |
| `merging` | `merging` = merger |
| `round-smooth` | `smooth-or-featured` = smooth ∧ `how-rounded` = round |
| `in-between-round-smooth` | `smooth-or-featured` = smooth ∧ `how-rounded` = in-between |
| `cigar-shaped-smooth` | `smooth-or-featured` = smooth ∧ `how-rounded` = cigar-shaped |
| `barred-spiral` | `smooth-or-featured` = featured-or-disk ∧ `disk-edge-on` = no ∧ `bar` ∈ {strong, weak} ∧ `has-spiral-arms` = yes |
| `unbarred-tight-spiral` | featured-or-disk ∧ disk-edge-on=no ∧ `bar` = no ∧ `has-spiral-arms` = yes ∧ `spiral-winding` = tight |
| `unbarred-loose-spiral` | featured-or-disk ∧ disk-edge-on=no ∧ `bar` = no ∧ `has-spiral-arms` = yes ∧ `spiral-winding` = loose |
| `edge-on-no-bulge` | featured-or-disk ∧ `disk-edge-on` = yes ∧ `edge-on-bulge` = none |
| `edge-on-with-bulge` | featured-or-disk ∧ `disk-edge-on` = yes ∧ `edge-on-bulge` ∈ {boxy, rounded} |

## Why the mapping is loose

* M1 is **mutually exclusive**: each image has one of 10 labels.
  M3 is **multi-output**: a galaxy can have non-zero posterior mass on
  many decision-tree branches simultaneously.
* The Galaxy10 curation took the dominant volunteer-plurality leaf;
  M3 surfaces the full posterior, including questions the volunteer
  decision tree might never have reached.
* Galaxy10 doesn't expose `how-rounded` cleanly for "smooth + featured
  ambiguous" galaxies, so M3 occasionally has a high featured-or-disk
  posterior on images Galaxy10 labels as round-smooth (and vice-versa).

## Reading the live demo's Compare panel

When the **Compare** sub-panel of the Classify tab is active:

* The left column shows M1's top-3 with the existing GradCAM overlay.
* The right column shows M3's per-question bars (the same view as the
  Posteriors tab).
* The plurality answer for `smooth-or-featured` should usually align
  with the smooth/featured implied by M1's top class. The downstream
  questions (`bar`, `bulge-size`, etc.) carry the additional
  information that M1 doesn't expose.

If they disagree, that's a useful diagnostic — usually it's a galaxy
the volunteers were uncertain about (high `predictive_entropy` in the
S-3 outlier panel) and the mapping above doesn't apply cleanly.

[galaxy10]: https://huggingface.co/datasets/matthieulel/galaxy10_decals
[gzdesi]: https://www.zooniverse.org/projects/zookeeper/galaxy-zoo-desi/
