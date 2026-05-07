# docs/SCHEMA.md — Galaxy Zoo DESI Decision-Tree Schema Reference

## 1. What This Document Is

A human-readable reference for the GZ DESI decision tree as imported in `galaxy_vit/data/schema.py`. The machine source of truth is the `gz_desi_pairs` and `gz_desi_dependencies` dicts from `galaxy-datasets/shared/label_metadata.py` (imported by Zoobot 2.0). Do NOT hand-edit the schema here; edit the upstream import instead and re-run `T3.1` tests.

## 2. Question / Answer Structure

Each question has a fixed answer set. Every answer appears in the flat output tensor of `DirichletMultinomialHead` exactly once, in the order given by `schema.question_index_groups`.

### Q1 — smooth-or-featured *(always asked)*
- smooth
- featured-or-disk
- artifact

### Q2 — disk-edge-on *(depends on Q1 = featured-or-disk)*
- yes
- no

### Q3 — has-spiral-arms *(depends on Q2 = no)*
- yes
- no

### Q4 — bar *(depends on Q2 = no)*
- strong
- weak
- no

### Q5 — bulge-size *(depends on Q2 = no)*
- dominant
- large
- moderate
- small
- none

### Q6 — how-rounded *(depends on Q1 = smooth)*
- round
- in-between
- cigar-shaped

### Q7 — edge-on-bulge *(depends on Q2 = yes)*
- boxy
- none
- rounded

### Q8 — spiral-winding *(depends on Q3 = yes)*
- tight
- medium
- loose

### Q9 — spiral-arm-count *(depends on Q3 = yes)*
- 1
- 2
- 3
- 4
- more-than-4
- cant-tell

### Q10 — merging *(always asked)*
- none
- minor-disturbance
- major-disturbance
- merger

Total answers across 10 questions: ~34 (exact count determined at runtime from the imported schema).

## 3. Dependency Rules

A question `q` contributes to the loss for galaxy `g` only if:

1. The parent answer that `q` depends on received the plurality of votes in the parent question, AND
2. The number of votes cast on `q` is at least `min_votes` (default 3).

Both conditions are combined in `dirichlet_mn.dirichlet_multinomial_loss` via the per-question mask `m_{g,q}`.

## 4. Invariants the Agent Must Preserve

- `len(schema.questions) == 10` for GZ DESI.
- `sum(end - start for start, end in schema.question_index_groups) == schema.num_answers`.
- Every key in `gz_desi_dependencies` is present as a question name.
- The parent answer named in each dependency exists in the parent question's answer list.
- `smooth-or-featured` and `merging` have no dependencies.
- Reordering the answers within a question breaks all saved checkpoints — never reorder without bumping the run ID.

## 5. Column Naming Convention (Catalog → Tensor)

Verified against `gz_desi_gzd8_volunteer_core_catalog.parquet` from Zenodo 8331338. Walmsley+23 uses three column families:

| Family | Pattern | Example |
|---|---|---|
| Integer vote count per answer | `<question>_<answer>` (no suffix) | `smooth-or-featured_smooth`, `bar_strong`, `merging_merger` |
| Per-question total votes cast | `<question>_total-votes` | `smooth-or-featured_total-votes`, `merging_total-votes` |
| Debiased vote fraction in `[0, 1]` | `<question>_<answer>_fraction` | `smooth-or-featured_smooth_fraction` |

The loader maps the integer-count columns into the flat `votes` tensor using `schema.question_index_groups`. The mapping is unit-tested in `tests/test_schema.py::test_column_to_index_mapping`.

The catalog also carries an auxiliary `anything-odd` question (with `yes`/`no` answers and a matching `anything-odd_total-votes`) that we do **not** include in the 10-question Dirichlet model — it's a free-text catch-all that doesn't fit the decision-tree structure. The validator in `galaxy_vit/data/gz_desi.py` ignores it.

## 5b. Catalog Size — Deviation from DEVPLAN

DEVPLAN T2.1 originally specified ``data/gz_desi_500k.parquet`` with a 400–600k row acceptance gate. After inspecting Zenodo record 8331338's actual contents, we ship a smaller catalog under a more accurate name:

- The ~100k volunteer-vote subset (``gzd8_volunteer_core`` + ``gzd8_volunteer_extended``, both DECaLS-DR8 by construction) is the *only* part of Zenodo 8331338 that contains the integer vote counts the Dirichlet-Multinomial loss requires. The 8.67M-row ``deep_learning_catalog_*`` parquets in the same record contain only Zoobot model-predicted vote *fractions*, not human counts.
- Output file is ``data/gz_desi_volunteer_decals.parquet`` (T2.1 commit). Acceptance gate is **80–150k rows after the ≥5-vote-per-always-asked-question filter**.
- T3 splits + downstream artefacts use the same prefix (e.g. ``data/splits/gz_desi_volunteer_decals_split.csv``).

The smaller catalog is genuinely sufficient for the Bayesian Dirichlet-Multinomial research arc — 100k DECaLS galaxies with real volunteer votes is plenty for calibration, posterior CIs, and active learning. The DEVPLAN's "500k" figure was an early approximation we corrected once we had ground truth on what's published.

## 6. Common Pitfalls

- **Zero votes must be `0`, not `NaN`.** Upstream catalogs sometimes leave deep-tree questions as NaN; the loader replaces NaN with 0.
- **Vote fractions ≠ probabilities.** Fractions are computed per question after volunteer debiasing; do not normalize them yourself.
- **`cant-tell` in Q9** is a legitimate answer, not a missing value.
- **Edge-on disks skip spiral-arm questions entirely.** Coverage metrics for Q3, Q8, Q9 are lower as a result — this is expected, not a bug.
- **Schema can drift between GZ DECaLS (v1/v2) and GZ DESI.** The agent must only use the `gz_desi_*` constants; importing `gz_decals_*` silently produces a subtly wrong tree.

## 7. References

- Walmsley et al. 2023, *Galaxy Zoo DESI: Detailed Morphology Measurements for 8.67M Galaxies*, MNRAS 526, 4768.
- Walmsley et al. 2022, *Galaxy Zoo DECaLS*, MNRAS 509, 3966.
- Zoobot 2.0 docs, *Training on Vote Counts*.
- Galaxy Zoo decision-tree visualization (blog.galaxyzoo.org, 2015).
