# Galaxy-ViT v2 — Visualization features plan

Handoff document for the next implementation session. Designed to be
picked up cold after a context-window reset.

---

## 1. Where the project is

### Phases complete

| Phase | Status |
|---|---|
| T0.1 – T5.3 | ✅ shipped, all tests green |
| T6.2 | ✅ HF model release (published to `roth1414/galaxy-vit-zoobot-dirichlet`) |
| T6.3 | ✅ live-demo infra (uptime monitor + integration tests + Loom shot-list) |
| T6.1 (paper) | ⏳ deliberately deferred to the end |

### Current branch + uncommitted state

- On branch `main`
- Uncommitted: `frontend/src/Explorer.tsx` (the `react-plotly.js`
  factory-pattern fix for the Plotly default-export issue — should
  be committed before starting v2)
- Untracked: `docs/_dataset_card_rendered.md`, `docs/_model_card_rendered.md`
  (publish artifacts; safe to leave or `.gitignore`)

### What's running locally

If the demo dev servers from the previous session are still up:
- Backend: `http://127.0.0.1:8000` (uvicorn on `galaxy_vit.serve.app:app`)
- Frontend: `http://localhost:5173` (Vite dev server, proxies `/api` to 8000)

If not, restart with:
```powershell
.venv\Scripts\python -m uvicorn galaxy_vit.serve.app:app --host 127.0.0.1 --port 8000
# in another terminal:
npm --prefix frontend run dev
```

### Test suite headline numbers

- 189 passed, 1 skipped, 0 xfailed
- ruff clean, mypy clean (38 source files)
- CI green on `27e6258` (the fix-CI commit; older red runs are
  historical and won't re-run)

### Released artifacts

- Dataset: https://huggingface.co/datasets/roth1414/galaxy-vit-gz-desi-dirichlet-predictions
- Model: https://huggingface.co/roth1414/galaxy-vit-zoobot-dirichlet
- GitHub: https://github.com/jroth1414/galaxy-vit (public)

---

## 2. Tech reference (copy-paste-able)

```
Python  : 3.11 in .venv (Windows; .venv/Scripts/python)
PyTorch : nightly cu128, sm_120 (RTX 5070 Ti)
Frontend: Vite + React 19 + TS + Tailwind v4 + react-plotly.js + plotly.js-dist-min
Backend : FastAPI + Uvicorn + Pydantic v2 + pandas + httpx

Local checkpoints (gitignored .pt; meta JSONs committed):
  runs/m1_zoobot_finetune/best.pt        # Galaxy10 ViT/Zoobot baseline
  runs/m3_dirichlet/best.pt              # The Dirichlet model

Local artifacts (committed):
  artifacts/demo_galaxies/manifest.json + thumbs/0000..0023.jpg
  artifacts/test_thumbs/00000..02461.jpg
  artifacts/umap_coords.parquet
  artifacts/umap_metrics.json

  releases/gz_desi_dirichlet_v1.parquet  # 61,440 rows × 36 cols (key, dr8_id, alpha_0..33)
  releases/gz_desi_dirichlet_v1.meta.json
  data/gz_desi_volunteer_decals.parquet  # 102,130 rows; has dr8_id, ra, dec, votes

Demo galaxies share dr8_id format "8000_<brick>_<object>" between
inference parquet and volunteer catalog -> 14,469 galaxies cross-match.
```

Canonical alpha column layout (matches `schema.question_index_groups()`):

| idx | question | answers |
|---:|---|---|
| 0-2 | smooth-or-featured | smooth, featured-or-disk, artifact |
| 3-4 | disk-edge-on | yes, no |
| 5-6 | has-spiral-arms | yes, no |
| 7-9 | bar | strong, weak, no |
| 10-14 | bulge-size | dominant, large, moderate, small, none |
| 15-17 | how-rounded | round, in-between, cigar-shaped |
| 18-20 | edge-on-bulge | boxy, none, rounded |
| 21-23 | spiral-winding | tight, medium, loose |
| 24-29 | spiral-arm-count | 1, 2, 3, 4, more-than-4, cant-tell |
| 30-33 | merging | none, minor-disturbance, major-disturbance, merger |

---

## 3. Implementation plan

### Recommended order

```
S-1 → S-3 → S-4 → A-6 → S-2 → A-5 → A-7 → A-8 → C-16 → C-15
```

Rationale:
- **S-1 first** because every other feature wants its `<SampleGrid>` component
  and the cached feature vectors land here
- **S-3, S-4 next** are cheap wins that compound visually
- **A-6 (3D UMAP)** is the cheapest A-tier; ships before the harder ones
- **S-2 (sky + Aladin)** is the credibility upgrade; do after we have a
  solid base of features to bounce between
- **A-5 (Sankey)** is the conceptual capstone
- **A-7 (per-question GradCAM)** is risky; do after the Posteriors tab
  has the question-selector UI from A-5
- **A-8 (RA/Dec input)** is a small QoL win for the sky tab
- **C-16 (multi-model compare)** before **C-15 (training movie)** because
  C-15 requires a full retraining pass

### Pre-flight checklist (do once before starting)

1. Commit the Explorer.tsx fix that's currently uncommitted:
   ```powershell
   git add frontend/src/Explorer.tsx
   git commit -m "fix(frontend): react-plotly.js factory pattern for Vite + React 19 ESM"
   git push
   ```
2. Confirm the demo runs locally end-to-end (all 4 tabs) before adding features
3. Decide on a versioning scheme. Suggestion: each tier-S feature is a
   single PR-equivalent commit on a `v2/<feature>` branch off main, then
   merged to main when done; tier-A features may need 2 commits each

---

## 4. Tier S — high impact, low effort

### S-1. Similar-galaxy kNN search

**Scope.** User uploads an image OR clicks any galaxy thumbnail. Backend
encodes through the Zoobot encoder → 640-D feature, computes cosine
distance against the 2,462 cached UMAP-set features, returns top-K
nearest neighbors as `{idx, distance, thumbnail_url}`. Frontend renders
a 4×5 thumbnail grid.

**Astronomical credibility note.** This is exactly how the
"morphology-similar" feature works in upstream catalog cross-matchers
(Walmsley+22 §4.3). The features come from a galaxy-pretrained encoder
so the similarity reflects morphology, not just pixel-level
brightness/color.

**Files to create / modify.**

| Path | Action | LOC |
|---|---|---:|
| `galaxy_vit/inference/similarity.py` | new — load + cache features, cosine kNN | ~80 |
| `scripts/cache_test_thumb_features.py` | new — precompute features for all 2,462 test thumbnails (mirrors `extract_umap.py`'s iteration order so the kNN indices align with `umap_coords.parquet` / `test_thumbs/<idx>.jpg`) | ~100 |
| `artifacts/test_thumb_features.parquet` | new — committed, ~6 MB (2462 × 640 × 4 bytes) | — |
| `galaxy_vit/serve/schemas.py` | add `SimilarGalaxyItem`, `SimilarGalaxiesResponse` | ~30 |
| `galaxy_vit/serve/app.py` | add `POST /api/similar` (image upload), `GET /api/similar/{idx}?k=20` (by existing thumb idx) | ~50 |
| `frontend/src/SampleGrid.tsx` | new — reusable component (extract from Explorer.tsx's sample-grid render) | ~80 |
| `frontend/src/SimilarGalaxies.tsx` | new — search input + sample-grid result | ~120 |
| `frontend/src/{Classify,Posteriors,Explorer}.tsx` | add "find similar" button hooks | ~30 |
| `tests/test_similarity.py` | unit tests: cosine ordering, deterministic with seed, top-K bounds | ~80 |
| `tests/test_v2_similar_api.py` | API tests for both endpoints | ~80 |

**Acceptance criteria.**
- Sample-grid for `/api/similar/0` (idx 0 itself) returns idx 0 as the
  first result with distance ≈ 0
- Top-20 results are stable across two calls (deterministic)
- Cosine distance ∈ [0, 2] always
- Frontend "find similar" buttons in three tabs all open the same panel

**Gotchas.**
- Feature cache lives in memory after first load; bound at ~6 MB so
  this is fine. Don't re-load per request.
- Use `torch.no_grad()` in the encoding path; the encoder is the
  bottleneck (~50 ms/image on CPU).
- The 2,462 thumbnails are at 128×128 but the encoder was trained at
  224×224. Resize before encoding (the existing
  `build_eval_transform` does this).

**Effort estimate.** ½ day.

---

### S-2. Sky map + Aladin Lite embed

**Scope.** A new "Sky" tab with two sub-views:

(a) **Sky scatter view** — plot the 14,469 matched DR8 galaxies on an
RA/Dec scatter (Plotly scattergl, optional Mollweide projection).
Color by predicted smooth-or-featured class or by entropy. Hover →
thumbnail; click → posterior summary panel.

(b) **Aladin Lite embed** — iframe of `aladin.cds.unistra.fr/AladinLite`
configured to show DECaLS DR10 imagery. JavaScript message handler
captures the user's click coordinates (RA/Dec), posts them to
`/api/predict_sdss`, displays the top-3 + GradCAM as an overlay panel.

**Astronomical credibility note.** Reframes the demo from "ML toy" to
"hooked to a real telescope survey." Aladin Lite's CDS license requires
their logo/attribution to stay visible — do not strip it.

**Files to create / modify.**

| Path | Action | LOC |
|---|---|---:|
| `galaxy_vit/serve/schemas.py` | add `SkyPoint`, `SkyPointsResponse` | ~30 |
| `galaxy_vit/serve/app.py` | add `GET /api/sky_points` (joins inference + volunteer on dr8_id, returns ra/dec/idx/label) | ~50 |
| `frontend/src/Sky.tsx` | new — tab with two sub-views | ~280 |
| `frontend/src/App.tsx` | add 5th tab (Sky) | ~10 |
| `tests/test_v2_sky_api.py` | API tests: response shape, ra/dec ranges, ≥10k points | ~50 |

**Acceptance criteria.**
- `/api/sky_points` returns ≥10,000 entries (~14k expected)
- Every point has `ra ∈ [0, 360]`, `dec ∈ [-90, 90]`
- Aladin Lite iframe loads without console errors; clicking populates
  RA/Dec into the prediction panel
- CDS attribution visible in the iframe (don't disable it)

**Gotchas.**
- The Aladin Lite v3 API uses postMessage; cross-origin iframe needs
  the right `sandbox` attribute. Test the click handler in a fresh
  browser session, not just dev tools.
- SDSS cutout fetches can be slow / rate-limited; the existing
  `/predict_sdss` endpoint has retry logic. Don't hammer it.
- For Mollweide projection, Plotly's `scattergeo` works but is
  heavyweight; consider regular 2D scatter with axis labels first.

**Effort estimate.** 1 full day (Aladin Lite integration is the long pole).

---

### S-3. "Most interesting galaxies" outlier panel

**Scope.** Precompute three sort orders on the 61,440-row inference
parquet:
1. Predictive entropy (sum across questions, on Dirichlet means)
2. BALD score (closed-form digamma version from
   `galaxy_vit/training/active_learning.py`)
3. |model − volunteer| per-answer L1 distance (only for the 14k joined
   subset)

Serve top-K thumbnails per metric. Frontend shows three side-by-side
columns with the gallery, labeled with the metric value.

**Files to create / modify.**

| Path | Action | LOC |
|---|---|---:|
| `galaxy_vit/inference/outliers.py` | new — sort logic on parquet | ~80 |
| `scripts/build_outlier_indices.py` | new — precompute, write `artifacts/outliers.json` (small) | ~80 |
| `artifacts/outliers.json` | new — `{entropy: [idx, ...], bald: [...], disagreement: [...]}`, top-100 per metric | — |
| `galaxy_vit/serve/app.py` | add `GET /api/outliers?metric=entropy|bald|disagreement&k=20` | ~30 |
| `frontend/src/Outliers.tsx` | new — three-column gallery | ~150 |
| `frontend/src/Posteriors.tsx` | embed Outliers as a section in the existing tab | ~10 |
| `tests/test_outliers.py` | sort ordering, top-K bounds, metric values | ~60 |

**Acceptance criteria.**
- Three sort orders are stable across runs (deterministic)
- Galaxies in the entropy/BALD lists have visibly higher predictive
  uncertainty than the median
- The disagreement list contains only galaxies in the joined subset
  (where volunteer votes exist)

**Gotchas.**
- The disagreement metric needs the volunteer fractions; reuse
  `expected_fractions` from `losses/dirichlet_mn.py`
- Some test-thumb indices won't have inference parquet rows (the
  inference pass covered train+val+test = 61k, but our umap_coords
  parquet's 2,462 rows are val + test only). When linking from outliers
  back to a thumbnail, make sure the mapping is correct.

**Effort estimate.** ½ day.

---

### S-4. Saliency animation on hover

**Scope.** Precompute GradCAM overlays for all 2,462 UMAP-set galaxies
(the same set that has thumbnails). Frontend Explorer-tab hover
handler: thumbnail crossfades to GradCAM (300ms) and back on unhover.

**Files to create / modify.**

| Path | Action | LOC |
|---|---|---:|
| `scripts/build_test_saliencies.py` | new — iterate test_thumbs, compute GradCAM, save to `artifacts/test_saliencies/<idx>.jpg` | ~100 |
| `artifacts/test_saliencies/*.jpg` | new — 2,462 × ~6 KB = ~15 MB | — |
| `.gitignore` | allowlist `!/artifacts/test_saliencies/*.jpg` | ~3 |
| `galaxy_vit/serve/app.py` | add `GET /api/test_thumbs/{idx}/saliency` | ~10 |
| `frontend/src/Explorer.tsx` | add `<img>` overlay with CSS keyframe crossfade | ~30 |
| `tests/test_v2_saliency.py` | endpoint smoke; verify file shape | ~30 |

**Acceptance criteria.**
- 2,462 saliency JPEGs exist on disk
- `/api/test_thumbs/0/saliency` returns valid JPEG content
- Hover on a UMAP point in Explorer visibly crossfades to the saliency
  (verify manually; integration test confirms the endpoint returns 200)

**Gotchas.**
- GradCAM uses the M1 Galaxy10 model by default. For Dirichlet
  per-question GradCAMs, see A-7. Decide which model's saliency to
  show here; default to whatever GradCAM is shown in Classify (M1).
- The 15 MB of new JPEGs are committable but bring the repo size to
  ~80 MB. Consider compressing to 64×64 or quality=70 if size matters.

**Effort estimate.** ½ day (mostly waiting on the precompute run).

---

## 5. Tier A — high impact, medium effort

### A-5. GZ DESI question-tree Sankey diagram

**Scope.** Render the 10-question / 34-answer GZ DESI decision tree as
a Sankey diagram for a selected galaxy. Node widths = model's
predicted probability of reaching that node (factoring in the
parent-dependency chain). Greyed-out branches show the parts the
model's prediction would not have reached.

**Why it pops.** Single clearest visual explanation of the project's
central thesis (multi-question + Bayesian + decision-tree-aware).

**Files to create / modify.**

| Path | Action | LOC |
|---|---|---:|
| `galaxy_vit/inference/tree_flow.py` | new — given alpha, compute the cumulative reach probability of every node | ~120 |
| `galaxy_vit/serve/schemas.py` | add `TreeNode`, `TreeFlowResponse` | ~40 |
| `galaxy_vit/serve/app.py` | add `POST /api/tree_flow` (image upload) and `GET /api/tree_flow/test_thumbs/{idx}` | ~40 |
| `frontend/src/QuestionTree.tsx` | new — Plotly Sankey trace; build links from `schema.get_dependencies()` | ~250 |
| `frontend/src/Posteriors.tsx` | toggle button: "Bar view" / "Tree view" | ~30 |
| `tests/test_v2_tree_flow.py` | reach-probability math: leaf prob ≤ parent prob; sum at each branching node = parent prob | ~80 |

**Acceptance criteria.**
- For a galaxy where smooth-or-featured P(smooth) > 0.9, the
  how-rounded subtree's reach probability ≈ 0.9, and the
  disk-edge-on subtree's reach probability ≈ 0.1
- Reach probability of any leaf node ≤ reach probability of its parent
  (monotonicity)
- The visualization correctly greys nodes whose reach prob < threshold

**Gotchas.**
- The Sankey math is multiplicative through the tree (parent reach ×
  parent's predicted prob of the gating answer). Don't accidentally
  multiply the parent's prob of ALL answers, only the gating one.
- Plotly's Sankey is awkward for nested-question groups; consider a
  radial dendrogram (d3-sankey or d3-hierarchy) if the result looks
  cluttered. Decide after prototyping.

**Effort estimate.** 1 full day.

---

### A-6. 3D UMAP

**Scope.** Re-run `scripts/extract_umap.py` with `n_components=3`,
save `artifacts/umap_3d_coords.parquet` alongside the 2D version.
Frontend Explorer tab gets a "2D / 3D" toggle button; Plotly
`scatter3d` renders the 3D cloud with built-in rotation/zoom.

**Files to create / modify.**

| Path | Action | LOC |
|---|---|---:|
| `scripts/extract_umap.py` | accept `--n-dims` flag (default 2; rerun with 3) | ~30 |
| `artifacts/umap_3d_coords.parquet` | new — same row order as 2D version | — |
| `galaxy_vit/serve/app.py` | extend `/api/umap_points` with `?n_dims=2|3` | ~30 |
| `galaxy_vit/serve/schemas.py` | `UMAPPoint` gets optional `z` field | ~10 |
| `frontend/src/Explorer.tsx` | toggle, conditional 2D Plot vs 3D Plot | ~80 |

**Acceptance criteria.**
- `/api/umap_points?n_dims=3` returns 2,462 points with `x`, `y`, `z`
- The 3D coords file's row order matches the 2D file's
- Frontend toggle preserves the selection/hover state across views

**Gotchas.**
- UMAP 3D is not just the 2D projection with a z-axis; it's a
  re-fit. Don't expect 2D coords to be a slice of 3D coords.
- Plotly `scatter3d` lasso doesn't work the same way as 2D; consider
  disabling lasso in 3D mode and falling back to box-select.

**Effort estimate.** ½ day.

---

### A-7. Per-question GradCAM

**Scope.** Currently GradCAM uses the M1 model's logit gradient.
Extend to per-question: for each of the 10 questions, compute GradCAM
gated on that question's alpha-sum-gradient. Frontend dropdown in
Posteriors tab picks the question; overlay updates.

**Why it pops.** Directly visualizes the model's multi-question
reasoning. "Look — for the bar question it focuses on the central
bulge; for spiral-arm-count it focuses on the disk."

**Files to create / modify.**

| Path | Action | LOC |
|---|---|---:|
| `galaxy_vit/inference/attention.py` | new function `per_question_gradcam(model, image, question_idx)` | ~120 |
| `galaxy_vit/serve/app.py` | new endpoint `POST /api/per_question_gradcam?question=bar` | ~50 |
| `frontend/src/Posteriors.tsx` | dropdown selector + overlay swap | ~80 |
| `tests/test_v2_per_question_gradcam.py` | numerical: heatmap is non-zero, finite, in [0, 1] | ~60 |

**Acceptance criteria.**
- The 10 per-question heatmaps visibly differ from each other on a
  single image (qualitative check)
- All values are finite, non-NaN
- For a galaxy where the model predicts "no bar," the bar-question
  GradCAM should look qualitatively different from a "strong-bar"
  galaxy

**Gotchas.**
- GradCAM needs `requires_grad=True` on the input or a `register_hook`
  on an intermediate feature map. The existing `gradcam` function
  in `attention.py` uses the forward-hook pattern; copy that.
- Per-question gradient = gradient of the alpha slice's sum, not the
  posterior mean. Sum is more numerically stable.
- Risk: GradCAM at the per-question level can be noisy. If results
  look unconvincing, fall back to attention rollout (more robust but
  heavier).

**Effort estimate.** 1 full day (the visual debugging is the long pole).

---

### A-8. Real-time RA/Dec input + name resolver

**Scope.** A small input field accepting either
"RA Dec" coordinates (decimal degrees) OR an object name (e.g.
"NGC 1300", "M31"). On submit, resolve the name via Simbad if
needed, then call `/predict_sdss?ra&dec`. Display the cutout, top-3,
and full posterior bars.

**Files to create / modify.**

| Path | Action | LOC |
|---|---|---:|
| `galaxy_vit/serve/sdss.py` | add `resolve_object_name(name) -> (ra, dec)` using astroquery.simbad | ~60 |
| `galaxy_vit/serve/app.py` | new endpoint `GET /api/resolve_name?name=NGC+1300` | ~30 |
| `frontend/src/Sky.tsx` (or Classify) | input field + result panel | ~120 |
| `pyproject.toml` | add `astroquery>=0.4,<1` to `[m1-serve]` | ~5 |
| `tests/test_v2_resolve_name.py` | mocked Simbad response; format validation | ~50 |

**Acceptance criteria.**
- "M31" resolves to RA ≈ 10.68, Dec ≈ 41.27 (within 0.01°)
- Invalid object name returns 404 with a clear error
- The prediction flow works end-to-end for a known galaxy name

**Gotchas.**
- Simbad has rate limits; cache resolved names in-memory (OrderedDict
  LRU like the existing attention cache).
- Some objects (Local Group, very bright nearby galaxies) are too
  large for a 224×224 DECaLS cutout. The Classify model may give
  nonsense predictions on M31 because the cutout captures only a
  tiny piece of the galaxy. Document this as a known limitation.

**Effort estimate.** ½ day.

---

## 6. Tier C — risky / expensive (do last)

### C-15. Real-time training visualization

**Scope.** Animate the model's "learning" of a specific set of demo
galaxies across training epochs. For each epoch of the T3.6 training
run, extract the encoder's features for the 24 demo galaxies, then
play back the UMAP-embedded trajectory.

**Why it pops.** Visceral "the model is getting better" demo.

**Cost.**
- Storage: full per-epoch checkpoints = 30 × 60 MB = 1.8 GB (NO,
  don't do this). Instead: save per-epoch features for the 24 demo
  galaxies = 24 × 640 × 30 × 4 bytes = 1.8 MB ✅
- Compute: re-run T3.6 training (~3 hours on RTX 5070 Ti) with the
  modified trainer that saves per-epoch features

**Files to create / modify.**

| Path | Action | LOC |
|---|---|---:|
| `galaxy_vit/training/dirichlet_trainer.py` | add per-epoch hook: extract features for a fixed eval set after each epoch, append to a parquet | ~80 |
| `configs/m3_dirichlet.yaml` | new field `logging.per_epoch_features_path` | ~5 |
| `artifacts/per_epoch_demo_features.parquet` | new — one row per (epoch, galaxy_idx) = ~720 rows × 640 floats | — |
| `scripts/build_training_movie.py` | new — UMAP-embed each epoch's features (or PCA for speed); save (epoch, idx, x, y) parquet | ~120 |
| `galaxy_vit/serve/app.py` | new endpoint `GET /api/training_history` | ~30 |
| `frontend/src/TrainingMovie.tsx` | new tab — slider for epoch + animated points | ~250 |
| `tests/test_v2_training_movie.py` | data shape, monotone metric improvement | ~50 |

**Acceptance criteria.**
- Demo galaxies' UMAP positions visibly migrate across epochs
- Final positions (epoch 30) approximately match the static T2.4
  UMAP for those same galaxies
- The animation runs at ≥15 fps in the browser

**Gotchas.**
- ~3 hours of GPU time for the retraining. Plan the run overnight.
- UMAP is stochastic; if you re-fit per epoch, the cloud will jitter
  unrelated to training progress. Solutions: (a) fit UMAP once on
  the FINAL epoch, then project each earlier epoch's features
  through the same UMAP. (b) use PCA instead of UMAP (deterministic).
  Option (a) is cleaner.
- Don't ship the per-epoch checkpoints to HF — keep them local. Only
  the precomputed parquet of UMAP coords is needed at runtime.

**Effort estimate.** 2 days (1 day for code, 3 hours for the run, 1 day
for UI polish).

---

### C-16. Multi-model comparison (M1 vs M3 side-by-side)

**Scope.** Side-by-side panel: same image goes to both M1 (Galaxy10
plurality) and M3 (Dirichlet); user sees M1's top-3 plurality bars
next to M3's 10-question posterior panel. A small mapping table
points out which M3 questions correspond loosely to which M1 classes.

**Files to create / modify.**

| Path | Action | LOC |
|---|---|---:|
| `galaxy_vit/serve/app.py` | new endpoint `POST /api/compare` (returns both M1 and M3 predictions on the same image) | ~60 |
| `galaxy_vit/serve/schemas.py` | `CompareResponse` wrapping both | ~30 |
| `docs/m1_to_m3_mapping.md` | new — table mapping Galaxy10 classes to GZ DESI tree leaves | ~50 |
| `frontend/src/Compare.tsx` | new tab — split panel | ~200 |
| `tests/test_v2_compare.py` | endpoint shape, both predictions populated | ~50 |

**Acceptance criteria.**
- `/api/compare` with a featured-or-disk barred-spiral image returns
  M1 top-1 ∈ {barred-spiral, unbarred-tight-spiral, ...} AND M3
  posterior with smooth-or-featured P(featured) > 0.7 AND bar P(any
  bar) > 0.5
- Tab renders both panels without layout overflow at 1280×720

**Gotchas.**
- M1 and M3 use different image transforms (different normalization?
  Probably the same, but verify). If different, encode the image
  with both pipelines.
- The Galaxy10 classes don't map 1:1 to GZ DESI questions; the
  mapping is loose. Be honest about this in the UI (a small "see
  mapping table" link).
- This feature adds clutter; consider making it a sub-panel inside
  Classify rather than a 5th tab.

**Effort estimate.** 1 day.

---

## 7. Cross-cutting work (do once at the start)

### Component extraction
- Extract `<SampleGrid>` from Explorer.tsx into `frontend/src/SampleGrid.tsx`
- Extract `<PosteriorBars>` from Posteriors.tsx into a reusable component
- This unblocks S-1, S-2, S-3 (all of which want to render thumbnail grids)

### Backend feature cache
S-1 needs the 2,462 test-thumb feature vectors cached. Once cached,
S-3 (outliers) and A-7 (per-question GradCAM) can also use them. Build
this in S-1 and design the cache to be reusable.

### Test infrastructure
- Add `tests/conftest.py` with a session-scoped `TestClient` fixture
  shared across the new `test_v2_*` files (the current per-file
  fixtures duplicate the lifespan setup, ~5 s per file)
- Decide on a `@pytest.mark.v2` marker so we can run only the v2
  tests during iteration: `pytest -m v2`

---

## 8. Summary table

| ID | Feature | Tier | Effort | Storage Δ | Compute Δ | Key files |
|---|---|---|---:|---:|---:|---|
| S-1 | Similar-galaxy kNN | S | 0.5 d | +6 MB | one-shot ~5 min | `inference/similarity.py`, `SimilarGalaxies.tsx` |
| S-2 | Sky map + Aladin | S | 1.0 d | 0 | 0 | `Sky.tsx`, new `/api/sky_points` |
| S-3 | Outlier panel | S | 0.5 d | +10 KB | one-shot ~30 s | `inference/outliers.py`, `Outliers.tsx` |
| S-4 | Saliency animation | S | 0.5 d | +15 MB | one-shot ~10 min | `scripts/build_test_saliencies.py` |
| A-5 | Question-tree Sankey | A | 1.0 d | 0 | 0 | `inference/tree_flow.py`, `QuestionTree.tsx` |
| A-6 | 3D UMAP | A | 0.5 d | +50 KB | one-shot ~5 min | extend `extract_umap.py`, Explorer toggle |
| A-7 | Per-question GradCAM | A | 1.0 d | 0 | per-request | extend `inference/attention.py` |
| A-8 | RA/Dec name resolver | A | 0.5 d | 0 | per-request | `serve/sdss.py`, astroquery |
| C-16 | M1 vs M3 compare | C | 1.0 d | 0 | per-request | `Compare.tsx`, mapping doc |
| C-15 | Training movie | C | 2.0 d | +2 MB | ~3 h retrain | retrain T3.6, `TrainingMovie.tsx` |

**Total effort**: ~8.5 days for everything; ~2.5 days for Tier S alone;
~3 more days for Tier A; ~3 more for the two Tier C items.

---

## 9. After-feature housekeeping (do at the end)

- Update `README.md` headline metrics table with any new features
- Update `docs/loom_shotlist.md` to incorporate the new tabs into the
  60-second demo flow (probably stretch to 90 s)
- Re-run `scripts/release_to_hf.py --publish` and
  `scripts/release_model_to_hf.py --publish` so the HF Hub cards
  carry the updated demo URL / Loom link
- Bump the project version (e.g., add a CHANGELOG.md tracking
  v1 → v2 features)
- Update CI to run the new test files (no-op if you use the
  `tests/test_v2_*.py` naming convention since pytest auto-discovers)
- T6.1 paper: now is the time. The paper can cite the v2 demo's
  richer feature set as the headline.

---

## 10. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Aladin Lite iframe cross-origin click handler doesn't fire | Medium | Test in a fresh browser before relying on it; have a fallback "type RA/Dec manually" UI |
| Per-question GradCAM produces noisy / unconvincing heatmaps | Medium | Have attention rollout as backup; if both fail, drop A-7 and mark as future work |
| C-15 retraining run doesn't converge to similar metrics as the original | Low | Seed everything; if metrics drift, document and use the new run as canonical |
| Repo size balloons past 100 MB and GitHub starts complaining | Low | The new artifacts add ~30 MB. Still well under 1 GB. Use Git LFS only if we hit a real limit. |
| Astroquery / Simbad rate-limits the demo on a busy day | Low | LRU cache resolved names; document the limit |
| react-plotly.js factory pattern breaks again on a minor version bump | Low | Pin both `react-plotly.js` and `plotly.js-dist-min` exactly; the unwrap helper in Explorer.tsx handles both shapes |

---

## 11. How to start the next session

Suggested opening prompt for the agent:

> Read `docs/v2_features_plan.md` first. Then start with feature **S-1
> (Similar-galaxy kNN search)**. Follow the file table; commit each
> file as you go. Before any commit, run
> `pytest -q && ruff check . && mypy galaxy_vit` and fix anything red.
> When S-1 is done and pushed, pause for review before starting S-3.

If you want a different starting point, replace S-1 with whichever
feature ID makes sense — the table in §3 has the recommended order
but each feature is independently buildable once the prerequisites
listed in its own section are met.
