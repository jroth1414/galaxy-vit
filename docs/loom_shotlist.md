# 90-second Loom shot list (v2)

DEVPLAN T6.3 originally called for a 60-second Loom over the v1 4-tab
demo. The v2 release adds three new tabs (Sky, Similar, Training) and
several in-tab enhancements, so the shot list is extended to ~90
seconds. The integration test
[`tests/test_t6_3_demo_integration.py`](../tests/test_t6_3_demo_integration.py)
still covers every endpoint each tab uses.

## Before recording

1. **Build the SPA**: `npm --prefix frontend run build` writes
   `frontend/dist/`. The backend auto-serves it at `/`.
2. **Start the backend**: `uvicorn galaxy_vit.serve.app:app --port 7860`.
   First load downloads the Zoobot weights from HF; subsequent loads
   are cached.
3. **Sanity-check tabs**: open http://localhost:7860 and click each of
   the seven tabs once. Confirm in particular:
   - Explorer hover thumbnails crossfade to the GradCAM overlay (~600 ms).
   - Sky scatter shows the ~14k point cloud and the name input resolves
     "M31" to RA ≈ 10.68, Dec ≈ 41.27.
   - Similar finds idx 0 first with distance 0 when you query by idx.
   - Training plays cleanly through every epoch on the slider.
4. **Pick one good galaxy image** on your desk for the Classify +
   Posteriors uploads (any clean DECaLS-style cutout; the demo
   galaxies in `artifacts/demo_galaxies/thumbs/` are a reliable choice).

## The 90-second shot list

| Time | Tab | Action | What the viewer sees |
|---:|---|---|---|
| 0:00 – 0:05 | (header) | Open http://localhost:7860 | Title bar, 7 tabs visible |
| 0:05 – 0:18 | **Classify** | Upload one galaxy image · click "Classify" · then "Compare with M3" | Top-3 + GradCAM, then M3 per-question bars appear inline (Compare side-panel) |
| 0:18 – 0:32 | **Posteriors** | Click a demo galaxy · toggle Bar/Tree · pick a question in the GradCAM dropdown | All 10 question panels, then Sankey reach-flow, then a per-question GradCAM swap |
| 0:32 – 0:42 | (Posteriors) | Scroll up to the outlier panel · click a high-entropy thumbnail | Three columns (entropy / BALD / disagreement), click loads that test galaxy's posterior |
| 0:42 – 0:55 | **Explorer** | Hover (catch the saliency crossfade) · 2-D ↔ 3-D toggle · click a point · "Find similar →" | UMAP cloud, GradCAM crossfade, 3-D rotation, deep-link into Similar |
| 0:55 – 1:08 | **Similar** | The preset query auto-loads · scroll the result grid · click one thumbnail | 4×5 sample grid with cosine distances, distance ≈ 0 for the query itself |
| 1:08 – 1:20 | **Sky** | Type "NGC 1300" → Resolve & centre · click "Predict at coords" | Aladin iframe centres; M1 top-3 + GradCAM at the bottom |
| 1:20 – 1:30 | **Training** | Press Play | 24 demo galaxies migrate across UMAP space as epochs advance |
| 1:30 – 1:32 | (close) | Pan back to header | "Live at https://roth1414-galaxy-vit-demo.hf.space" overlay |

If you need to trim to 60 s, the safest cuts are the Compare side-panel
(0:13-0:18), the outlier-click flow (0:32-0:42), and the Sky name
resolver (1:08-1:20). The other beats each demonstrate a unique
capability.

## Talking points (for voiceover)

Three sentences that pack the most:

1. *"Bayesian galaxy classifier — Zoobot ConvNeXt-nano plus a
   Dirichlet-Multinomial head over Galaxy Zoo DESI's 10-question
   decision tree."*
2. *"Macro vote-fraction MAE 0.088, calibrated coverage 93% at 95% CI,
   and it recovers the bar-bulge anti-correlation without ever training
   on that conditional."*
3. *"Seven interactive views in the demo — predictions, posteriors,
   feature-space exploration, sky map with Aladin, kNN search,
   training-progress movie. Predictions + model weights + source all
   public; links below."*

## After recording

1. Upload the Loom (any visibility level you're comfortable with).
2. Drop the Loom URL into:
   - `docs/dataset_card.md` (paste as a `## Demo` section near the top)
   - `docs/model_card.md` (ditto)
   - `README.md` (under the headline metrics table)
3. Optionally re-run `scripts/release_to_hf.py --publish` and
   `scripts/release_model_to_hf.py --publish` so the HF Hub cards
   carry the Loom link too.

## DEVPLAN T6.3 acceptance status (unchanged from v1)

| Gate | Path | Status |
|---|---|---|
| Live demo deployed (now 7 tabs) | HF Space at `roth1414-galaxy-vit-demo.hf.space` | gated on you running `.github/workflows/deploy.yml` |
| Playwright e2e on all tabs | `tests/test_t6_3_demo_integration.py` (backend-integration coverage) + the new `test_v2_*.py` files | ✅ green locally |
| Uptime monitor green for 72h | `.github/workflows/uptime.yml` pings `/api/health` every 15 min | gated on deploy + 72h wall-clock |
| 60+ s Loom | this shot list + your recording | manual |
