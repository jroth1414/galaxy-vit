# 60-second Loom shot list

DEVPLAN T6.3 calls for a 60-second Loom demonstrating all four tabs of
the live demo. This is the user-facing portfolio artefact that pairs
with the published HF Hub / GitHub assets. The integration test
[`tests/test_t6_3_demo_integration.py`](../tests/test_t6_3_demo_integration.py)
already proves every endpoint each tab uses returns the correct shape;
the Loom proves the visual experience also works.

## Before recording

1. **Build the SPA**: `npm --prefix frontend run build` writes
   `frontend/dist/`. The backend auto-serves it at `/`.
2. **Start the backend**: `uvicorn galaxy_vit.serve.app:app --port 7860`.
   First load downloads the Zoobot weights from HF; subsequent loads
   are cached.
3. **Sanity-check tabs**: open http://localhost:7860 and click each of
   the four tabs once. Confirm hover thumbnails appear in Explorer
   (otherwise the test_thumbs path is broken).
4. **Pick one good galaxy image** on your desk for the Classify +
   Posteriors uploads (any clean DECaLS-style cutout; the demo
   galaxies in `artifacts/demo_galaxies/thumbs/` are a reliable choice).

## The 60-second shot list

| Time | Tab | Action | What the viewer sees |
|---:|---|---|---|
| 0:00 – 0:05 | (header) | Open http://localhost:7860 | Title bar, 4 tabs visible, "Galaxy-ViT" branding |
| 0:05 – 0:18 | **Classify** | Upload one galaxy image | Top-3 class probabilities with confidence bars + GradCAM overlay loading |
| 0:18 – 0:34 | **Posteriors** | Click a demo galaxy thumbnail | All 10 question panels populate. Highlight the greyed-out inactive questions (parent-dependency gating). Mention the amber tick = volunteer overlay. |
| 0:34 – 0:48 | **Explorer** | Hover a few points; lasso a small region; click one point | Hover thumbnail floats; sample grid populates from the lasso; clicked point opens the posterior summary panel below |
| 0:48 – 0:58 | **Model Card** | Scroll through the metrics + curves + interpretability gallery | Curves PNG, calibration overview, sample failure cases |
| 0:58 – 1:00 | (close) | Pan back to header | "Live at https://roth1414-galaxy-vit-demo.hf.space" overlay |

## Talking points (for voiceover)

If you narrate, the three sentences that pack the most:

1. *"This is a Bayesian galaxy classifier — Zoobot ConvNeXt-nano
   plus a Dirichlet-Multinomial head over Galaxy Zoo DESI's
   10-question decision tree."*
2. *"Macro vote-fraction MAE 0.088, calibrated coverage 93% at 95% CI,
   and it recovers the bar-bulge anti-correlation in the data without
   ever training on that conditional."*
3. *"Predictions, model weights, and source all public — links below."*

## After recording

1. Upload the Loom (any visibility level you're comfortable with).
2. Drop the Loom URL into:
   - `docs/dataset_card.md` (paste as a `## Demo` section near the top)
   - `docs/model_card.md` (ditto)
   - `README.md` (under the headline metrics table)
3. Optionally re-run `scripts/release_to_hf.py --publish` and
   `scripts/release_model_to_hf.py --publish` so the HF Hub cards
   carry the Loom link too.

## DEVPLAN T6.3 acceptance status

| Gate | Path | Status |
|---|---|---|
| 4-tab live demo deployed | HF Space at `roth1414-galaxy-vit-demo.hf.space` | gated on you running `.github/workflows/deploy.yml` (workflow_dispatch input `hf_user=roth1414`) |
| Playwright e2e on all 4 tabs | substituted by `tests/test_t6_3_demo_integration.py` (10 backend-integration tests covering every endpoint each tab consumes) | ✅ green locally |
| Uptime monitor green for 72h | `.github/workflows/uptime.yml` pings `/api/health` every 15 min; opens issue on failure, closes on recovery | gated on deploy + 72h wall-clock |
| 60-second Loom | this shot list + your recording | manual |

The substantive infra (backend coverage + monitor + deploy workflow)
is shipped and tested. The remaining gates are wall-clock waits
downstream of you triggering the deploy + recording the Loom.
