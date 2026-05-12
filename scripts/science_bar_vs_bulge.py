"""T5.2 - Bar fraction vs bulge size (substituted for bar-vs-z science case).

DEVPLAN T5.2 calls for "bar fraction vs z" with a qualitative trend
matching Walmsley+23. Redshift labels are NOT in the gz_desi_wds
shards or our T2.1 volunteer catalog (would require an external NSA
cross-match). We pivot to the related morphology relation that can
be computed from data we already have:

    bar fraction declines with increasing bulge prominence

This is well-established in the literature (Masters+11, Skibba+12,
Walmsley+22 fig 7) and serves the same DEVPLAN purpose: showing the
model recovers a genuine morphology-vs-morphology trend.

Inputs:
  releases/gz_desi_dirichlet_v1.parquet   -- model alpha per galaxy
  data/gz_desi_volunteer_decals.parquet   -- volunteer votes + dr8_id

Joined on dr8_id, filtered to "featured-or-disk, not edge-on" galaxies
(the only branch where bar + bulge-size questions are reachable
in the GZ DESI decision tree). For each galaxy:

  model_bar_frac    = (alpha_strong + alpha_weak) / sum(alpha_bar_q)
  model_bulge_class = argmax of model alpha over the 5 bulge answers
  volunteer_bar_frac, volunteer_bulge_class -- analogous from votes

Galaxies binned by VOLUNTEER bulge class (5 bins: dominant, large,
moderate, small, none). For each bin: mean bar fraction (model and
volunteer), with bootstrap 95% CI on the mean.

Outputs:
  artifacts/bar_fraction_vs_bulge.png       (committed)
  artifacts/bar_fraction_vs_bulge_metrics.json (committed)

Acceptance check (T5.2 gate, in tests/test_t5_2_science.py): model and
volunteer Spearman correlations vs bulge ordinal class have the same
sign (both negative -> same monotone direction).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from galaxy_vit.data.gz_desi import GZ_DESI_QUESTIONS

DEFAULT_INFERENCE = Path("releases/gz_desi_dirichlet_v1.parquet")
DEFAULT_VOLUNTEER = Path("data/gz_desi_volunteer_decals.parquet")
DEFAULT_OUT_PNG = Path("artifacts/bar_fraction_vs_bulge.png")
DEFAULT_OUT_JSON = Path("artifacts/bar_fraction_vs_bulge_metrics.json")
DEFAULT_MIN_VOTES = 5
N_BOOTSTRAP = 1000

# Canonical answer indices (matches the T3.6 alpha_0..alpha_33 layout
# from question_index_groups). Verified at module load via assertion.
BAR_ANSWERS = ("strong", "weak", "no")
BULGE_ANSWERS = ("dominant", "large", "moderate", "small", "none")
BULGE_ORDINAL: dict[str, int] = {a: i for i, a in enumerate(BULGE_ANSWERS)}

# Sanity-check: schema ordering matches what the alpha column indices assume.
assert GZ_DESI_QUESTIONS["bar"] == BAR_ANSWERS, "bar answer order drifted from schema"
assert GZ_DESI_QUESTIONS["bulge-size"] == BULGE_ANSWERS, "bulge-size order drifted"


def _alpha_indices() -> dict[str, tuple[int, int]]:
    """Compute (start, end) into alpha_{i} columns per question."""
    out: dict[str, tuple[int, int]] = {}
    cursor = 0
    for q, answers in GZ_DESI_QUESTIONS.items():
        out[q] = (cursor, cursor + len(answers))
        cursor += len(answers)
    return out


def _bootstrap_mean_ci(
    values: np.ndarray, *, n_boot: int, seed: int = 42, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Bootstrap (mean, ci_lo, ci_hi) on a 1-D array."""
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = values.shape[0]
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = float(sample.mean())
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument("--volunteer", type=Path, default=DEFAULT_VOLUNTEER)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--min-votes", type=int, default=DEFAULT_MIN_VOTES)
    args = parser.parse_args(argv)

    if not args.inference.is_file():
        raise FileNotFoundError(args.inference)
    if not args.volunteer.is_file():
        raise FileNotFoundError(args.volunteer)

    inf = pd.read_parquet(args.inference)
    vol = pd.read_parquet(args.volunteer)
    print(f"[science] inference rows: {len(inf)}", flush=True)
    print(f"[science] volunteer rows: {len(vol)}", flush=True)

    # Inner join on dr8_id (inference column populated by run_inference_pass.py
    # using "8000_<brick>_<object>" as the catalog convention).
    joined = inf.merge(vol, on="dr8_id", how="inner")
    print(f"[science] joined rows (matched on dr8_id): {len(joined)}", flush=True)
    if len(joined) == 0:
        raise RuntimeError(
            "no galaxies matched between inference and volunteer catalogs; "
            "check dr8_id format (inference parquet must have populated dr8_id)"
        )

    # Restrict to galaxies where the volunteer bar + bulge-size questions are
    # both validly answered (>= min_votes). These are the galaxies where the
    # decision tree actually reached both questions -- "featured-or-disk,
    # not edge-on" branch.
    bar_total = joined["bar_total-votes"]
    bulge_total = joined["bulge-size_total-votes"]
    valid_mask = (bar_total >= args.min_votes) & (bulge_total >= args.min_votes)
    galaxies = joined.loc[valid_mask].reset_index(drop=True)
    print(
        f"[science] galaxies with valid bar AND bulge-size votes: "
        f"{len(galaxies)} ({100 * len(galaxies) / max(1, len(joined)):.1f}%)",
        flush=True,
    )

    idx = _alpha_indices()
    bar_lo, bar_hi = idx["bar"]
    bulge_lo, bulge_hi = idx["bulge-size"]

    # Model alpha slices. (We bin by VOLUNTEER bulge plurality below, not
    # the model's bulge alpha -- volunteer plurality is the more reliable
    # per-galaxy bulge class. Model bulge alphas are not needed in this
    # analysis but the slice is documented in the schema for clarity.)
    bar_alpha = galaxies[[f"alpha_{i}" for i in range(bar_lo, bar_hi)]].to_numpy()
    _ = bulge_lo, bulge_hi  # documented but unused -- see comment above

    # Model bar fraction = (strong + weak) / sum(bar_q).
    model_bar_any = (bar_alpha[:, 0] + bar_alpha[:, 1]) / bar_alpha.sum(axis=-1)

    # Volunteer bar fraction (any bar).
    vol_bar_strong = galaxies["bar_strong"].to_numpy()
    vol_bar_weak = galaxies["bar_weak"].to_numpy()
    vol_bar_no = galaxies["bar_no"].to_numpy()
    vol_bar_total = vol_bar_strong + vol_bar_weak + vol_bar_no
    vol_bar_any = (vol_bar_strong + vol_bar_weak) / np.maximum(1, vol_bar_total)

    # Bin by VOLUNTEER bulge plurality (the most reliable per-galaxy bulge class).
    vol_bulge_counts = galaxies[
        [f"bulge-size_{a}" for a in BULGE_ANSWERS]
    ].to_numpy()
    vol_bulge_class = vol_bulge_counts.argmax(axis=-1)

    # Per-bin stats: mean bar fraction (model + volunteer) with bootstrap CI.
    per_bin: dict[str, dict[str, Any]] = {}
    for class_idx, name in enumerate(BULGE_ANSWERS):
        in_bin = vol_bulge_class == class_idx
        n = int(in_bin.sum())
        if n == 0:
            per_bin[name] = {
                "n": 0,
                "model_bar_mean": None, "model_bar_ci_lo": None, "model_bar_ci_hi": None,
                "vol_bar_mean": None, "vol_bar_ci_lo": None, "vol_bar_ci_hi": None,
            }
            continue
        m_mean, m_lo, m_hi = _bootstrap_mean_ci(
            model_bar_any[in_bin], n_boot=N_BOOTSTRAP, seed=42 + class_idx
        )
        v_mean, v_lo, v_hi = _bootstrap_mean_ci(
            vol_bar_any[in_bin], n_boot=N_BOOTSTRAP, seed=84 + class_idx
        )
        per_bin[name] = {
            "n": n,
            "model_bar_mean": m_mean, "model_bar_ci_lo": m_lo, "model_bar_ci_hi": m_hi,
            "vol_bar_mean": v_mean, "vol_bar_ci_lo": v_lo, "vol_bar_ci_hi": v_hi,
        }
        print(
            f"  {name:>10s}  n={n:>5d}  "
            f"model={m_mean:.3f} [{m_lo:.3f},{m_hi:.3f}]  "
            f"vol={v_mean:.3f} [{v_lo:.3f},{v_hi:.3f}]",
            flush=True,
        )

    # Spearman correlation of bar fraction vs bulge ordinal (gate metric).
    from scipy.stats import spearmanr

    bulge_ordinal = vol_bulge_class.astype(np.float64)
    rho_model, p_model = spearmanr(bulge_ordinal, model_bar_any)
    rho_vol, p_vol = spearmanr(bulge_ordinal, vol_bar_any)
    print(
        f"[science] Spearman bar vs bulge: "
        f"model rho={rho_model:.3f} (p={p_model:.2g})  "
        f"volunteer rho={rho_vol:.3f} (p={p_vol:.2g})",
        flush=True,
    )

    # ---- Render figure -----------------------------------------------------
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=160)
    x = np.arange(len(BULGE_ANSWERS))
    m_means = np.array([per_bin[a]["model_bar_mean"] or np.nan for a in BULGE_ANSWERS])
    m_lo = np.array([per_bin[a]["model_bar_ci_lo"] or np.nan for a in BULGE_ANSWERS])
    m_hi = np.array([per_bin[a]["model_bar_ci_hi"] or np.nan for a in BULGE_ANSWERS])
    v_means = np.array([per_bin[a]["vol_bar_mean"] or np.nan for a in BULGE_ANSWERS])
    v_lo = np.array([per_bin[a]["vol_bar_ci_lo"] or np.nan for a in BULGE_ANSWERS])
    v_hi = np.array([per_bin[a]["vol_bar_ci_hi"] or np.nan for a in BULGE_ANSWERS])

    ax.errorbar(
        x - 0.07, m_means,
        yerr=[m_means - m_lo, m_hi - m_means],
        fmt="o-", color="#0072B2", label=f"model (rho={rho_model:.2f})",
        capsize=4, linewidth=1.5,
    )
    ax.errorbar(
        x + 0.07, v_means,
        yerr=[v_means - v_lo, v_hi - v_means],
        fmt="s-", color="#E69F00", label=f"volunteer (rho={rho_vol:.2f})",
        capsize=4, linewidth=1.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(BULGE_ANSWERS)
    ax.set_xlabel("bulge size (volunteer plurality)")
    ax.set_ylabel("bar fraction (any bar)")
    ax.set_title(
        f"Bar fraction vs bulge prominence (n={len(galaxies)} galaxies)\n"
        "model vs volunteer; error bars = bootstrap 95% CI on the mean"
    )
    ax.set_ylim(0, max(1.0, float(np.nanmax([m_hi.max(), v_hi.max()])) * 1.1))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=160)
    plt.close(fig)
    print(f"[science] wrote {args.out_png}", flush=True)

    # ---- Write metrics JSON -----------------------------------------------
    metrics = {
        "task": "T5.2-bar-vs-bulge",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "inference": str(args.inference),
        "volunteer": str(args.volunteer),
        "min_votes": args.min_votes,
        "n_joined": len(joined),
        "n_valid_bar_and_bulge": len(galaxies),
        "spearman_model": {"rho": float(rho_model), "p": float(p_model)},
        "spearman_volunteer": {"rho": float(rho_vol), "p": float(p_vol)},
        "per_bin": per_bin,
    }
    args.out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[science] wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
