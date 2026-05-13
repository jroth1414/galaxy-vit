"""S-2 — Precompute the joined catalog for the Sky tab.

Joins ``releases/gz_desi_dirichlet_v1.parquet`` (the full-pass
inference output; one row per galaxy with alpha_0..33) against
``data/gz_desi_volunteer_decals.parquet`` (the volunteer catalog;
carries ra/dec/dr8_id alongside the raw vote counts) on ``dr8_id``,
then attaches a small amount of summary information per galaxy:

* ``ra``, ``dec``                — sky coordinates (decimal degrees)
* ``smooth_or_featured_label``  — argmax of the model's posterior
  mean over the 3 smooth-or-featured answers (0=smooth,
  1=featured-or-disk, 2=artifact). Same code/name palette as
  ``artifacts/umap_coords.parquet`` so the legend can be shared.
* ``smooth_or_featured_name``   — human-readable name.
* ``entropy``                   — predictive entropy summed across
  all 10 GZ DESI questions (Dirichlet posterior mean); usable as a
  "color by uncertainty" channel in the Sky tab.

The output is committed as ``artifacts/sky_points.parquet`` (~500 KB
for the ~14 k joined subset).

Invocation::

    python -m scripts.build_sky_points
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

DEFAULT_INFERENCE = Path("releases/gz_desi_dirichlet_v1.parquet")
DEFAULT_VOLUNTEER = Path("data/gz_desi_volunteer_decals.parquet")
DEFAULT_OUT = Path("artifacts/sky_points.parquet")

# Order matches galaxy_vit.data.schema.question_index_groups()['smooth-or-featured']:
# smooth, featured-or-disk, artifact (3 classes).
SOF_NAMES = ("smooth", "featured-or-disk", "artifact")


def _alpha_to_question_groups() -> list[tuple[str, int, int]]:
    """Lazy import the canonical slice layout so this script stays optional-dep-friendly."""
    from galaxy_vit.data.schema import question_index_groups

    return question_index_groups()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", type=Path, default=DEFAULT_INFERENCE)
    parser.add_argument("--volunteer", type=Path, default=DEFAULT_VOLUNTEER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.inference.is_file():
        print(f"[sky] missing inference parquet: {args.inference}", file=sys.stderr)
        return 2
    if not args.volunteer.is_file():
        print(f"[sky] missing volunteer parquet: {args.volunteer}", file=sys.stderr)
        return 2

    import numpy as np
    import pandas as pd

    print(f"[sky] loading inference parquet: {args.inference}", flush=True)
    inf_df = pd.read_parquet(args.inference)
    print(f"[sky]   {len(inf_df)} inference rows", flush=True)

    print(f"[sky] loading volunteer parquet: {args.volunteer}", flush=True)
    vol_df = pd.read_parquet(
        args.volunteer, columns=["dr8_id", "ra", "dec"]
    )
    print(f"[sky]   {len(vol_df)} volunteer rows", flush=True)

    print("[sky] joining on dr8_id", flush=True)
    joined = inf_df.merge(vol_df, on="dr8_id", how="inner")
    print(f"[sky]   {len(joined)} joined rows", flush=True)
    if len(joined) == 0:
        print("[sky] empty join; dr8_id formats mismatch?", file=sys.stderr)
        return 2

    # Compute summary signals from alpha_0..33 directly on the joined frame.
    alpha_cols = [f"alpha_{i}" for i in range(34)]
    alpha = joined[alpha_cols].to_numpy(dtype=np.float64)  # (N, 34)

    # Argmax over smooth-or-featured (alpha_0..alpha_2). Argmax of alpha
    # is the same as argmax of the posterior mean (alpha / sum(alpha))
    # within a question slice, so no need to normalise.
    sof = alpha[:, 0:3]
    sof_label = sof.argmax(axis=1).astype(np.int8)

    # Per-question predictive entropy on the posterior mean, summed
    # across all 10 questions. Mirrors
    # galaxy_vit.training.active_learning.predictive_entropy.
    groups = _alpha_to_question_groups()
    eps = 1e-12
    entropy = np.zeros(len(joined), dtype=np.float32)
    for _q, start, end in groups:
        slice_alpha = alpha[:, start:end]
        denom = slice_alpha.sum(axis=1, keepdims=True).clip(min=eps)
        p = (slice_alpha / denom).clip(min=eps)
        entropy += (-(p * np.log(p)).sum(axis=1)).astype(np.float32)

    out_df = pd.DataFrame(
        {
            "dr8_id": joined["dr8_id"].astype(str).values,
            "ra": joined["ra"].astype("float32").values,
            "dec": joined["dec"].astype("float32").values,
            "smooth_or_featured_label": sof_label,
            "smooth_or_featured_name": [SOF_NAMES[i] for i in sof_label],
            "entropy": entropy,
        }
    )

    # Sanity-check ra / dec ranges before writing.
    if not (
        (out_df["ra"].min() >= 0.0)
        and (out_df["ra"].max() <= 360.0)
        and (out_df["dec"].min() >= -90.0)
        and (out_df["dec"].max() <= 90.0)
    ):
        print(
            f"[sky] WARN: ra/dec out of expected ranges "
            f"(ra=[{out_df['ra'].min()}, {out_df['ra'].max()}], "
            f"dec=[{out_df['dec'].min()}, {out_df['dec'].max()}])",
            file=sys.stderr,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(
        f"[sky] wrote {args.out} "
        f"({len(out_df)} rows, "
        f"entropy median={float(np.median(entropy)):.3f}, "
        f"smooth/featured/artifact counts={dict(out_df['smooth_or_featured_name'].value_counts())})",
        flush=True,
    )
    # Quick parity check: entropy should be finite everywhere.
    nan_n = int(out_df["entropy"].isna().sum())
    if nan_n:
        print(f"[sky] FATAL: {nan_n} NaN entropy values", file=sys.stderr)
        return 2
    # Sanity: total entropy bounded by sum of log(K_q) over questions.
    max_entropy = sum(math.log(end - start) for _q, start, end in groups)
    if float(out_df["entropy"].max()) > max_entropy + 1e-3:
        print(
            f"[sky] FATAL: entropy {float(out_df['entropy'].max())} "
            f"exceeds theoretical max {max_entropy}",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
