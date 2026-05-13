"""C-15 — Post-process per-epoch demo features into a UMAP movie parquet.

Reads ``artifacts/per_epoch_demo_features.parquet`` (produced by the
T3.6 trainer's per-epoch hook), fits a UMAP on the FINAL epoch's
features (per the plan note: "fit UMAP once on the FINAL epoch, then
project each earlier epoch's features through the same UMAP" -- this
avoids the per-epoch cloud jitter that a re-fit per epoch would
introduce), and writes
``artifacts/training_movie.parquet`` with per-(epoch, galaxy) UMAP
coordinates ready for the frontend.

Output schema::

    epoch        int       (-1 = pretrained snapshot; 0..N-1 = training epochs)
    galaxy_id    str       demo-galaxy identifier
    umap_x       float32   UMAP coord 1
    umap_y       float32   UMAP coord 2
    label_name   str       smooth | featured-or-disk | artifact (from manifest plurality)

Invocation::

    python -m scripts.build_training_movie
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_FEATURES = Path("artifacts/per_epoch_demo_features.parquet")
DEFAULT_MANIFEST = Path("artifacts/demo_galaxies/manifest.json")
DEFAULT_OUT = Path("artifacts/training_movie.parquet")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-neighbors", type=int, default=10)
    parser.add_argument("--min-dist", type=float, default=0.3)
    args = parser.parse_args(argv)

    if not args.features.is_file():
        print(
            f"[movie] missing per-epoch features parquet: {args.features}; "
            "rerun the trainer with logging.per_epoch_features_path set",
            file=sys.stderr,
        )
        return 2
    if not args.manifest.is_file():
        print(
            f"[movie] missing demo manifest: {args.manifest}", file=sys.stderr
        )
        return 2

    import numpy as np
    import pandas as pd
    import umap

    df = pd.read_parquet(args.features)
    if df.empty:
        print("[movie] feature parquet is empty", file=sys.stderr)
        return 2
    print(
        f"[movie] {len(df)} rows over {df['epoch'].nunique()} epochs, "
        f"{df['galaxy_id'].nunique()} galaxies",
        flush=True,
    )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    label_lookup = {
        str(g["id"]): str(g["smooth_or_featured_plurality"]) for g in manifest
    }

    # Final-epoch feature matrix (largest epoch number wins).
    last_epoch = int(df["epoch"].max())
    final_df = df[df["epoch"] == last_epoch]
    final_features = np.array(final_df["features"].tolist(), dtype=np.float32)
    print(
        f"[movie] fitting UMAP on epoch {last_epoch} features "
        f"({final_features.shape})",
        flush=True,
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(args.n_neighbors, max(2, final_features.shape[0] - 1)),
        min_dist=args.min_dist,
        random_state=args.seed,
        metric="cosine",
    )
    reducer.fit(final_features)

    # Project EVERY epoch through the same UMAP fit so the movie's
    # cloud doesn't jitter unrelated to training progress.
    all_coords: list[dict[str, object]] = []
    for epoch_value in sorted(df["epoch"].unique()):
        sub = df[df["epoch"] == epoch_value]
        feats = np.array(sub["features"].tolist(), dtype=np.float32)
        coords = reducer.transform(feats)
        for (_, row), (xy_x, xy_y) in zip(
            sub.iterrows(), coords, strict=True
        ):
            gid = str(row["galaxy_id"])
            all_coords.append(
                {
                    "epoch": int(epoch_value),
                    "galaxy_id": gid,
                    "umap_x": float(xy_x),
                    "umap_y": float(xy_y),
                    "label_name": label_lookup.get(gid, "unknown"),
                }
            )

    out_df = pd.DataFrame(all_coords)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(
        f"[movie] wrote {args.out} "
        f"({len(out_df)} rows; epochs={sorted({int(e) for e in df['epoch'].unique()})})",
        flush=True,
    )

    # Quick sanity: every epoch should contribute the same n_galaxies.
    counts_by_epoch = out_df.groupby("epoch").size()
    if counts_by_epoch.nunique() != 1:
        print(
            f"[movie] WARN: per-epoch row counts vary: {dict(counts_by_epoch)}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
