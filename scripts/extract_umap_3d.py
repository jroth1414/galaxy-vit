"""A-6 — Fit a 3-D UMAP over the same 2,462 UMAP-set features.

This script piggybacks on two existing artifacts to avoid re-running
the ConvNeXt encoder:

* ``artifacts/test_thumb_features.parquet`` (from S-1) — the (N, D)
  feature matrix in test-thumb idx order.
* ``artifacts/umap_coords.parquet``         (from T2.4) — per-row
  ``smooth-or-featured`` label (same idx order) used to colour the
  scatter; copied verbatim into the 3-D output so the frontend can
  share the legend / palette with the 2-D view.

UMAP 3-D is a re-fit, not a 2-D + z axis projection — so the new
coords have NO relation to the 2-D umap_x / umap_y values. Row idx
alignment is the only invariant.

Output: ``artifacts/umap_3d_coords.parquet`` with columns
``umap_x, umap_y, umap_z, smooth_or_featured_label,
smooth_or_featured_name``. Same idx order as
``artifacts/test_thumbs/<idx>.jpg`` and
``artifacts/test_thumb_features.parquet``.

Invocation::

    python -m scripts.extract_umap_3d
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_FEATURES = Path("artifacts/test_thumb_features.parquet")
DEFAULT_LABELS = Path("artifacts/umap_coords.parquet")
DEFAULT_OUT = Path("artifacts/umap_3d_coords.parquet")
DEFAULT_METRICS = Path("artifacts/umap_3d_metrics.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--labels-from", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--umap-neighbors", type=int, default=15, help="UMAP n_neighbors."
    )
    parser.add_argument(
        "--umap-min-dist", type=float, default=0.1, help="UMAP min_dist."
    )
    parser.add_argument(
        "--umap-metric", type=str, default="cosine", help="UMAP distance metric."
    )
    args = parser.parse_args(argv)

    if not args.features.is_file():
        print(
            f"[umap3d] missing feature cache: {args.features}; "
            "run `python -m scripts.cache_test_thumb_features` first",
            file=sys.stderr,
        )
        return 2
    if not args.labels_from.is_file():
        print(
            f"[umap3d] missing 2-D labels parquet: {args.labels_from}; "
            "run `python -m scripts.extract_umap` first",
            file=sys.stderr,
        )
        return 2

    import numpy as np
    import pandas as pd
    import umap

    print(f"[umap3d] loading features from {args.features}", flush=True)
    feat_df = pd.read_parquet(args.features)
    features = np.array(feat_df["features"].tolist(), dtype=np.float32)
    print(f"[umap3d] features shape={features.shape}", flush=True)

    print(f"[umap3d] loading labels from {args.labels_from}", flush=True)
    labels_df = pd.read_parquet(args.labels_from)
    if len(labels_df) != len(feat_df):
        print(
            f"[umap3d] row-count mismatch: features={len(feat_df)} "
            f"labels={len(labels_df)} -- iterations have drifted",
            file=sys.stderr,
        )
        return 2

    print(
        f"[umap3d] fitting UMAP 3-D "
        f"(n_neighbors={args.umap_neighbors}, "
        f"min_dist={args.umap_min_dist}, metric={args.umap_metric})",
        flush=True,
    )
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
    )
    coords = reducer.fit_transform(features)
    print(f"[umap3d] coords shape={coords.shape}", flush=True)

    out_df = pd.DataFrame(
        {
            "umap_x": coords[:, 0].astype("float32"),
            "umap_y": coords[:, 1].astype("float32"),
            "umap_z": coords[:, 2].astype("float32"),
            "smooth_or_featured_label": labels_df["smooth_or_featured_label"],
            "smooth_or_featured_name": labels_df["smooth_or_featured_name"],
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"[umap3d] wrote {args.out}", flush=True)

    metrics = {
        "task": "A-6-umap-3d",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "source_features": str(args.features),
        "source_labels": str(args.labels_from),
        "n_samples": int(features.shape[0]),
        "n_features_dim": int(features.shape[1]),
        "umap_params": {
            "n_components": 3,
            "n_neighbors": args.umap_neighbors,
            "min_dist": args.umap_min_dist,
            "metric": args.umap_metric,
            "random_state": args.seed,
        },
        "label_counts": {
            str(k): int(v)
            for k, v in labels_df["smooth_or_featured_name"]
            .value_counts()
            .to_dict()
            .items()
        },
    }
    args.metrics_out.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[umap3d] wrote {args.metrics_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
