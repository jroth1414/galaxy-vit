"""T2.4 — Penultimate-feature UMAP visualization + silhouette gate.

Loads the W+23 reproduction checkpoint (default
``runs/m2_w23_reproduction/best.pt``), iterates the GZ DESI DR8 test
shards used by T2.3, extracts the ConvNeXt-nano encoder's penultimate
features (``model.encoder(x)`` returns ``(B, num_features)`` because the
encoder is built with ``num_classes=0``), fits UMAP to 2-D, and writes
a static figure colored by the ``smooth-or-featured`` plurality answer.

Acceptance gate (T2.4): silhouette score across the 3 ``smooth-or-featured``
classes must be at least ``--silhouette-min`` (default 0.15). The score
is computed on a stratified subsample to keep silhouette's O(n^2)
distance computation feasible.

Outputs (under ``--out-dir``, default ``artifacts/``):

  * ``umap_penultimate.png``  — committed, scatter colored by class
  * ``umap_metrics.json``     — committed, silhouette + run metadata
  * ``umap_coords.parquet``   — NOT committed (regenerable, large)

Invocation::

    python -m scripts.extract_umap \\
        --config configs/m2_w23_reproduction.yaml \\
        --checkpoint runs/m2_w23_reproduction/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from galaxy_vit.config import Settings
from galaxy_vit.data.gz_desi_hf_dataset import (
    build_gz_desi_hf_dataset,
    collate_multi_question,
)
from galaxy_vit.data.gz_desi_labels import QUESTION_NAMES
from galaxy_vit.data.transforms import build_eval_transform, load_normalization
from galaxy_vit.models.zoobot_encoder import build_zoobot_finetune
from galaxy_vit.training.multi_question_trainer import MQConfig, _resolve_shards

DEFAULT_CONFIG = Path("configs/m2_w23_reproduction.yaml")
DEFAULT_CHECKPOINT = Path("runs/m2_w23_reproduction/best.pt")
DEFAULT_OUT_DIR = Path("artifacts")
DEFAULT_SILHOUETTE_MIN = 0.15
DEFAULT_SILHOUETTE_SUBSAMPLE = 5000
DEFAULT_MAX_BATCHES: int | None = None  # None => iterate all test shards

SMOOTH_OR_FEATURED_IDX = QUESTION_NAMES.index("smooth-or-featured")
SMOOTH_OR_FEATURED_NAMES = ("smooth", "featured-or-disk", "artifact")
# Hand-picked colorblind-friendly trio (Wong palette).
CLASS_COLORS = ("#0072B2", "#E69F00", "#CC79A7")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def _extract_features(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    max_batches: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Iterate the loader, return (features, smooth_or_featured_labels, valid_mask).

    ``model.encoder`` returns the (B, num_features) penultimate ConvNeXt-nano
    embedding directly; we never invoke the classifier head here.
    """
    model.eval()
    feats: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []

    encoder = model.encoder  # type: ignore[attr-defined]
    for batch_idx, (x, plurality, valid) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        f = encoder(x).float().cpu()
        feats.append(f)
        labels.append(plurality[:, SMOOTH_OR_FEATURED_IDX].cpu())
        valids.append(valid[:, SMOOTH_OR_FEATURED_IDX].cpu())
        if (batch_idx + 1) % 25 == 0:
            print(f"[umap] processed {batch_idx + 1} batches", flush=True)

    feats_t = torch.cat(feats, dim=0) if feats else torch.empty(0)
    labels_t = torch.cat(labels, dim=0) if labels else torch.empty(0, dtype=torch.long)
    valids_t = torch.cat(valids, dim=0) if valids else torch.empty(0, dtype=torch.bool)
    return feats_t, labels_t, valids_t


def _stratified_subsample_indices(
    labels: torch.Tensor, *, k_total: int, seed: int
) -> torch.Tensor:
    """Pick at most ``k_total`` indices stratified by class label."""
    g = torch.Generator().manual_seed(seed)
    classes = torch.unique(labels)
    per_class = max(1, k_total // max(1, classes.numel()))
    keep: list[torch.Tensor] = []
    for c in classes.tolist():
        idx = torch.where(labels == c)[0]
        if idx.numel() <= per_class:
            keep.append(idx)
        else:
            perm = torch.randperm(idx.numel(), generator=g)[:per_class]
            keep.append(idx[perm])
    out = torch.cat(keep)
    perm = torch.randperm(out.numel(), generator=g)
    return out[perm]


def _save_scatter(
    coords: Any,
    labels: Any,
    *,
    out_path: Path,
    silhouette: float,
    n_samples: int,
    title_suffix: str = "",
) -> None:
    """Render the 2-D UMAP scatter colored by smooth-or-featured class."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.0, 7.0), dpi=160)
    for class_idx, name in enumerate(SMOOTH_OR_FEATURED_NAMES):
        mask = labels == class_idx
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=4,
            alpha=0.55,
            color=CLASS_COLORS[class_idx],
            label=f"{name} (n={int(mask.sum())})",
            linewidths=0,
        )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    title = (
        "GZ DESI penultimate-feature UMAP "
        f"(silhouette={silhouette:.3f}, n={n_samples})"
    )
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(title)
    ax.legend(loc="best", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--silhouette-min", type=float, default=DEFAULT_SILHOUETTE_MIN)
    parser.add_argument(
        "--silhouette-subsample",
        type=int,
        default=DEFAULT_SILHOUETTE_SUBSAMPLE,
        help="Stratified subsample size for silhouette (O(n^2)); 0 disables subsampling.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=DEFAULT_MAX_BATCHES,
        help="Cap test batches (debugging); default None iterates all DR8 test shards.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--umap-neighbors", type=int, default=15, help="UMAP n_neighbors hyperparameter."
    )
    parser.add_argument(
        "--umap-min-dist", type=float, default=0.1, help="UMAP min_dist hyperparameter."
    )
    parser.add_argument(
        "--umap-metric", type=str, default="cosine", help="UMAP distance metric."
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {args.checkpoint}; "
            "run the T2.3 trainer first (configs/m2_w23_reproduction.yaml)"
        )
    cfg = MQConfig.from_yaml(args.config)
    settings = Settings()  # type: ignore[call-arg]
    os.environ["HF_TOKEN"] = settings.HF_TOKEN.get_secret_value()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[umap] config={args.config} checkpoint={args.checkpoint}", flush=True)
    print(f"[umap] out_dir={args.out_dir}", flush=True)

    _, _, test_paths = _resolve_shards(cfg.data.shards)
    print(f"[umap] test shards: {len(test_paths)}", flush=True)

    mean, std = load_normalization(cfg.data.normalization)
    eval_tf = build_eval_transform(image_size=cfg.data.image_size, mean=mean, std=std)
    test_ds = build_gz_desi_hf_dataset(
        test_paths,
        eval_tf,
        shuffle_buffer=1,
        shardshuffle=False,
        min_votes=cfg.data.min_votes,
    )
    test_loader: DataLoader[Any] = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_multi_question,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[umap] device={device}", flush=True)
    model, _, _ = build_zoobot_finetune(
        num_classes=cfg.model.num_classes, encoder_id=cfg.model.encoder
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    feats_t, labels_t, valid_t = _extract_features(
        model, test_loader, device=device, max_batches=args.max_batches
    )
    if feats_t.numel() == 0:
        raise RuntimeError("no features extracted; check shard paths + min_votes filter")

    # Restrict to galaxies whose smooth-or-featured question has >= min_votes
    # (that's the question we color by, so its label must be reliable).
    keep = valid_t.bool()
    feats = feats_t[keep].numpy()
    labels = labels_t[keep].numpy()
    print(
        f"[umap] features shape={feats.shape}, "
        f"labels per class={[(int(c), int((labels == c).sum())) for c in sorted(set(labels.tolist()))]}",
        flush=True,
    )

    import umap
    from sklearn.metrics import silhouette_score

    print(
        f"[umap] fitting UMAP (n_neighbors={args.umap_neighbors}, "
        f"min_dist={args.umap_min_dist}, metric={args.umap_metric})...",
        flush=True,
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        random_state=args.seed,
    )
    coords = reducer.fit_transform(feats)
    print(f"[umap] coords shape={coords.shape}", flush=True)

    # Silhouette on the original feature space (not UMAP coords) gives a
    # more honest cluster-quality signal; UMAP can artificially separate
    # clusters in 2-D.
    if args.silhouette_subsample and feats.shape[0] > args.silhouette_subsample:
        sub_idx = _stratified_subsample_indices(
            torch.from_numpy(labels),
            k_total=args.silhouette_subsample,
            seed=args.seed,
        ).numpy()
        sil_feats = feats[sub_idx]
        sil_labels = labels[sub_idx]
        print(
            f"[umap] silhouette subsampled to {sil_feats.shape[0]} (stratified)",
            flush=True,
        )
    else:
        sil_feats = feats
        sil_labels = labels

    silhouette = float(
        silhouette_score(sil_feats, sil_labels, metric=args.umap_metric)
    )
    print(f"[umap] silhouette = {silhouette:.4f}", flush=True)

    out_png = args.out_dir / "umap_penultimate.png"
    _save_scatter(
        coords,
        labels,
        out_path=out_png,
        silhouette=silhouette,
        n_samples=int(feats.shape[0]),
    )
    print(f"[umap] wrote {out_png}", flush=True)

    # Parquet (regenerable; not committed). Use pandas because pyarrow's
    # surface for mixed-dtype columns is more verbose.
    try:
        import pandas as pd

        df = pd.DataFrame(
            {
                "umap_x": coords[:, 0].astype("float32"),
                "umap_y": coords[:, 1].astype("float32"),
                "smooth_or_featured_label": labels.astype("int8"),
                "smooth_or_featured_name": [
                    SMOOTH_OR_FEATURED_NAMES[int(c)] for c in labels
                ],
            }
        )
        out_pq = args.out_dir / "umap_coords.parquet"
        df.to_parquet(out_pq, index=False)
        print(f"[umap] wrote {out_pq}", flush=True)
    except ImportError:
        print("[umap] pandas not installed; skipping parquet dump", flush=True)

    metrics = {
        "task": "T2.4-umap-embedding",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": str(device),
        "n_test_shards": len(test_paths),
        "n_features_dim": int(feats.shape[1]),
        "n_samples_total": int(feats.shape[0]),
        "n_samples_silhouette": int(sil_feats.shape[0]),
        "silhouette_score": silhouette,
        "silhouette_min": args.silhouette_min,
        "silhouette_passes": silhouette >= args.silhouette_min,
        "umap_params": {
            "n_components": 2,
            "n_neighbors": args.umap_neighbors,
            "min_dist": args.umap_min_dist,
            "metric": args.umap_metric,
            "random_state": args.seed,
        },
        "label_counts": {
            SMOOTH_OR_FEATURED_NAMES[int(c)]: int((labels == c).sum())
            for c in sorted(set(labels.tolist()))
        },
    }
    out_json = args.out_dir / "umap_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[umap] wrote {out_json}", flush=True)

    if silhouette < args.silhouette_min:
        print(
            f"[umap] FAIL: silhouette {silhouette:.4f} < min {args.silhouette_min}",
            flush=True,
        )
        return 2
    print(
        f"[umap] PASS: silhouette {silhouette:.4f} >= min {args.silhouette_min}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
