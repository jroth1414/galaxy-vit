"""T4.2 - Build the full DR8 test-set thumbnail atlas for the UMAP Explorer.

The T2.4 ``extract_umap.py`` produced ``artifacts/umap_coords.parquet``
with 2,462 rows (one per DR8 test galaxy that survived the
``has_any_dr8_votes`` filter). The Explorer tab needs a thumbnail per
row for the hover-preview + lasso sample-grid.

This script iterates the test shards in the same deterministic order
T2.4 used (sorted shard glob, sequential per-shard sample order,
``has_any_dr8_votes`` filter) and dumps a thumbnail for each row,
indexed by row number to align with umap_coords.parquet:

    artifacts/test_thumbs/<row_idx>.jpg

Each thumbnail is 128x128 JPEG at quality 85 (~4 KB each). Total
~10 MB across the full set.

Invocation::

    python -m scripts.build_test_thumbs \\
        --config configs/m3_dirichlet.yaml
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from galaxy_vit.config import Settings
from galaxy_vit.data.gz_desi import GZ_DESI_QUESTIONS
from galaxy_vit.data.gz_desi_hf import has_any_dr8_votes, hf_labels_to_canonical
from galaxy_vit.data.gz_desi_streaming import _iter_samples_from_shard
from galaxy_vit.training.dirichlet_trainer import DirichletConfig
from galaxy_vit.training.multi_question_trainer import _resolve_shards

THUMB_SIZE = 128
DEFAULT_OUT = Path("artifacts/test_thumbs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap for debugging; default = all DR8 test galaxies.",
    )
    args = parser.parse_args(argv)

    cfg = DirichletConfig.from_yaml(args.config)
    settings = Settings()  # type: ignore[call-arg]
    os.environ["HF_TOKEN"] = settings.HF_TOKEN.get_secret_value()

    _, _, test_paths = _resolve_shards(cfg.data.shards)
    print(f"[thumbs] test shards: {len(test_paths)}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    # Wipe any stale entries so the row indices stay contiguous.
    for stale in args.out.glob("*.jpg"):
        stale.unlink()

    # Same filter chain T2.4's extract_umap.py applied: DR8 only AND
    # smooth-or-featured has >= min_votes (otherwise the row would have
    # been dropped from umap_coords.parquet's index too).
    sof_answers = GZ_DESI_QUESTIONS["smooth-or-featured"]
    sof_keys = [f"smooth-or-featured_{a}" for a in sof_answers]
    row_idx = 0
    for shard in test_paths:
        for img, hf_labels in _iter_samples_from_shard(shard):
            if not has_any_dr8_votes(hf_labels):
                continue
            canonical = hf_labels_to_canonical(hf_labels)
            sof_total = sum(int(canonical.get(k, 0) or 0) for k in sof_keys)
            if sof_total < cfg.data.min_votes:
                continue
            if args.limit is not None and row_idx >= args.limit:
                break
            thumb = img.resize((THUMB_SIZE, THUMB_SIZE)).convert("RGB")
            thumb.save(args.out / f"{row_idx:05d}.jpg", format="JPEG", quality=85)
            row_idx += 1
            if row_idx % 250 == 0:
                print(f"[thumbs] wrote {row_idx} thumbnails", flush=True)
        if args.limit is not None and row_idx >= args.limit:
            break
    print(f"[thumbs] wrote {row_idx} thumbnails to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
