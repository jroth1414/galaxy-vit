"""S-1 — Precompute Zoobot-encoder features for the 2,462 UMAP-set thumbnails.

Iterates ``artifacts/test_thumbs/<idx>.jpg`` (the deterministic set
``scripts/build_test_thumbs.py`` produced for the Explorer tab) in row-
index order, encodes each through the M3 (Dirichlet) checkpoint's
Zoobot ConvNeXt-nano encoder, and writes a single parquet whose row
index aligns 1:1 with the test-thumb idx:

    artifacts/test_thumb_features.parquet

Schema:

    features    list<float32>     length = encoder.num_features (640 for ConvNeXt-nano)

Row count == number of jpgs under ``artifacts/test_thumbs/``. Feature
vectors are NOT L2-normalised here — that's the consumer's job
(:class:`galaxy_vit.inference.similarity.SimilarityIndex` normalises on
load).

Output size: 2462 x 640 x 4 bytes raw ≈ 6 MB; parquet ~3 MB compressed.
Safe to commit alongside ``artifacts/umap_coords.parquet``.

Why we encode the saved 128x128 thumbnails (instead of re-iterating
the HF DESI test shards at native 256x256): the live demo always
serves test_thumbs/<idx>.jpg through the same eval transform when a
user clicks a UMAP point. Encoding the saved thumbnails here means the
cached feature for idx i and a fresh encode of ``test_thumbs/i.jpg``
are bit-for-bit identical, so the kNN sanity property
(``topk_by_index(i)[0].distance == 0``) holds exactly.

Invocation::

    python -m scripts.cache_test_thumb_features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from galaxy_vit.data.transforms import build_eval_transform, load_normalization
from galaxy_vit.inference.similarity import encode_image_to_feature
from galaxy_vit.models.dirichlet_head import build_zoobot_dirichlet

DEFAULT_THUMBS = Path("artifacts/test_thumbs")
DEFAULT_OUT = Path("artifacts/test_thumb_features.parquet")
DEFAULT_CKPT = Path("runs/m3_dirichlet/best.pt")
DEFAULT_NORMALIZATION = Path("configs/normalization.json")
DEFAULT_IMAGE_SIZE = 224


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thumbs-dir", type=Path, default=DEFAULT_THUMBS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument(
        "--normalization", type=Path, default=DEFAULT_NORMALIZATION
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for debugging; default = all thumbnails.",
    )
    args = parser.parse_args(argv)

    if not args.thumbs_dir.is_dir():
        print(f"[cache] thumbs dir not found: {args.thumbs_dir}", file=sys.stderr)
        return 2
    if not args.checkpoint.is_file():
        print(
            f"[cache] checkpoint not found: {args.checkpoint}; "
            "run the T3.6 trainer first",
            file=sys.stderr,
        )
        return 2

    thumb_paths = sorted(args.thumbs_dir.glob("*.jpg"))
    if not thumb_paths:
        print(f"[cache] no jpgs under {args.thumbs_dir}", file=sys.stderr)
        return 2
    if args.limit is not None:
        thumb_paths = thumb_paths[: args.limit]
    print(f"[cache] {len(thumb_paths)} thumbnails to encode", flush=True)

    device = torch.device(args.device)
    print(f"[cache] device={device}", flush=True)

    print(f"[cache] loading checkpoint {args.checkpoint}", flush=True)
    model, _encoder, _head = build_zoobot_dirichlet()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    mean, std = load_normalization(args.normalization)
    transform = build_eval_transform(image_size=args.image_size, mean=mean, std=std)

    from PIL import Image as PILImage_

    feats_per_row: list[list[float]] = []
    for i, path in enumerate(thumb_paths):
        # Sanity: filenames are zero-padded indices; row order must
        # match numeric idx exactly so /api/test_thumbs/{idx}/thumbnail
        # and the feature cache stay aligned.
        expected_name = f"{i:05d}.jpg"
        if path.name != expected_name:
            print(
                f"[cache] WARN: thumbnail #{i} has name {path.name} "
                f"(expected {expected_name}); cache row order may drift",
                file=sys.stderr,
            )
        image = PILImage_.open(path).convert("RGB")
        feat = encode_image_to_feature(model, transform, image, device=device)
        feats_per_row.append(feat.squeeze(0).tolist())
        if (i + 1) % 250 == 0:
            print(f"[cache] encoded {i + 1}/{len(thumb_paths)}", flush=True)

    import pandas as pd

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"features": feats_per_row})
    df.to_parquet(args.out, index=False)
    n_dim = len(feats_per_row[0]) if feats_per_row else 0
    print(
        f"[cache] wrote {args.out} "
        f"({len(feats_per_row)} rows x {n_dim} dims)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
