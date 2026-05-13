"""S-4 — Precompute GradCAM overlays for every UMAP-set test thumbnail.

For each ``artifacts/test_thumbs/<idx>.jpg`` (the 2,462 deterministic
thumbnails ``scripts/build_test_thumbs.py`` produced for the Explorer
tab), run the M1 Galaxy10 classifier's GradCAM and save the alpha-
blended overlay to ``artifacts/test_saliencies/<idx>.jpg`` at the same
128x128 resolution as the source.

Why M1 (not the M3 Dirichlet model): the existing Classify tab and the
``/api/predict`` endpoint both use the M1 GradCAM, so saliency on
hover stays visually consistent across the demo. Per-question
GradCAMs against the M3 Dirichlet head are A-7's scope.

Output sizing: 2462 x ~6 KB at JPEG quality 85 = ~15 MB total.
Committed alongside ``artifacts/test_thumbs/``.

Invocation::

    python -m scripts.build_test_saliencies
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from galaxy_vit.inference.predict import GalaxyClassifier

DEFAULT_CKPT = Path("runs/m1_zoobot_finetune/best.pt")
DEFAULT_THUMBS = Path("artifacts/test_thumbs")
DEFAULT_OUT = Path("artifacts/test_saliencies")
JPEG_QUALITY = 85


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--thumbs-dir", type=Path, default=DEFAULT_THUMBS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render every overlay even if it already exists on disk.",
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"[saliency] missing checkpoint: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.thumbs_dir.is_dir():
        print(f"[saliency] missing thumbs dir: {args.thumbs_dir}", file=sys.stderr)
        return 2

    thumb_paths = sorted(args.thumbs_dir.glob("*.jpg"))
    if not thumb_paths:
        print(f"[saliency] no jpgs under {args.thumbs_dir}", file=sys.stderr)
        return 2
    if args.limit is not None:
        thumb_paths = thumb_paths[: args.limit]
    print(f"[saliency] {len(thumb_paths)} thumbnails to process", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)

    print(
        f"[saliency] loading classifier from {args.checkpoint} on {args.device}",
        flush=True,
    )
    classifier = GalaxyClassifier(args.checkpoint, device=args.device)

    from PIL import Image as PILImage_

    n_written = 0
    n_skipped = 0
    for i, path in enumerate(thumb_paths):
        expected_name = f"{i:05d}.jpg"
        if path.name != expected_name:
            print(
                f"[saliency] WARN: thumbnail #{i} has name {path.name} "
                f"(expected {expected_name}); idx alignment may drift",
                file=sys.stderr,
            )
        out_path = args.out / expected_name
        if out_path.is_file() and not args.overwrite:
            n_skipped += 1
            continue
        image = PILImage_.open(path).convert("RGB")
        overlay = classifier.gradcam_overlay(image)
        overlay.save(out_path, format="JPEG", quality=JPEG_QUALITY)
        n_written += 1
        if (n_written + n_skipped) % 250 == 0:
            print(
                f"[saliency] processed {n_written + n_skipped}/{len(thumb_paths)} "
                f"(written={n_written} skipped={n_skipped})",
                flush=True,
            )

    print(
        f"[saliency] done: wrote {n_written}, skipped {n_skipped}, "
        f"out_dir={args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
