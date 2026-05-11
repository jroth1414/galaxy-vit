"""T4.3 - Pre-compute a small demo-galaxies artifact for the Posteriors tab.

Picks ~N stratified DR8 test galaxies covering the smooth-or-featured
3-class diversity (smooth, featured-or-disk, artifact), saves:

* ``artifacts/demo_galaxies/thumbs/<id>.jpg`` -- 128x128 thumbnails (small;
  served directly to the frontend galaxy picker)
* ``artifacts/demo_galaxies/manifest.json`` -- per-galaxy:
  - ``id``: stable identifier (zero-padded index in the iteration order)
  - ``smooth_or_featured_plurality``: ground-truth coarse class
  - ``counts``: flat (num_answers,) list of integer vote counts
  - ``valid``: flat (num_questions,) bool list (per-question >= min_votes)

The Posteriors tab reads ``manifest.json`` for the dropdown contents
and the volunteer-overlay numbers; the actual model prediction is
computed on demand by ``/api/posteriors`` from the cached thumbnail
(running the encoder over a 128x128 image is fast enough for the
serve loop). The model PREDICTS on the thumbnail, then we compare
against the volunteer-observed counts also stored here.

Invocation::

    python -m scripts.build_demo_galaxies \\
        --config configs/m3_dirichlet.yaml \\
        --out artifacts/demo_galaxies \\
        --n 24
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import torch

from galaxy_vit.config import Settings
from galaxy_vit.data.gz_desi import GZ_DESI_QUESTIONS, NUM_ANSWERS
from galaxy_vit.data.gz_desi_hf import has_any_dr8_votes, hf_labels_to_canonical
from galaxy_vit.data.gz_desi_streaming import _iter_samples_from_shard
from galaxy_vit.training.dirichlet_trainer import DirichletConfig
from galaxy_vit.training.multi_question_trainer import _resolve_shards

DEFAULT_N = 24
THUMB_SIZE = 128
SMOOTH_OR_FEATURED_NAMES = ("smooth", "featured-or-disk", "artifact")


def _stratify_by_smooth_or_featured(counts_tensor: torch.Tensor) -> int:
    """Coarse 3-class label from the smooth-or-featured counts."""
    # smooth-or-featured is the first question; answers occupy indices 0..2.
    sof = counts_tensor[:3]
    if sof.sum() == 0:
        return -1
    return int(sof.argmax().item())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/demo_galaxies")
    )
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    cfg = DirichletConfig.from_yaml(args.config)
    settings = Settings()  # type: ignore[call-arg]
    os.environ["HF_TOKEN"] = settings.HF_TOKEN.get_secret_value()

    _, _, test_paths = _resolve_shards(cfg.data.shards)
    print(f"[demo] test shards: {len(test_paths)}", flush=True)

    # We need access to the underlying PIL images, not the transformed tensors.
    # Re-walk shards directly via _iter_samples_from_shard to keep the
    # full-resolution image around for thumbnailing.
    rng = random.Random(args.seed)

    # Pass 1: collect all DR8 candidates with (image, canonical counts, valid).
    candidates: list[dict[str, Any]] = []
    print("[demo] scanning shards for DR8 candidates...", flush=True)
    for shard in test_paths:
        for img, hf_labels in _iter_samples_from_shard(shard):
            if not has_any_dr8_votes(hf_labels):
                continue
            canonical = hf_labels_to_canonical(hf_labels)
            counts = []
            valid = []
            for question, answers in GZ_DESI_QUESTIONS.items():
                total = 0
                for answer in answers:
                    v = canonical.get(f"{question}_{answer}", 0)
                    c = 0 if v is None else int(v)
                    counts.append(c)
                    total += c
                valid.append(total >= cfg.data.min_votes)
            assert len(counts) == NUM_ANSWERS
            candidates.append(
                {
                    "image": img,
                    "counts": counts,
                    "valid": valid,
                }
            )
    print(f"[demo] {len(candidates)} DR8 candidates", flush=True)

    # Pass 2: stratified pick by smooth-or-featured plurality.
    per_class_target = max(1, args.n // 3)
    by_class: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for idx, c in enumerate(candidates):
        cls = _stratify_by_smooth_or_featured(torch.tensor(c["counts"]))
        if cls in by_class:
            by_class[cls].append(idx)
    chosen: list[int] = []
    for _cls, idxs in by_class.items():
        rng.shuffle(idxs)
        chosen.extend(idxs[:per_class_target])
    rng.shuffle(chosen)
    chosen = chosen[: args.n]
    print(
        f"[demo] stratified pick: smooth={sum(1 for i in chosen if _stratify_by_smooth_or_featured(torch.tensor(candidates[i]['counts'])) == 0)}  "
        f"featured-or-disk={sum(1 for i in chosen if _stratify_by_smooth_or_featured(torch.tensor(candidates[i]['counts'])) == 1)}  "
        f"artifact={sum(1 for i in chosen if _stratify_by_smooth_or_featured(torch.tensor(candidates[i]['counts'])) == 2)}",
        flush=True,
    )

    out_dir = args.out
    thumbs_dir = out_dir / "thumbs"
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for slot, cand_idx in enumerate(chosen):
        c = candidates[cand_idx]
        thumb_id = f"{slot:04d}"
        img = c["image"]
        thumb = img.resize((THUMB_SIZE, THUMB_SIZE)).convert("RGB")
        thumb_path = thumbs_dir / f"{thumb_id}.jpg"
        with open(thumb_path, "wb") as f:
            thumb.save(f, format="JPEG", quality=88)
        sof_cls = _stratify_by_smooth_or_featured(torch.tensor(c["counts"]))
        manifest.append(
            {
                "id": thumb_id,
                "smooth_or_featured_plurality": (
                    SMOOTH_OR_FEATURED_NAMES[sof_cls] if sof_cls >= 0 else "unknown"
                ),
                "counts": c["counts"],
                "valid": c["valid"],
            }
        )

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[demo] wrote {manifest_path} ({len(manifest)} galaxies)", flush=True)
    print(f"[demo] wrote {len(manifest)} thumbnails to {thumbs_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
