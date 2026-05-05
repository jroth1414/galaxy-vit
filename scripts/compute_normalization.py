"""CLI: compute per-channel mean / std on the Galaxy10 train split.

Outputs (next to ``--out``):

* ``normalization.json`` — ``{"mean": [c0,c1,c2], "std": [c0,c1,c2], ...}``
  with values in ``[0, 1]``. Frozen after T1.2 — T1.3+ augmentations read
  this file and never recompute.
* ``run_config.json`` — same reproducibility metadata (seed, git SHA,
  pip freeze, dataset id, settings redacted, n_train) emitted by every
  script per DEVPLAN rule 8.

Usage::

    python scripts/compute_normalization.py [--seed 42] \\
        [--splits-csv data/splits/galaxy10_split.csv] \\
        [--out configs/normalization.json]

Requires ``[m1]`` extra (datasets + Pillow) and a populated ``.env``
(HF_TOKEN). Reuses the HF cache populated by ``build_galaxy10_split.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from galaxy_vit.config import Settings
from galaxy_vit.data.galaxy10 import GALAXY10_HF_ID, load_galaxy10
from galaxy_vit.data.transforms import compute_normalization_stats

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from numpy.typing import NDArray

DEFAULT_OUT = Path("configs/normalization.json")
DEFAULT_SPLITS_CSV = Path("data/splits/galaxy10_split.csv")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _pip_freeze() -> list[str]:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def _read_train_indices(splits_csv: Path) -> list[int]:
    """Return sorted global indices of rows marked split='train' in the CSV."""
    train: list[int] = []
    with splits_csv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == "train":
                train.append(int(row["index"]))
    train.sort()
    return train


def _iter_train_images(
    data_dir: Path,
    train_indices: list[int],
    *,
    log_every: int = 1000,
) -> Iterator[NDArray[Any]]:
    """Yield numpy uint8 image arrays for each global index marked train.

    Bridges the global index space defined in ``data/splits/galaxy10_split.csv``
    (concatenation of HF Galaxy10 splits in ``sorted(ds.keys())`` order — i.e.
    "test" then "train") back to ``(hf_split, local_idx)`` for HF lookup.
    """
    ds = load_galaxy10(data_dir)
    splits_in_order = sorted(ds.keys())

    # Build [start, end) ranges per HF split in concatenation order.
    boundaries: list[tuple[str, int, int]] = []
    cumulative = 0
    for split_key in splits_in_order:
        n = len(ds[split_key])
        boundaries.append((split_key, cumulative, cumulative + n))
        cumulative += n

    for i, global_idx in enumerate(train_indices, start=1):
        for split_key, start, end in boundaries:
            if start <= global_idx < end:
                local_idx = global_idx - start
                row = ds[split_key][local_idx]
                yield np.array(row["image"], dtype=np.uint8)
                break
        else:
            raise IndexError(
                f"global index {global_idx} out of range for HF dataset "
                f"(total rows = {cumulative})"
            )
        if i % log_every == 0:
            print(f"  ...processed {i}/{len(train_indices)} train images", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits-csv", type=Path, default=DEFAULT_SPLITS_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = Settings()  # type: ignore[call-arg]
    print(
        f"[compute_normalization] HF_USER={settings.HF_USER} "
        f"DATA_DIR={settings.DATA_DIR}",
        flush=True,
    )
    os.environ["HF_TOKEN"] = settings.HF_TOKEN.get_secret_value()

    if not args.splits_csv.is_file():
        raise FileNotFoundError(
            f"splits CSV not found at {args.splits_csv}; "
            f"run `python scripts/build_galaxy10_split.py` first (T1.1)."
        )

    train_indices = _read_train_indices(args.splits_csv)
    print(
        f"[compute_normalization] {len(train_indices)} train indices "
        f"loaded from {args.splits_csv}",
        flush=True,
    )

    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[compute_normalization] iterating train images from {GALAXY10_HF_ID}...",
        flush=True,
    )
    images = _iter_train_images(settings.DATA_DIR, train_indices)
    stats = compute_normalization_stats(images)

    print(
        f"[compute_normalization] mean={[round(v, 6) for v in stats['mean']]} "
        f"std={[round(v, 6) for v in stats['std']]}",
        flush=True,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "mean": stats["mean"],
        "std": stats["std"],
        "channels": ["R", "G", "B"],
        "value_range": [0.0, 1.0],
        "n_train": len(train_indices),
        "dataset_hf_id": GALAXY10_HF_ID,
        "splits_csv": str(args.splits_csv),
        "seed": args.seed,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[compute_normalization] wrote {args.out}", flush=True)

    cfg: dict[str, Any] = {
        "task": "T1.2-decals-normalization",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "splits_csv": str(args.splits_csv),
        "n_train": len(train_indices),
        "dataset_hf_id": GALAXY10_HF_ID,
        "out_json": str(args.out),
        "stats": payload,
        "settings_redacted": settings.redacted(),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "pip_freeze": _pip_freeze(),
    }
    cfg_path = args.out.parent / "run_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[compute_normalization] wrote {cfg_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
