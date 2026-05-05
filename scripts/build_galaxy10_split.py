"""CLI: download Galaxy10 DECaLS and write the canonical splits CSV.

Produces two files under ``--out``'s parent directory:

* ``galaxy10_split.csv`` — ``index,label,split`` rows for every dataset row.
* ``run_config.json``    — resolved settings (redacted), seed, ratios, git
                           SHA, ``pip freeze``, dataset HF ID, per-split
                           counts.  Required by DEVPLAN rule 8 ("every
                           script writes run_config.json next to outputs").

Usage::

    python scripts/build_galaxy10_split.py [--seed 42] [--out data/splits/galaxy10_split.csv]

Requires:
* ``[m1]`` optional extra installed (``pip install -e ".[m1]"``).
* ``HF_TOKEN`` populated in ``.env`` (read via ``galaxy_vit.config.Settings``).
* Network access to Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from galaxy_vit.config import Settings
from galaxy_vit.data.galaxy10 import (
    GALAXY10_HF_ID,
    GALAXY10_NUM_CLASSES,
    make_split,
)

DEFAULT_OUT = Path("data/splits/galaxy10_split.csv")


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the stratified split (default: 42).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output CSV path (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[0.70, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Three ratios summing to 1.0 (default: 0.70 0.15 0.15).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    ratios: tuple[float, float, float] = (
        float(args.ratios[0]),
        float(args.ratios[1]),
        float(args.ratios[2]),
    )

    settings = Settings()  # type: ignore[call-arg]
    print(
        f"[build_galaxy10_split] HF_USER={settings.HF_USER} "
        f"DATA_DIR={settings.DATA_DIR}",
        flush=True,
    )

    # Bridge .env -> process env so the `datasets` / `huggingface_hub`
    # libraries pick up the token. pydantic-settings populates the Settings
    # model but does not mutate os.environ, so this step is required.
    os.environ["HF_TOKEN"] = settings.HF_TOKEN.get_secret_value()

    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"[build_galaxy10_split] loading {GALAXY10_HF_ID} "
        f"(downloads on first call)...",
        flush=True,
    )
    rows = make_split(settings.DATA_DIR, seed=args.seed, ratios=ratios)
    print(f"[build_galaxy10_split] split {len(rows)} rows", flush=True)

    # Write CSV.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "label", "split"])
        writer.writerows(rows)
    print(f"[build_galaxy10_split] wrote {args.out}", flush=True)

    # Per-split + per-class breakdown for the run log.
    split_counts: Counter[str] = Counter()
    class_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for _idx, label, split in rows:
        split_counts[split] += 1
        class_counts[split][label] += 1
    for split_name in ("train", "val", "test"):
        print(
            f"  {split_name}: n={split_counts[split_name]:>6d}",
            flush=True,
        )

    # Persist run_config.json next to the CSV (rule 8).
    cfg: dict[str, Any] = {
        "task": "T1.1-galaxy10-split",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "ratios": list(ratios),
        "dataset_hf_id": GALAXY10_HF_ID,
        "num_classes": GALAXY10_NUM_CLASSES,
        "out_csv": str(args.out),
        "n_rows": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts_per_split": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(class_counts.items())
        },
        "settings_redacted": settings.redacted(),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "pip_freeze": _pip_freeze(),
    }
    cfg_path = args.out.parent / "run_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[build_galaxy10_split] wrote {cfg_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
