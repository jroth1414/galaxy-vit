"""CLI: stratified 70/15/15 train/val/test split for the GZ DESI catalog (T2.3).

Reads ``data/gz_desi_volunteer_decals.parquet`` (T2.1 output), derives a
3-class stratification key from the smooth-or-featured plurality answer,
runs :func:`galaxy_vit.data.splits.stratified_split`, and writes the
split assignment to ``data/splits/gz_desi_volunteer_decals_split.csv``
(plus a sibling ``run_config.json`` per rule 8).

Output CSV schema::

    dr8_id,strat_label,split

Where ``strat_label`` is ``0/1/2`` (smooth / featured-or-disk / artifact)
and ``split`` is ``train``/``val``/``test``. Downstream training reads
this file and filters the streaming shard dataset by ``dr8_id``.

Usage::

    python scripts/build_gz_desi_split.py [--seed 42]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from galaxy_vit.config import Settings
from galaxy_vit.data.gz_desi_labels import stratification_label
from galaxy_vit.data.splits import stratified_split

DEFAULT_CATALOG = Path("data/gz_desi_volunteer_decals.parquet")
DEFAULT_OUT = Path("data/splits/gz_desi_volunteer_decals_split.csv")
DEFAULT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=list(DEFAULT_RATIOS),
        metavar=("TRAIN", "VAL", "TEST"),
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
        f"[build_gz_desi_split] HF_USER={settings.HF_USER} catalog={args.catalog}",
        flush=True,
    )

    if not args.catalog.is_file():
        raise FileNotFoundError(
            f"catalog not found at {args.catalog}; run "
            f"`python scripts/build_gz_desi_catalog.py` first (T2.1)"
        )

    df = pd.read_parquet(args.catalog)
    print(f"[build_gz_desi_split] catalog: {len(df):,} rows", flush=True)

    # Derive 3-class stratification labels.
    strat_labels: list[int] = []
    for _, row in df.iterrows():
        strat_labels.append(stratification_label(row.to_dict()))
    n_invalid = sum(1 for x in strat_labels if x < 0)
    if n_invalid:
        print(
            f"[build_gz_desi_split] WARNING: {n_invalid} rows have <5 votes on "
            f"smooth-or-featured (T2.1 filter should have removed these); "
            f"ignoring",
            flush=True,
        )
    valid_mask = [s >= 0 for s in strat_labels]
    df_valid = df[valid_mask].reset_index(drop=True)
    valid_strat = [s for s in strat_labels if s >= 0]

    print(
        f"[build_gz_desi_split] valid rows for split: {len(df_valid):,}; "
        f"strat distribution: {dict(pd.Series(valid_strat).value_counts().sort_index())}",
        flush=True,
    )

    # Stratified split.
    splits = stratified_split(valid_strat, ratios=ratios, seed=args.seed)
    for name, indices in splits.items():
        print(f"  {name}: {len(indices):,}", flush=True)

    # Build CSV: dr8_id, strat_label, split.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_total = len(df_valid)
    index_to_split: dict[int, str] = {}
    for split_name, indices in splits.items():
        for idx in indices:
            index_to_split[idx] = split_name

    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dr8_id", "strat_label", "split"])
        for i in range(n_total):
            writer.writerow(
                [df_valid.iloc[i]["dr8_id"], int(valid_strat[i]), index_to_split[i]]
            )

    print(
        f"[build_gz_desi_split] wrote {args.out} ({n_total:,} rows)",
        flush=True,
    )

    # Per-split, per-strat-class breakdown.
    breakdown: dict[str, dict[int, int]] = {}
    for split_name, indices in splits.items():
        counts: dict[int, int] = {}
        for idx in indices:
            counts[valid_strat[idx]] = counts.get(valid_strat[idx], 0) + 1
        breakdown[split_name] = dict(sorted(counts.items()))

    cfg: dict[str, Any] = {
        "task": "T2.3-build-gz-desi-split",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "ratios": list(ratios),
        "catalog": str(args.catalog),
        "out_csv": str(args.out),
        "n_rows_total": len(df),
        "n_rows_valid": n_total,
        "n_rows_dropped_invalid": n_invalid,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_strat_breakdown": breakdown,
        "settings_redacted": settings.redacted(),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "pip_freeze": _pip_freeze(),
    }
    cfg_path = args.out.parent / "gz_desi_volunteer_decals_split.run_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[build_gz_desi_split] wrote {cfg_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
