"""CLI: download a subset of HF mwalmsley/gz_desi_wds tar shards (T2.3).

Pulls ``--n-train`` train shards and ``--n-test`` test shards into the
local HF cache via ``huggingface_hub.snapshot_download``. Each shard is
~39 MB / 512 image+label pairs at 300x300 resolution. The shard
filenames follow the convention ``gz_desi_{train,test}_{N}_512.tar``,
where N is a 0-indexed integer.

DR8 coverage: based on a 20-sample test from shard 0, ~30% of galaxies
in each shard have DR8 votes (the rest are DR5 or DR12 from older
campaigns). So ``--n-train 100`` ⇒ ~50k samples ⇒ ~15k DR8-voted
galaxies, which is a reasonable training set for the W+23 reproduction.

Default targets (~4.7 GB total): 100 train + 20 test shards.

Usage::

    python scripts/download_gz_desi_wds.py [--n-train 100] [--n-test 20]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from galaxy_vit.config import Settings

HF_REPO = "mwalmsley/gz_desi_wds"
SHARD_PATTERN = "data/gz_desi_{split}_{idx}_512.tar"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-train",
        type=int,
        default=100,
        help="Number of train shards to download (~50k samples; default 100).",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=20,
        help="Number of test shards to download (~10k samples; default 20).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HF cache dir override. Defaults to $HF_HOME or "
        "~/.cache/huggingface (HF defaults).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = Settings()  # type: ignore[call-arg]

    print(
        f"[download] HF_USER={settings.HF_USER} repo={HF_REPO} "
        f"n_train={args.n_train} n_test={args.n_test}",
        flush=True,
    )

    # Build allow_patterns for snapshot_download
    patterns = ["README.md"]
    for i in range(args.n_train):
        patterns.append(SHARD_PATTERN.format(split="train", idx=i))
    for i in range(args.n_test):
        patterns.append(SHARD_PATTERN.format(split="test", idx=i))

    print(
        f"[download] downloading {len(patterns) - 1} shards "
        f"(~{(args.n_train + args.n_test) * 39 / 1000:.1f} GB)...",
        flush=True,
    )
    cache_dir_arg = str(args.cache_dir) if args.cache_dir else None
    snapshot = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        allow_patterns=patterns,
        cache_dir=cache_dir_arg,
    )
    print(f"[download] downloaded to {snapshot}", flush=True)

    # Inventory
    data_dir = Path(snapshot) / "data"
    train_shards = sorted(data_dir.glob("gz_desi_train_*_512.tar"))
    test_shards = sorted(data_dir.glob("gz_desi_test_*_512.tar"))
    total_size = sum(p.stat().st_size for p in train_shards + test_shards)
    print(
        f"[download] inventory: {len(train_shards)} train + {len(test_shards)} test "
        f"= {len(train_shards) + len(test_shards)} shards "
        f"({total_size / 1e9:.2f} GB)",
        flush=True,
    )

    cfg: dict[str, Any] = {
        "task": "T2.3-download-gz-desi-wds",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hf_repo": HF_REPO,
        "n_train_shards": len(train_shards),
        "n_test_shards": len(test_shards),
        "snapshot_path": str(snapshot),
        "total_size_bytes": total_size,
        "settings_redacted": settings.redacted(),
        "git_sha": _git_sha(),
        "python_version": sys.version,
    }
    cfg_dest = settings.DATA_DIR / "shards" / "gz_desi_wds_run_config.json"
    cfg_dest.parent.mkdir(parents=True, exist_ok=True)
    cfg_dest.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[download] wrote {cfg_dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
