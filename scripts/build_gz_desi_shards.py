"""CLI: pre-fetch DECaLS cutouts for the GZ DESI volunteer catalog (T2.2).

For each row in ``data/gz_desi_volunteer_decals.parquet``, fetches a
256x256 DECaLS-DR8 JPEG cutout from the Legacy Survey viewer and packs
it (alongside a JSON metadata sidecar with vote counts + dr8_id) into
WebDataset tar shards under ``$DATA_DIR/shards/gz_desi_volunteer/``.

This script is **not** required for the T2.2 acceptance gate — that
gate runs against synthetic shards. This script is what the user runs
once when they're ready to actually train (T2.3+).

Bandwidth: ~100k galaxies x ~75 KB JPEG ~= 7.5 GB on disk. With
``--max-concurrent 8`` and a typical home connection, end-to-end
fetch is ~15-30 minutes.

Usage::

    python scripts/build_gz_desi_shards.py [--limit N] [--max-concurrent 8]

Requires ``[m1]`` (pandas) and ``[m1-train]`` (webdataset is not
strictly used here — we write tars directly with :mod:`tarfile` —
but the loader needs it).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from galaxy_vit.config import Settings
from galaxy_vit.data.gz_desi import (
    expected_vote_count_columns,
    expected_vote_total_columns,
)

LEGACY_VIEWER_BASE = "https://www.legacysurvey.org/viewer/cutout.jpg"
DEFAULT_LAYER = "ls-dr8"
DEFAULT_PIXEL_SCALE = 0.262  # arcsec/pix at native DECaLS resolution
DEFAULT_IMAGE_SIZE = 256
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_SHARD_SIZE = 3_000


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


def _fetch_one(
    client: httpx.Client,
    ra: float,
    dec: float,
    *,
    image_size: int,
    pixel_scale: float,
    layer: str,
    timeout: float,
    max_retries: int,
) -> bytes:
    """Fetch a single DECaLS cutout JPEG with bounded retries."""
    params = {
        "ra": f"{ra}",
        "dec": f"{dec}",
        "layer": layer,
        "pixscale": f"{pixel_scale}",
        "size": str(image_size),
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = client.get(LEGACY_VIEWER_BASE, params=params, timeout=timeout)
            r.raise_for_status()
            if not r.content or len(r.content) < 256:
                raise RuntimeError(
                    f"suspiciously small response ({len(r.content)} bytes); "
                    f"likely a 'no imaging' fallback"
                )
            return r.content
        except (httpx.HTTPError, RuntimeError) as exc:
            last_err = exc
            if attempt < max_retries - 1:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(
        f"DECaLS cutout fetch failed for ra={ra}, dec={dec}: {last_err}"
    ) from last_err


def _row_metadata(row: pd.Series) -> dict[str, Any]:
    """Extract the JSON sidecar fields from a catalog row."""
    md: dict[str, Any] = {
        "dr8_id": row["dr8_id"],
        "ra": float(row["ra"]),
        "dec": float(row["dec"]),
    }
    for col in expected_vote_count_columns() + expected_vote_total_columns():
        v = row[col]
        # NaN totals on dependent questions are real — keep them as null
        # in the JSON so downstream code can treat null vs 0 distinctly.
        md[col] = None if pd.isna(v) else int(v)
    return md


def _write_shard(
    shard_path: Path,
    items: list[tuple[str, bytes, dict[str, Any]]],
) -> None:
    """Write a list of (key, jpg_bytes, metadata_dict) into a single tar."""
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(shard_path, "w") as tf:
        for key, jpg_bytes, metadata in items:
            jpg_info = tarfile.TarInfo(name=f"{key}.jpg")
            jpg_info.size = len(jpg_bytes)
            tf.addfile(jpg_info, io.BytesIO(jpg_bytes))
            meta_bytes = json.dumps(metadata).encode("utf-8")
            meta_info = tarfile.TarInfo(name=f"{key}.json")
            meta_info.size = len(meta_bytes)
            tf.addfile(meta_info, io.BytesIO(meta_bytes))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/gz_desi_volunteer_decals.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to $DATA_DIR/shards/gz_desi_volunteer.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Galaxies per tar shard (default 3000).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after this many galaxies (default: full catalog).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Concurrent DECaLS fetches (default 8; respects Legacy Survey API).",
    )
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--pixel-scale", type=float, default=DEFAULT_PIXEL_SCALE)
    parser.add_argument("--layer", type=str, default=DEFAULT_LAYER)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = Settings()  # type: ignore[call-arg]

    out_dir = args.out_dir or (settings.DATA_DIR / "shards" / "gz_desi_volunteer")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[shards] catalog={args.catalog} out_dir={out_dir} "
        f"shard_size={args.shard_size} limit={args.limit} "
        f"max_concurrent={args.max_concurrent}",
        flush=True,
    )

    df = pd.read_parquet(args.catalog)
    if args.limit is not None:
        df = df.head(args.limit)
    n_total = len(df)
    n_shards = math.ceil(n_total / args.shard_size)
    print(f"[shards] {n_total:,} galaxies -> {n_shards} shards", flush=True)

    n_failed = 0
    shard_metadata: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    with httpx.Client(http2=False, follow_redirects=True) as client:
        for shard_i in range(n_shards):
            shard_path = out_dir / f"gz_desi_shard_{shard_i:04d}.tar"
            if shard_path.is_file():
                print(f"[shards] {shard_path.name} already exists; skipping", flush=True)
                continue

            start = shard_i * args.shard_size
            end = min(start + args.shard_size, n_total)
            rows = df.iloc[start:end]

            items: list[tuple[str, bytes, dict[str, Any]]] = []

            with ThreadPoolExecutor(max_workers=args.max_concurrent) as ex:
                futures = {}
                for global_idx, (_, row) in enumerate(rows.iterrows(), start=start):
                    key = f"{global_idx:08d}"
                    fut = ex.submit(
                        _fetch_one,
                        client,
                        float(row["ra"]),
                        float(row["dec"]),
                        image_size=args.image_size,
                        pixel_scale=args.pixel_scale,
                        layer=args.layer,
                        timeout=DEFAULT_TIMEOUT_S,
                        max_retries=DEFAULT_MAX_RETRIES,
                    )
                    futures[fut] = (key, row)

                for fut in as_completed(futures):
                    key, row = futures[fut]
                    try:
                        jpg_bytes = fut.result()
                    except Exception as exc:
                        n_failed += 1
                        print(
                            f"[shards] FAIL {key} (dr8_id={row['dr8_id']}): {exc}",
                            flush=True,
                        )
                        continue
                    items.append((key, jpg_bytes, _row_metadata(row)))

            items.sort(key=lambda triple: triple[0])
            _write_shard(shard_path, items)

            elapsed = time.perf_counter() - t0
            shard_metadata.append(
                {
                    "shard": shard_path.name,
                    "n_items": len(items),
                    "size_mb": round(shard_path.stat().st_size / 1e6, 2),
                }
            )
            print(
                f"[shards] {shard_path.name}: {len(items)}/{end - start} items "
                f"({shard_path.stat().st_size / 1e6:.1f} MB) — "
                f"elapsed {elapsed:.1f}s",
                flush=True,
            )

    cfg: dict[str, Any] = {
        "task": "T2.2-build-gz-desi-shards",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "catalog": str(args.catalog),
        "out_dir": str(out_dir),
        "image_size": args.image_size,
        "pixel_scale": args.pixel_scale,
        "layer": args.layer,
        "shard_size": args.shard_size,
        "n_total": n_total,
        "n_shards": n_shards,
        "n_failed": n_failed,
        "shards": shard_metadata,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "settings_redacted": settings.redacted(),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "pip_freeze": _pip_freeze(),
    }
    cfg_path = out_dir / "run_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[shards] wrote {cfg_path}", flush=True)
    if n_failed:
        print(
            f"[shards] WARNING: {n_failed} galaxies failed to fetch; "
            f"see stderr for details",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
