"""CLI: download + combine the GZ DESI volunteer-vote catalogs (T2.1).

Downloads the two volunteer-classification parquets from Zenodo record
8331338 (``gz_desi_gzd8_volunteer_core_catalog.parquet`` and
``gz_desi_gzd8_volunteer_extended_catalog.parquet``), concatenates them,
deduplicates on ``dr8_id``, applies the ``--min-votes`` filter on the
always-asked questions, validates the GZ DESI vote schema, and writes
the merged dataframe to ``--out`` (default
``data/gz_desi_volunteer_decals.parquet``).

A run-config sidecar JSON (rule 8) is written next to the output with
the source files, row counts at each stage, settings (redacted),
git SHA, and pip freeze for reproducibility.

DEVPLAN T2.1 originally targeted ``gz_desi_500k.parquet`` with a
400-600k row gate; that figure was approximate. Zenodo 8331338's
volunteer subset is ~100k DECaLS-DR8 galaxies — the 8.67M-row
"deep_learning_catalog" parquets in the same record contain only
*predicted* vote fractions from a Zoobot model, not the integer
human-vote counts the Dirichlet-Multinomial loss needs. We therefore
ship the volunteer catalog with bounds adjusted to 80-150k rows.
See ``docs/SCHEMA.md`` for the full justification.

Usage::

    python scripts/build_gz_desi_catalog.py [--min-votes 5]

Requires ``[m1]`` extras (pandas + pyarrow) and a populated ``.env``
(``DATA_DIR`` is used as the Zenodo download cache).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from galaxy_vit.config import Settings
from galaxy_vit.data.gz_desi import validate_schema

ZENODO_RECORD_ID = "8331338"
ZENODO_API_BASE = "https://zenodo.org/api/records"
ZENODO_DOWNLOAD_TIMEOUT_S = 1800.0

# The two volunteer-vote-count parquets in Zenodo record 8331338. Both
# are DECaLS-DR8-only by construction (the GZD-8 phase of Galaxy Zoo
# DESI used DR8 imaging for its volunteer campaign). They share an
# identical column schema; we concatenate + deduplicate on ``dr8_id``.
DEFAULT_VOLUNTEER_FILES: tuple[str, ...] = (
    "gz_desi_gzd8_volunteer_core_catalog.parquet",
    "gz_desi_gzd8_volunteer_extended_catalog.parquet",
)

DR8_ID_COLUMN = "dr8_id"


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


def _resolve_zenodo_file(files: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for f in files:
        if f["key"] == name:
            return f
    available = "\n".join(
        f"  {f['key']:60s} {f['size'] / 1e6:>8.1f} MB"
        for f in files
        if f["key"].lower().endswith(".parquet")
    )
    raise FileNotFoundError(
        f"{name!r} not in Zenodo record {ZENODO_RECORD_ID}; "
        f"available .parquet files:\n{available}"
    )


def _stream_download(url: str, out_path: Path) -> None:
    """Stream a download with progress every 25 MB. Caches across reruns."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    next_log_at = 25 * 1024 * 1024
    with httpx.stream(
        "GET", url, timeout=ZENODO_DOWNLOAD_TIMEOUT_S, follow_redirects=True
    ) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        with out_path.open("wb") as fh:
            for chunk in r.iter_bytes(chunk_size=1 << 20):
                fh.write(chunk)
                bytes_written += len(chunk)
                if bytes_written >= next_log_at:
                    pct = (bytes_written / total * 100) if total else 0.0
                    print(
                        f"  ...{bytes_written / 1e6:>6.1f} / "
                        f"{total / 1e6 if total else 0:.1f} MB ({pct:.1f}%)",
                        flush=True,
                    )
                    next_log_at += 25 * 1024 * 1024
    print(
        f"  download complete: {bytes_written / 1e6:.1f} MB to {out_path}",
        flush=True,
    )


def _ensure_downloaded(
    file_meta: dict[str, Any],
    cache_dir: Path,
) -> Path:
    key = str(file_meta["key"])
    size = int(file_meta["size"])
    url = str(file_meta["links"]["self"])
    raw_path: Path = cache_dir / key
    if raw_path.is_file() and raw_path.stat().st_size == size:
        print(f"[build_gz_desi] using cached {raw_path}", flush=True)
        return raw_path
    print(
        f"[build_gz_desi] downloading {key} ({size / 1e6:.1f} MB)...",
        flush=True,
    )
    _stream_download(url, raw_path)
    return raw_path


def _apply_min_votes_filter(df: pd.DataFrame, min_votes: int) -> pd.DataFrame:
    """Keep rows where every always-asked question has >= ``min_votes``.

    Uses the per-question ``<question>_total-votes`` rollup column. NaN
    totals (galaxies that were never shown that question) are treated
    as 0 and therefore filtered out by the threshold.

    Dependent questions are not enforced — when the parent answer
    disqualifies them their totals are legitimately 0. The trainer's
    per-question loss mask (T3.4) handles dependency gating.
    """
    mask = pd.Series(True, index=df.index)
    for q in ("smooth-or-featured", "merging"):
        col = f"{q}_total-votes"
        totals = df[col].fillna(0)
        mask &= totals >= min_votes
    kept = df[mask]
    print(
        f"[build_gz_desi] min_votes>={min_votes} on always-asked questions: "
        f"{len(df):,} -> {len(kept):,} rows",
        flush=True,
    )
    return kept


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-votes",
        type=int,
        default=5,
        help="Minimum total votes per always-asked question (default 5).",
    )
    parser.add_argument(
        "--filenames",
        nargs="+",
        default=list(DEFAULT_VOLUNTEER_FILES),
        help="Zenodo files to combine (default: both GZD-8 volunteer catalogs).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/gz_desi_volunteer_decals.parquet"),
        help="Output parquet path.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Where to cache the raw Zenodo downloads. Defaults to $DATA_DIR/zenodo/.",
    )
    parser.add_argument(
        "--expected-min-rows",
        type=int,
        default=80_000,
        help="Lower bound on filtered row count (T2.1: 80k-150k).",
    )
    parser.add_argument(
        "--expected-max-rows",
        type=int,
        default=150_000,
        help="Upper bound on filtered row count (T2.1: 80k-150k).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = Settings()  # type: ignore[call-arg]
    print(
        f"[build_gz_desi] HF_USER={settings.HF_USER} DATA_DIR={settings.DATA_DIR}",
        flush=True,
    )

    cache_dir = args.cache_dir or (settings.DATA_DIR / "zenodo")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve all requested files in the Zenodo record.
    api_url = f"{ZENODO_API_BASE}/{ZENODO_RECORD_ID}"
    print(f"[build_gz_desi] fetching record metadata from {api_url}", flush=True)
    meta = httpx.get(api_url, timeout=30.0).raise_for_status().json()
    record_files: list[dict[str, Any]] = meta.get("files", [])

    file_metas = [_resolve_zenodo_file(record_files, name) for name in args.filenames]

    # 2. Download (or use cache) for each.
    raw_paths = [_ensure_downloaded(fm, cache_dir) for fm in file_metas]

    # 3. Load + concat + dedup + validate schema.
    print(f"[build_gz_desi] loading {len(raw_paths)} parquet(s)...", flush=True)
    dfs = []
    for p in raw_paths:
        d = pd.read_parquet(p)
        print(f"  {p.name}: {len(d):,} rows, {len(d.columns)} columns", flush=True)
        validate_schema(d.columns)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    n_before_dedup = len(df)
    df = df.drop_duplicates(subset=DR8_ID_COLUMN, keep="first").reset_index(drop=True)
    print(
        f"[build_gz_desi] concat + dedup on {DR8_ID_COLUMN!r}: "
        f"{n_before_dedup:,} -> {len(df):,} rows",
        flush=True,
    )

    # 4. Filter on min votes.
    df = _apply_min_votes_filter(df, args.min_votes)

    n_final = len(df)
    if not (args.expected_min_rows <= n_final <= args.expected_max_rows):
        raise RuntimeError(
            f"filtered row count {n_final:,} outside expected "
            f"[{args.expected_min_rows:,}, {args.expected_max_rows:,}]; "
            f"check filter logic"
        )

    # 5. Write.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, compression="snappy", index=False)
    print(
        f"[build_gz_desi] wrote {args.out} "
        f"({args.out.stat().st_size / 1e6:.1f} MB, {n_final:,} rows)",
        flush=True,
    )

    # 6. run_config.json sidecar (rule 8).
    cfg: dict[str, Any] = {
        "task": "T2.1-gz-desi-catalog",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "zenodo_record_id": ZENODO_RECORD_ID,
        "zenodo_files": [
            {
                "key": fm["key"],
                "size": int(fm["size"]),
                "url": fm["links"]["self"],
            }
            for fm in file_metas
        ],
        "min_votes": args.min_votes,
        "raw_rows_total": int(sum(_count_rows_from_parquet(p) for p in raw_paths)),
        "rows_after_dedup": int(n_before_dedup),
        "rows_after_min_votes": n_final,
        "filtered_columns": len(df.columns),
        "out_path": str(args.out),
        "settings_redacted": settings.redacted(),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "pip_freeze": _pip_freeze(),
    }
    cfg_path = args.out.parent / "gz_desi_volunteer_decals.run_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[build_gz_desi] wrote {cfg_path}", flush=True)
    return 0


def _count_rows_from_parquet(path: Path) -> int:
    """Read just metadata to get row count without loading the full file."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)  # type: ignore[no-untyped-call]
    return int(pf.metadata.num_rows)


if __name__ == "__main__":
    raise SystemExit(main())
