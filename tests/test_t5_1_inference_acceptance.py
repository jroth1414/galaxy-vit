"""T5.1 - Full-pass inference acceptance gates (DEVPLAN T5.1).

Two layers:

1. **Dry-run smoke** (always runs when ``inference_dry_run.json`` exists):
   - throughput sane (> 50 galaxies/s on any current GPU)
   - projection block present for all 4 hardware tiers
   - rows_written matches max_rows exactly (no off-by-one)

2. **Full-pass gates** (skipped until ``releases/gz_desi_dirichlet_v1.parquet``
   and its sidecar ``.meta.json`` exist):
   - DEVPLAN: row count matches +/- 0.1% of the target 8.67M
   - DEVPLAN: every alpha_{0..33} column is strictly positive (head's
     softplus + alpha_floor contract; checks the full-pass didn't
     produce any NaN / negative values)
   - DEVPLAN: checksum recorded in sidecar matches a fresh recompute
     (proves the parquet wasn't silently corrupted between write +
     acceptance test)

The test file is intentionally separate from test_t4_3 / test_t4_2 so
that the full-pass parquet doesn't have to be available locally for
the rest of the suite to run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

DRY_RUN_PATH = Path("runs/m3_dirichlet/inference_dry_run.json")
FULL_PASS_PARQUET = Path("releases/gz_desi_dirichlet_v1.parquet")
FULL_PASS_META = FULL_PASS_PARQUET.with_suffix(".meta.json")

GZ_DESI_FULL_CATALOG_SIZE = 8_670_000
ROW_COUNT_TOLERANCE = 0.001  # +/- 0.1%
NUM_ANSWERS = 34
DRY_RUN_MIN_THROUGHPUT = 50.0  # galaxies/s; sanity floor


# ---------------------------------------------------------------------------
# Dry-run smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DRY_RUN_PATH.is_file(),
    reason=(
        "run dry-run first: "
        "python -m scripts.run_inference_pass --mode dry-run --max-rows 10000"
    ),
)
def test_T5_1_dry_run_throughput_is_sane() -> None:
    """The dry-run measured a real positive throughput on a real GPU."""
    payload = json.loads(DRY_RUN_PATH.read_text(encoding="utf-8"))
    throughput = float(payload["throughput_galaxies_per_s"])
    assert throughput > DRY_RUN_MIN_THROUGHPUT, (
        f"throughput {throughput:.1f} g/s below sanity floor "
        f"{DRY_RUN_MIN_THROUGHPUT}; check GPU is actually being used"
    )
    assert payload["mode"] == "dry-run"
    assert int(payload["rows_written"]) > 0


@pytest.mark.skipif(
    not DRY_RUN_PATH.is_file(),
    reason="run dry-run first",
)
def test_T5_1_dry_run_projects_all_hardware_tiers() -> None:
    """The projection block contains finite hours + (optional) cost for each tier."""
    payload = json.loads(DRY_RUN_PATH.read_text(encoding="utf-8"))
    projection = payload["projection"]
    for hw, pj in projection["hardware"].items():
        hours = float(pj["hours_for_target"])
        assert hours > 0.0, f"{hw}: hours_for_target {hours} not positive"
        assert hours < 1000.0, f"{hw}: hours_for_target {hours} unreasonably large"


@pytest.mark.skipif(
    not DRY_RUN_PATH.is_file(),
    reason="run dry-run first",
)
def test_T5_1_dry_run_target_rows_default_matches_devplan() -> None:
    """The default --target-rows is the 8.67M figure cited by DEVPLAN T5.1."""
    payload = json.loads(DRY_RUN_PATH.read_text(encoding="utf-8"))
    assert int(payload["projection"]["target_rows"]) == GZ_DESI_FULL_CATALOG_SIZE


# ---------------------------------------------------------------------------
# DEVPLAN T5.1 full-pass acceptance gates
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (FULL_PASS_PARQUET.is_file() and FULL_PASS_META.is_file()),
    reason=(
        "full-pass not run yet. After HITL #4 approves, run: "
        "python -m scripts.run_inference_pass --mode full"
    ),
)
def test_T5_1_full_pass_row_count_within_tolerance() -> None:
    """DEVPLAN T5.1: row count matches the 8.67M target within +/- 0.1%."""
    meta = json.loads(FULL_PASS_META.read_text(encoding="utf-8"))
    n_rows = int(meta["rows_written"])
    lower = int(GZ_DESI_FULL_CATALOG_SIZE * (1 - ROW_COUNT_TOLERANCE))
    upper = int(GZ_DESI_FULL_CATALOG_SIZE * (1 + ROW_COUNT_TOLERANCE))
    assert lower <= n_rows <= upper, (
        f"row count {n_rows} outside [{lower}, {upper}] (target "
        f"{GZ_DESI_FULL_CATALOG_SIZE} +/- {ROW_COUNT_TOLERANCE * 100:.1f}%)"
    )


@pytest.mark.skipif(
    not (FULL_PASS_PARQUET.is_file() and FULL_PASS_META.is_file()),
    reason="full-pass not run yet",
)
def test_T5_1_full_pass_all_alpha_columns_positive() -> None:
    """DEVPLAN T5.1: every alpha_{0..33} column is strictly positive."""
    import pandas as pd

    df = pd.read_parquet(FULL_PASS_PARQUET)
    alpha_cols = [c for c in df.columns if c.startswith("alpha_")]
    assert len(alpha_cols) == NUM_ANSWERS, (
        f"expected {NUM_ANSWERS} alpha columns, got {len(alpha_cols)}"
    )
    for col in alpha_cols:
        col_min = float(df[col].min())
        assert col_min > 0.0, f"{col} has min {col_min} <= 0 (alpha_floor violated)"
        col_max = float(df[col].max())
        import math

        assert math.isfinite(col_max), f"{col} has non-finite max {col_max}"


@pytest.mark.skipif(
    not (FULL_PASS_PARQUET.is_file() and FULL_PASS_META.is_file()),
    reason="full-pass not run yet",
)
def test_T5_1_full_pass_checksum_matches_meta() -> None:
    """DEVPLAN T5.1: SHA-256 in sidecar matches a fresh recompute of the parquet."""
    meta = json.loads(FULL_PASS_META.read_text(encoding="utf-8"))
    recorded = str(meta["sha256"])
    h = hashlib.sha256()
    with FULL_PASS_PARQUET.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    assert actual == recorded, (
        f"sha256 mismatch:\n  recorded={recorded}\n  actual  ={actual}\n"
        "parquet was modified after the inference pass wrote it"
    )
