"""T2.5 — Per-question calibration acceptance gate.

Reads ``artifacts/calibration_metrics.json`` (written by
``scripts/extract_calibration.py``) and asserts:

* macro-ECE across the 10 GZ DESI questions is at most 0.10.
* The 2x5 reliability overview PNG and per-question PNGs are present
  for every question that has data.
* All 10 questions reported nonzero valid examples (catches a partial
  test-set capture).
* No per-question ECE is catastrophic (each <= 0.20 — twice the macro
  floor; a single question blowing past 0.20 strongly suggests a bug
  upstream even if the macro mean stays under 0.10).

Skipped when the script hasn't been run; re-run with::

    python -m scripts.extract_calibration \\
        --config configs/m2_w23_reproduction.yaml \\
        --checkpoint runs/m2_w23_reproduction/best.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from galaxy_vit.data.gz_desi_labels import QUESTION_NAMES

METRICS_PATH = Path("artifacts/calibration_metrics.json")
OVERVIEW_PNG = Path("artifacts/reliability_overview.png")
MACRO_ECE_FLOOR = 0.10
PER_QUESTION_ECE_CEILING = 0.20


@pytest.mark.skipif(
    not METRICS_PATH.is_file(),
    reason=(
        "run calibration extractor first: "
        "python -m scripts.extract_calibration "
        "--config configs/m2_w23_reproduction.yaml "
        "--checkpoint runs/m2_w23_reproduction/best.pt"
    ),
)
def test_T2_5_calibration_macro_ece_under_floor() -> None:
    """T2.5 acceptance: macro-ECE <= 0.10 across the 10 GZ DESI questions."""
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    assert OVERVIEW_PNG.is_file(), f"missing reliability overview: {OVERVIEW_PNG}"

    macro_ece = float(metrics["macro_ece"])
    assert macro_ece <= MACRO_ECE_FLOOR, (
        f"macro-ECE {macro_ece:.4f} above T2.5 floor {MACRO_ECE_FLOOR}; "
        f"see {METRICS_PATH} for per-question breakdown"
    )

    per_q = metrics["per_question"]
    missing = [q for q in QUESTION_NAMES if q not in per_q]
    assert not missing, f"calibration metrics missing for question(s): {missing}"

    failures: list[str] = []
    for q in QUESTION_NAMES:
        m = per_q[q]
        n_valid = int(m["n_valid"])
        if n_valid == 0:
            failures.append(f"{q}: n_valid=0 (no test samples captured)")
            continue
        per_q_png = Path(f"artifacts/calibration_{q}.png")
        if not per_q_png.is_file():
            failures.append(f"{q}: missing per-question figure {per_q_png}")
        ece = float(m["ece"])
        if ece > PER_QUESTION_ECE_CEILING:
            failures.append(
                f"{q}: ECE={ece:.4f} above per-question ceiling "
                f"{PER_QUESTION_ECE_CEILING} (n_valid={n_valid})"
            )

    assert not failures, "calibration sanity checks failed:\n  " + "\n  ".join(failures)
