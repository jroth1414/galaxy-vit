"""T2.3 — Walmsley+23 per-question reproduction acceptance gate.

Two tiers of assertion:

* **Smoke**: macro_top1 above a basic floor and every per-question top-1
  above a per-question floor. Catches "training was broken" regressions.
* **Reference**: each per-question test top-1 within ``_tolerance_pp`` of
  Walmsley+23's published number (``walmsley23_per_question_top1.json``).
  Reference values are approximate pending exact verification against
  the paper; tolerance is wide (5 pp) until tightened.

Skipped if the run hasn't been done — re-run with::

    python -m galaxy_vit.training.multi_question_trainer \\
        --config configs/m2_w23_reproduction.yaml
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from galaxy_vit.data.gz_desi_labels import QUESTION_NAMES

METRICS_PATH = Path("runs/m2_w23_reproduction/metrics.json")
REF_PATH = Path("tests/fixtures/walmsley23_per_question_top1.json")


def _load_reference() -> dict[str, float]:
    return json.loads(REF_PATH.read_text(encoding="utf-8"))


@pytest.mark.skipif(
    not METRICS_PATH.is_file(),
    reason=(
        "run trainer first: "
        "python -m galaxy_vit.training.multi_question_trainer "
        "--config configs/m2_w23_reproduction.yaml"
    ),
)
def test_T2_3_macro_top1_above_smoke_floor() -> None:
    """Smoke check: training produced sensible per-question + macro accuracies."""
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    test_payload = payload["test"]
    macro = test_payload["macro_top1"]
    per_q = test_payload["per_question"]

    ref = _load_reference()
    macro_min = float(ref["_macro_min"])
    per_q_floor = float(ref["_per_question_floor"])

    assert macro >= macro_min, (
        f"test macro_top1 {macro:.4f} below smoke floor {macro_min}"
    )
    for q in QUESTION_NAMES:
        q_top1 = float(per_q[q]["top1"])
        n_valid = int(per_q[q]["n_valid"])
        if n_valid < 50:
            # Too few test samples for a stable per-question accuracy check;
            # skip the floor (still passes the macro_min above).
            continue
        assert q_top1 >= per_q_floor, (
            f"per-question top-1 for {q!r} = {q_top1:.4f} below floor {per_q_floor} "
            f"(n_valid={n_valid})"
        )


@pytest.mark.skipif(
    not METRICS_PATH.is_file(),
    reason="run trainer first (see test_T2_3_macro_top1_above_smoke_floor)",
)
def test_T2_3_per_question_within_tolerance_of_w23() -> None:
    """T2.3 paper-reproduction check: each per-question test top-1 within tolerance.

    Skipped while the reference JSON is in placeholder status — replace
    its values with the actual Walmsley+23 figures and flip ``_status``
    to ``verified`` to enable the comparison.

    Tolerance defaults to +/-5 pp; tighten to +/-2 pp (DEVPLAN target)
    via the ``_tolerance_pp`` field once the paper figures are
    confirmed.
    """
    ref = _load_reference()
    if ref.get("_status", "placeholder") != "verified":
        pytest.skip(
            "reference JSON is placeholder; populate with actual W+23 values + "
            "set _status='verified' to enable this test"
        )

    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    test_payload = payload["test"]
    per_q = test_payload["per_question"]
    tol = float(ref["_tolerance_pp"]) / 100.0

    failures = []
    for q in QUESTION_NAMES:
        q_top1 = float(per_q[q]["top1"])
        n_valid = int(per_q[q]["n_valid"])
        if n_valid < 50:
            continue
        ref_v = float(ref[q])
        if abs(q_top1 - ref_v) > tol:
            failures.append(
                f"{q}: ours={q_top1:.4f} ref={ref_v:.4f} "
                f"(diff={q_top1 - ref_v:+.4f}, tol=+/-{tol:.4f}, n_valid={n_valid})"
            )

    assert not failures, "per-question top-1 outside tolerance:\n  " + "\n  ".join(
        failures
    )
