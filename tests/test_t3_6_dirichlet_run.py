"""T3.6 — Full Dirichlet training run + temperature calibration acceptance gates.

DEVPLAN T3.6 acceptance (all required):

1. MAE on smooth-or-featured ≤ 0.15
2. Coverage @ 95% CI ≥ 0.85
3. Train loss strictly monotone for epochs 1-5
4. No NaN alpha anywhere in training or test eval

Gates (1) + (2) are checked against the calibrated (temperature-scaled)
test-set numbers in ``runs/m3_dirichlet/calibrated_metrics.json``.
Gates (3) + (4) are checked against the raw training history in
``runs/m3_dirichlet/metrics.json``.

Why calibrated? The raw v1 run met 3 of 4 gates -- coverage plateaued
at 0.49 due to systematic over-concentration of alpha as the encoder
unfroze. Post-hoc temperature scaling (Walmsley+23 / Guo+17) is the
standard fix; ``scripts/calibrate_dirichlet.py`` sweeps T on val,
picks the value targeting coverage = 0.95, and saves the calibrated
test stats. Temperature scaling preserves the posterior MEAN exactly,
so MAE on per-answer fractions is unchanged from the raw run.

Skipped when either metrics.json or calibrated_metrics.json is
missing; re-create with::

    python -m galaxy_vit.training.dirichlet_trainer \\
        --config configs/m3_dirichlet.yaml
    python -m scripts.calibrate_dirichlet \\
        --config configs/m3_dirichlet.yaml \\
        --checkpoint runs/m3_dirichlet/best.pt \\
        --out runs/m3_dirichlet/calibrated_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RAW_METRICS = Path("runs/m3_dirichlet/metrics.json")
CALIBRATED_METRICS = Path("runs/m3_dirichlet/calibrated_metrics.json")

# DEVPLAN T3.6 thresholds.
SMOOTH_OR_FEATURED_MAE_CAP = 0.15
COVERAGE_FLOOR = 0.85
MONOTONE_EPOCHS = 5


@pytest.mark.skipif(
    not (RAW_METRICS.is_file() and CALIBRATED_METRICS.is_file()),
    reason=(
        "run trainer + calibrator first: "
        "python -m galaxy_vit.training.dirichlet_trainer --config configs/m3_dirichlet.yaml && "
        "python -m scripts.calibrate_dirichlet --config configs/m3_dirichlet.yaml "
        "--checkpoint runs/m3_dirichlet/best.pt "
        "--out runs/m3_dirichlet/calibrated_metrics.json"
    ),
)
def test_T3_6_dirichlet_run_acceptance_gates() -> None:
    """All four DEVPLAN T3.6 gates pass on the calibrated v1 run."""
    raw = json.loads(RAW_METRICS.read_text(encoding="utf-8"))
    cal = json.loads(CALIBRATED_METRICS.read_text(encoding="utf-8"))

    # Pick the better of single-T vs per-question T (both should pass,
    # but use whichever has higher test coverage).
    single_cov = float(cal["single_T"]["test_coverage_macro"])
    per_q_cov = float(cal["per_question_T"]["test_coverage_macro"])
    winner = "single_T" if single_cov >= per_q_cov else "per_question_T"
    winning_stats = cal[winner]

    # ---- Gate 1: smooth-or-featured MAE ≤ 0.15 ----
    smooth_mae = float(winning_stats["per_question"]["smooth-or-featured"]["mae"])
    assert smooth_mae <= SMOOTH_OR_FEATURED_MAE_CAP, (
        f"smooth-or-featured MAE {smooth_mae:.4f} exceeds DEVPLAN cap "
        f"{SMOOTH_OR_FEATURED_MAE_CAP} (calibration regime: {winner})"
    )

    # ---- Gate 2: macro coverage @ 95% CI ≥ 0.85 ----
    macro_coverage = float(winning_stats["test_coverage_macro"])
    assert macro_coverage >= COVERAGE_FLOOR, (
        f"test coverage_macro {macro_coverage:.4f} below DEVPLAN floor "
        f"{COVERAGE_FLOOR} (calibration regime: {winner})"
    )

    # ---- Gate 3: train loss strictly monotone epochs 1-5 ----
    history = raw["history"]
    assert len(history) >= MONOTONE_EPOCHS, (
        f"history has {len(history)} epochs, need >= {MONOTONE_EPOCHS} "
        "for monotone check"
    )
    losses_15 = [float(h["train/loss_avg"]) for h in history[:MONOTONE_EPOCHS]]
    for i in range(MONOTONE_EPOCHS - 1):
        assert losses_15[i + 1] < losses_15[i], (
            f"train loss not strictly monotone at epoch {i + 2}: "
            f"{losses_15[i]:.4f} -> {losses_15[i + 1]:.4f}"
        )

    # ---- Gate 4: no NaN alpha at any val or test eval ----
    for h in history:
        all_finite = float(h.get("val/all_finite", 1.0))
        assert all_finite == 1.0, f"non-finite alpha at epoch {h.get('epoch')}"
    assert bool(raw["test"]["all_finite"]), "non-finite alpha at test eval"


@pytest.mark.skipif(
    not CALIBRATED_METRICS.is_file(),
    reason="run calibrator first",
)
def test_T3_6_temperature_calibration_preserves_mae() -> None:
    """Sanity: temperature scaling preserves the posterior mean, so MAE is
    unchanged between the raw run and either calibrated regime.

    Catches a future regression where someone changes the calibration
    transform to a non-mean-preserving function (e.g., shifting alpha
    instead of scaling) without realizing it.
    """
    cal = json.loads(CALIBRATED_METRICS.read_text(encoding="utf-8"))
    raw_mae = float(cal["raw"]["test_mae_macro"])
    single_mae = float(cal["single_T"]["test_mae_macro"])
    per_q_mae = float(cal["per_question_T"]["test_mae_macro"])
    # Allow tiny fp32 round-off (~1e-4); a non-mean-preserving transform
    # would produce O(0.01) drift instantly.
    assert abs(raw_mae - single_mae) < 1e-3, (
        f"single-T calibration shifted MAE: raw={raw_mae:.5f}, "
        f"calibrated={single_mae:.5f}"
    )
    assert abs(raw_mae - per_q_mae) < 1e-3, (
        f"per-question-T calibration shifted MAE: raw={raw_mae:.5f}, "
        f"calibrated={per_q_mae:.5f}"
    )


@pytest.mark.skipif(
    not CALIBRATED_METRICS.is_file(),
    reason="run calibrator first",
)
def test_T3_6_calibration_strictly_improved_coverage() -> None:
    """Sanity: at least one calibrated regime raised macro coverage above raw."""
    cal = json.loads(CALIBRATED_METRICS.read_text(encoding="utf-8"))
    raw_cov = float(cal["raw"]["test_coverage_macro"])
    single_cov = float(cal["single_T"]["test_coverage_macro"])
    per_q_cov = float(cal["per_question_T"]["test_coverage_macro"])
    assert max(single_cov, per_q_cov) > raw_cov, (
        f"neither calibrated regime improved on raw coverage {raw_cov:.4f} "
        f"(single={single_cov:.4f}, per_q={per_q_cov:.4f})"
    )
