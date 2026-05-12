"""T5.2 - Science case acceptance gate.

DEVPLAN T5.2 (substituted from "bar fraction vs z" to "bar fraction vs
bulge size"; see docs/science_note.md): qualitative trend direction
matches the volunteer ground truth.

Concrete check: Spearman correlation of model-predicted bar fraction
against the volunteer-observed bulge ordinal has the same SIGN as the
volunteer-computed correlation, and both are statistically significant.
This is the substantive check that the model recovers a real
morphology relation; replicates the W+23 direction-matching gate
phrased for the substituted morphology pair.

Plus structural sanity:

* Per-bin sample counts non-empty across the realistic bulge classes.
* Both correlations significant at p < 0.001 (loose; we observed
  p ~ 1e-22).

Skipped when the artifact JSON is missing -- regenerate with
``python -m scripts.science_bar_vs_bulge``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

METRICS_PATH = Path("artifacts/bar_fraction_vs_bulge_metrics.json")
PNG_PATH = Path("artifacts/bar_fraction_vs_bulge.png")
SIGNIFICANCE_P = 1e-3
MIN_GALAXIES_TOTAL = 1000


@pytest.mark.skipif(
    not METRICS_PATH.is_file(),
    reason=(
        "run scripts/science_bar_vs_bulge.py first to produce "
        "artifacts/bar_fraction_vs_bulge_metrics.json"
    ),
)
def test_T5_2_model_and_volunteer_spearman_have_same_sign() -> None:
    """T5.2 acceptance: model + volunteer trends agree on direction.

    Both Spearman correlations must be (a) the same sign and (b)
    significant at p < 1e-3. The substantive check is sign agreement;
    the significance gate just rules out 'both noisy zeros'.
    """
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    assert PNG_PATH.is_file(), f"missing science figure: {PNG_PATH}"

    rho_model = float(metrics["spearman_model"]["rho"])
    p_model = float(metrics["spearman_model"]["p"])
    rho_vol = float(metrics["spearman_volunteer"]["rho"])
    p_vol = float(metrics["spearman_volunteer"]["p"])

    assert p_model < SIGNIFICANCE_P, (
        f"model Spearman p={p_model:.2g} not significant (>= {SIGNIFICANCE_P})"
    )
    assert p_vol < SIGNIFICANCE_P, (
        f"volunteer Spearman p={p_vol:.2g} not significant (>= {SIGNIFICANCE_P})"
    )
    # Same SIGN check: rho_model * rho_vol > 0 iff they agree on direction.
    assert rho_model * rho_vol > 0, (
        f"model rho={rho_model:.3f} and volunteer rho={rho_vol:.3f} "
        "disagree on direction (Spearman signs differ)"
    )


@pytest.mark.skipif(
    not METRICS_PATH.is_file(),
    reason="run scripts/science_bar_vs_bulge.py first",
)
def test_T5_2_sufficient_galaxies_in_analysis() -> None:
    """At least 1000 galaxies survived the gating + min_votes filter.

    Catches a future regression where the inference parquet's dr8_id
    column drifts and the join collapses to a near-empty intersection.
    """
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    n = int(metrics["n_valid_bar_and_bulge"])
    assert n >= MIN_GALAXIES_TOTAL, (
        f"only {n} galaxies in bar-vs-bulge analysis (need >= {MIN_GALAXIES_TOTAL}); "
        "check the dr8_id join between inference + volunteer catalog"
    )


@pytest.mark.skipif(
    not METRICS_PATH.is_file(),
    reason="run scripts/science_bar_vs_bulge.py first",
)
def test_T5_2_per_bin_population_non_degenerate() -> None:
    """The two largest bulge bins (moderate, small) hold the bulk of galaxies."""
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    per_bin = metrics["per_bin"]
    moderate_n = int(per_bin["moderate"]["n"])
    small_n = int(per_bin["small"]["n"])
    # Moderate + small should hold > 50% of the analysed sample (sanity:
    # the volunteer bulge-class distribution is not pathological).
    total = sum(int(per_bin[b]["n"]) for b in per_bin)
    assert (moderate_n + small_n) / total > 0.50, (
        f"moderate + small bins hold only "
        f"{(moderate_n + small_n) / total:.1%} of sample; bulge distribution looks off"
    )
