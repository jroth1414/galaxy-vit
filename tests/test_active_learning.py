"""T4.1 - Active-learning acquisitions + loop acceptance.

Two layers:

1. **Unit tests** on acquisition math (no checkpoint, no dataset) -- formula
   sanity, monotonicity, edge cases. These prove the acquisitions are
   correctly implemented; they catch regressions independent of the
   experiment outcome.

2. **Acceptance test** (DEVPLAN T4.1) reading
   ``artifacts/active_learning_metrics.json`` produced by
   ``scripts/run_active_learning.py``: entropy acquisition reaches 90%
   of the full-data MAE in <= 60% of labels, across 3 seeds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("galaxy_datasets")

from galaxy_vit.training.active_learning import (  # noqa: E402
    ALRound,
    acquisition_scores,
    bald_score,
    predictive_entropy,
    reaches_target_at_fraction,
)

METRICS_PATH = Path("artifacts/active_learning_metrics.json")


# ---------------------------------------------------------------------------
# Acquisition unit tests
# ---------------------------------------------------------------------------


def test_T4_1_predictive_entropy_max_at_uniform() -> None:
    """For a single 2-answer question, entropy is max at p = (0.5, 0.5)."""
    # Uniform alpha -> uniform p -> max entropy log(2) ~ 0.693.
    uniform = torch.tensor([[1.0, 1.0]])
    sharp = torch.tensor([[100.0, 1.0]])  # nearly all mass on answer 0
    h_uniform = predictive_entropy(uniform, question_groups=[("q", 0, 2)])
    h_sharp = predictive_entropy(sharp, question_groups=[("q", 0, 2)])
    assert h_uniform.item() == pytest.approx(math.log(2.0), abs=1e-6)
    assert h_sharp.item() < 0.1


def test_T4_1_predictive_entropy_sums_across_questions() -> None:
    """Per-galaxy score = sum of per-question entropies (verified analytically)."""
    alpha = torch.tensor([[1.0, 1.0, 50.0, 1.0]])
    groups = [("a", 0, 2), ("b", 2, 4)]
    h = predictive_entropy(alpha, question_groups=groups)
    # q_a: p = [0.5, 0.5], H = log 2 = 0.693
    # q_b: p = [50/51, 1/51], H = -(50/51)*log(50/51) - (1/51)*log(1/51)
    p_q2 = [50 / 51, 1 / 51]
    h_q2 = -(p_q2[0] * math.log(p_q2[0]) + p_q2[1] * math.log(p_q2[1]))
    expected = math.log(2.0) + h_q2
    assert h.item() == pytest.approx(expected, abs=1e-4)


def test_T4_1_bald_is_nonnegative() -> None:
    """BALD = H[p_pred] - E[H[p|alpha]] >= 0 always (mutual info bound)."""
    torch.manual_seed(0)
    alpha = torch.rand(50, 5) * 5.0 + 1.1
    bald = bald_score(alpha, question_groups=[("q", 0, 5)])
    assert (bald >= -1e-5).all(), f"BALD went negative: min={bald.min().item()}"


def test_T4_1_bald_le_predictive_entropy() -> None:
    """BALD = H[p] - E[H[p|alpha]] <= H[p] since both terms non-negative."""
    torch.manual_seed(1)
    alpha = torch.rand(30, 4) * 3.0 + 1.1
    groups = [("q", 0, 4)]
    bald = bald_score(alpha, question_groups=groups)
    h = predictive_entropy(alpha, question_groups=groups)
    assert (bald <= h + 1e-5).all(), (
        f"BALD exceeded H: max diff = {(bald - h).max().item()}"
    )


def test_T4_1_bald_vanishes_for_high_concentration() -> None:
    """As alpha grows, BALD -> 0 (model is certain, no MI between vote + params)."""
    high_alpha = torch.tensor([[1000.0, 1000.0]])
    bald = bald_score(high_alpha, question_groups=[("q", 0, 2)])
    assert bald.item() < 0.01


def test_T4_1_bald_peaks_for_uniform_low_alpha() -> None:
    """For low-concentration uniform alpha, BALD > 0 (max disagreement)."""
    low_uniform = torch.tensor([[1.1, 1.1]])
    bald = bald_score(low_uniform, question_groups=[("q", 0, 2)])
    # For Dirichlet(1.1, 1.1) the model is uncertain AND knows it is.
    assert bald.item() > 0.05


def test_T4_1_random_acquisition_returns_unsorted() -> None:
    """random acquisition shouldn't track alpha; same seed gives same scores."""
    alpha = torch.rand(10, 4)
    s1 = acquisition_scores(
        alpha, question_groups=[("q", 0, 4)], method="random", rng_seed=42
    )
    s2 = acquisition_scores(
        alpha, question_groups=[("q", 0, 4)], method="random", rng_seed=42
    )
    s3 = acquisition_scores(
        alpha, question_groups=[("q", 0, 4)], method="random", rng_seed=43
    )
    assert torch.allclose(s1, s2)
    assert not torch.allclose(s1, s3)


def test_T4_1_unknown_acquisition_raises() -> None:
    alpha = torch.rand(5, 3) + 1.1
    with pytest.raises(ValueError, match="unknown acquisition"):
        acquisition_scores(
            alpha, question_groups=[("q", 0, 3)], method="silly",  # type: ignore[arg-type]
        )


def test_T4_1_reaches_target_at_fraction_helper() -> None:
    """The helper used by the acceptance test does what it says."""
    history = [
        ALRound(n_labeled=100, fraction_labeled=0.10, test_mae_macro=0.20, test_coverage_macro=0.7),
        ALRound(n_labeled=300, fraction_labeled=0.30, test_mae_macro=0.15, test_coverage_macro=0.8),
        ALRound(n_labeled=600, fraction_labeled=0.60, test_mae_macro=0.11, test_coverage_macro=0.85),
    ]
    # target_mae=0.12 at 60% cap: round-3 has MAE 0.11 at 60%, qualifies.
    assert reaches_target_at_fraction(history, target_mae=0.12, label_fraction_cap=0.60)
    # target_mae=0.10 nowhere reachable.
    assert not reaches_target_at_fraction(history, target_mae=0.10, label_fraction_cap=0.60)
    # target_mae=0.15 at 30% cap: round-2 qualifies exactly.
    assert reaches_target_at_fraction(history, target_mae=0.15, label_fraction_cap=0.30)


# ---------------------------------------------------------------------------
# DEVPLAN T4.1 acceptance (reads artifact)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not METRICS_PATH.is_file(),
    reason=(
        "run scripts/run_active_learning.py first: "
        "python -m scripts.run_active_learning"
    ),
)
def test_T4_1_entropy_reaches_90pct_full_mae_at_60pct_labels() -> None:
    """DEVPLAN T4.1: entropy acquisition reaches 90% of full-data MAE in <= 60%
    of labels across all 3 seeds."""
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    full_mae = float(payload["full_data_mae"])
    # "Reaches 90% of full-data MAE" -- interpreted strictly as "MAE no
    # worse than 1/0.9 ~= 1.111x the full-data MAE". Lower MAE is better,
    # so the target is a CEILING: ``mae <= full_mae / 0.9``.
    target_mae = full_mae / 0.9
    label_cap = 0.60

    per_seed = payload["seeds"]
    failures: list[str] = []
    for seed_str, seed_results in per_seed.items():
        entropy_curve = seed_results["entropy"]["curve"]
        history = [
            ALRound(
                n_labeled=r["n_labeled"],
                fraction_labeled=r["fraction_labeled"],
                test_mae_macro=r["test_mae_macro"],
                test_coverage_macro=r["test_coverage_macro"],
            )
            for r in entropy_curve
        ]
        if not reaches_target_at_fraction(
            history, target_mae=target_mae, label_fraction_cap=label_cap
        ):
            failures.append(
                f"seed {seed_str}: entropy curve never reaches MAE <= {target_mae:.4f} "
                f"at fraction <= {label_cap}; final MAE = {history[-1].test_mae_macro:.4f}"
            )

    assert not failures, "T4.1 acceptance failures:\n  " + "\n  ".join(failures)
    assert len(per_seed) >= 3, (
        f"DEVPLAN requires >= 3 seeds; metrics file has {len(per_seed)}"
    )


@pytest.mark.skipif(
    not METRICS_PATH.is_file(),
    reason="run scripts/run_active_learning.py first",
)
def test_T4_1_entropy_not_worse_than_random_at_target_fraction() -> None:
    """Sanity: at the 60% acceptance fraction, entropy MAE is no worse than
    random + 0.005.

    At 100% labels both methods see the full pool and head-training noise
    dominates, so the final-round comparison is uninformative. The
    meaningful comparison is at the fraction where AL is supposed to be
    beating random -- the DEVPLAN target fraction.
    """
    payload = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    target_fraction = 0.60
    failures: list[str] = []
    for seed_str, seed_results in payload["seeds"].items():
        # Find the round closest to (but not exceeding) the target.
        entropy_curve = seed_results["entropy"]["curve"]
        random_curve = seed_results["random"]["curve"]
        # Same fraction grid across methods (init + acquisitions), so just match index.
        idx = max(
            i for i, r in enumerate(entropy_curve)
            if r["fraction_labeled"] <= target_fraction
        )
        e_mae = entropy_curve[idx]["test_mae_macro"]
        r_mae = random_curve[idx]["test_mae_macro"]
        if e_mae > r_mae + 0.005:
            failures.append(
                f"seed {seed_str} at {entropy_curve[idx]['fraction_labeled']:.0%}: "
                f"entropy={e_mae:.4f} worse than random={r_mae:.4f}"
            )
    assert not failures, "entropy underperformed random:\n  " + "\n  ".join(failures)
