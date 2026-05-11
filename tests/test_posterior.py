"""T3.7 — Analytic posterior module acceptance tests.

DEVPLAN T3.7 acceptance: ``test_beta_ci_matches_scipy`` within 1e-4.

Plus sanity checks on:

* posterior_mean = alpha_i / sum(alpha) per question (the Dirichlet mean).
* CI widths shrink as concentration grows (more data -> tighter posterior).
* CI bounds are in [0, 1] and ordered (lower <= upper).
* Coverage on perfectly-calibrated synthetic data approaches the target ci.
* Coverage with a per-question valid mask only counts valid (galaxy, answer)
  pairs in the denominator.
* Argument validation rejects ci outside (0, 1) and shape mismatches.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("galaxy_datasets")

from galaxy_vit.data.schema import question_index_groups  # noqa: E402
from galaxy_vit.inference.posterior import (  # noqa: E402
    coverage,
    credible_interval,
    marginal_beta_params,
    posterior_mean,
)

NUM_ANSWERS = 34


# ---------------------------------------------------------------------------
# DEVPLAN T3.7 acceptance — Beta CI matches scipy
# ---------------------------------------------------------------------------


def test_T3_7_beta_ci_matches_scipy() -> None:
    """credible_interval bounds match scipy.stats.beta.ppf within 1e-4 (DEVPLAN gate)."""
    import numpy as np
    from scipy.stats import beta as scipy_beta

    torch.manual_seed(0)
    alpha = torch.rand(8, NUM_ANSWERS) * 4.0 + 1.1
    groups = question_index_groups()
    lower, upper = credible_interval(alpha, question_groups=groups, ci=0.95)

    # Independently compute the bounds via scipy for each question slice.
    a = alpha.numpy()
    for _name, start, end in groups:
        slice_a = a[:, start:end]
        A = slice_a.sum(axis=-1, keepdims=True)
        b_ref = A - slice_a
        lo_ref = scipy_beta.ppf(0.025, slice_a, b_ref)
        hi_ref = scipy_beta.ppf(0.975, slice_a, b_ref)
        assert np.allclose(
            lower[:, start:end].numpy(), lo_ref, atol=1e-4
        ), f"lower bound mismatch for {start}:{end}"
        assert np.allclose(
            upper[:, start:end].numpy(), hi_ref, atol=1e-4
        ), f"upper bound mismatch for {start}:{end}"


def test_T3_7_beta_ci_at_99pct_matches_scipy() -> None:
    """Tighter quantiles (99% CI) also match scipy within 1e-4."""
    import numpy as np
    from scipy.stats import beta as scipy_beta

    torch.manual_seed(1)
    alpha = torch.rand(4, 5) * 6.0 + 1.5
    groups = [("q", 0, 5)]
    lower, upper = credible_interval(alpha, question_groups=groups, ci=0.99)
    a = alpha.numpy()
    A = a.sum(axis=-1, keepdims=True)
    b = A - a
    assert np.allclose(lower.numpy(), scipy_beta.ppf(0.005, a, b), atol=1e-4)
    assert np.allclose(upper.numpy(), scipy_beta.ppf(0.995, a, b), atol=1e-4)


# ---------------------------------------------------------------------------
# Mean / structural sanity
# ---------------------------------------------------------------------------


def test_T3_7_posterior_mean_equals_dirichlet_mean() -> None:
    """posterior_mean[:, i] = alpha_i / sum(alpha_q) for each question slice."""
    torch.manual_seed(2)
    alpha = torch.rand(6, NUM_ANSWERS) * 3.0 + 1.1
    groups = question_index_groups()
    means = posterior_mean(alpha, question_groups=groups)
    for _name, start, end in groups:
        slice_alpha = alpha[:, start:end].float()
        denom = slice_alpha.sum(dim=-1, keepdim=True)
        expected = slice_alpha / denom
        assert torch.allclose(means[:, start:end], expected, atol=1e-6)
        # Each slice sums to 1 (proper probability simplex).
        slice_sum = means[:, start:end].sum(dim=-1)
        assert torch.allclose(slice_sum, torch.ones_like(slice_sum), atol=1e-5)


def test_T3_7_marginal_beta_params_satisfy_a_plus_b_eq_total() -> None:
    """For each (i, q): a_i + b_i = sum(alpha_q)."""
    torch.manual_seed(3)
    alpha = torch.rand(5, NUM_ANSWERS) * 2.0 + 1.5
    groups = question_index_groups()
    a, b = marginal_beta_params(alpha, question_groups=groups)
    for _name, start, end in groups:
        A_q = alpha[:, start:end].float().sum(dim=-1, keepdim=True)
        # a + b should equal A_q broadcast across the answer dim.
        sum_ab = a[:, start:end] + b[:, start:end]
        assert torch.allclose(sum_ab, A_q.expand_as(sum_ab), atol=1e-5)


# ---------------------------------------------------------------------------
# CI bounds & monotonicity
# ---------------------------------------------------------------------------


def test_T3_7_ci_bounds_in_unit_interval_and_ordered() -> None:
    """0 <= lower <= upper <= 1 everywhere."""
    torch.manual_seed(4)
    alpha = torch.rand(7, NUM_ANSWERS) * 5.0 + 1.1
    groups = question_index_groups()
    lower, upper = credible_interval(alpha, question_groups=groups, ci=0.95)
    assert (lower >= 0.0).all()
    assert (upper <= 1.0).all()
    assert (lower <= upper).all()


def test_T3_7_ci_widens_as_concentration_decreases() -> None:
    """A weaker (smaller-sum) alpha gives wider CIs than a stronger one."""
    groups = [("q", 0, 3)]
    # Same per-answer ratios, different total concentrations.
    weak_alpha = torch.tensor([[2.0, 3.0, 1.0]])  # sum = 6
    strong_alpha = weak_alpha * 20.0  # sum = 120

    lo_w, hi_w = credible_interval(weak_alpha, question_groups=groups, ci=0.95)
    lo_s, hi_s = credible_interval(strong_alpha, question_groups=groups, ci=0.95)

    weak_widths = (hi_w - lo_w)[0]
    strong_widths = (hi_s - lo_s)[0]
    assert (weak_widths > strong_widths).all(), (
        f"weak widths {weak_widths.tolist()} not all > strong widths {strong_widths.tolist()}"
    )


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def test_T3_7_coverage_with_observed_at_mean_is_one() -> None:
    """If every observed fraction equals the posterior mean, coverage = 1.0."""
    torch.manual_seed(5)
    alpha = torch.rand(20, NUM_ANSWERS) * 4.0 + 1.5
    groups = question_index_groups()
    # Observed fractions = posterior mean -> all inside any CI > 0.
    obs = posterior_mean(alpha, question_groups=groups)
    cov = coverage(alpha, obs, question_groups=groups, ci=0.95)
    assert cov == pytest.approx(1.0, abs=1e-6)


def test_T3_7_coverage_with_observed_outside_is_zero() -> None:
    """If every observed fraction is forced to 0 with a sharp alpha pointing
    away from 0, coverage drops to 0 (the CI excludes the observation)."""
    # Sharp alpha skewed toward answer 0 (e.g., [50, 1, 1] -> mean ≈ 0.96 / 0.02 / 0.02).
    alpha = torch.tensor([[50.0, 1.0, 1.0]] * 10)
    groups = [("q", 0, 3)]
    # Observe fractions skewed the other way: all on answer 2.
    obs = torch.tensor([[0.0, 0.0, 1.0]] * 10)
    cov = coverage(alpha, obs, question_groups=groups, ci=0.95)
    # Each row contributes 3 answers; with such a sharp alpha none of the
    # observed fractions land in the CIs (which are tight around the
    # alpha-implied means). Allow a small slack for the tail of answer 0
    # to accommodate observed=0 if the lower bound rounds down.
    assert cov < 0.10, f"sharp-mismatched coverage too high: {cov}"


def test_T3_7_coverage_calibrated_synthetic_approximates_target() -> None:
    """On data sampled FROM the predicted Dirichlet, coverage approaches ``ci``.

    Sample ``p_i ~ Dirichlet(alpha)`` for many galaxies, then compute the
    coverage of those true ``p_i`` under the CI derived from the same
    alpha. By construction this should land near the target ``ci``
    (modulo Monte-Carlo noise from the finite sample).
    """
    torch.manual_seed(6)
    n = 2000
    K = 5
    # Use a single mid-magnitude alpha for all galaxies so noise is smooth.
    alpha_row = torch.tensor([3.0, 5.0, 2.0, 4.0, 1.5])
    alpha = alpha_row.expand(n, K).contiguous()
    groups = [("q", 0, K)]
    # True fractions sampled from Dirichlet(alpha).
    obs = torch.zeros(n, K)
    for i in range(n):
        obs[i] = torch.distributions.Dirichlet(alpha[i]).sample()
    cov = coverage(alpha, obs, question_groups=groups, ci=0.95)
    # Per-answer marginals are Beta-correct, but coverage is pooled
    # across answers, so we expect ~95% but with some slop around it.
    assert 0.92 < cov < 0.98, f"calibrated coverage outside [0.92, 0.98]: {cov}"


def test_T3_7_coverage_with_valid_mask() -> None:
    """A per-question valid mask scopes both numerator and denominator."""
    torch.manual_seed(7)
    alpha = torch.rand(4, NUM_ANSWERS) * 3.0 + 1.5
    groups = question_index_groups()
    obs = posterior_mean(alpha, question_groups=groups)

    # All-valid mask -> coverage = 1.0 (obs=mean lands inside any CI > 0).
    valid_full = torch.ones(4, len(groups), dtype=torch.bool)
    cov_full = coverage(
        alpha, obs, question_groups=groups, ci=0.95, valid=valid_full
    )
    assert cov_full == pytest.approx(1.0, abs=1e-6)

    # All-invalid mask -> coverage = 0 (no valid pairs).
    valid_none = torch.zeros(4, len(groups), dtype=torch.bool)
    cov_none = coverage(
        alpha, obs, question_groups=groups, ci=0.95, valid=valid_none
    )
    assert cov_none == 0.0

    # Half-and-half: still 1.0 because every valid pair has obs == mean.
    valid_half = torch.zeros_like(valid_full)
    valid_half[:, :5] = True
    cov_half = coverage(
        alpha, obs, question_groups=groups, ci=0.95, valid=valid_half
    )
    assert cov_half == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_T3_7_invalid_ci_raises() -> None:
    """ci outside (0, 1) raises ValueError before any compute."""
    alpha = torch.rand(2, 3) + 1.1
    groups = [("q", 0, 3)]
    for bad_ci in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="ci must be in"):
            credible_interval(alpha, question_groups=groups, ci=bad_ci)
        with pytest.raises(ValueError, match="ci must be in"):
            coverage(
                alpha, alpha, question_groups=groups, ci=bad_ci
            )


def test_T3_7_observed_shape_mismatch_raises() -> None:
    alpha = torch.rand(3, 5) + 1.1
    obs_bad = torch.rand(3, 4) + 0.1
    with pytest.raises(ValueError, match="observed shape"):
        coverage(
            alpha, obs_bad, question_groups=[("q", 0, 5)], ci=0.95
        )


def test_T3_7_valid_shape_mismatch_raises() -> None:
    alpha = torch.rand(3, 5) + 1.1
    obs = torch.rand_like(alpha)
    with pytest.raises(ValueError, match="valid batch dim"):
        coverage(
            alpha, obs,
            question_groups=[("q", 0, 5)],
            valid=torch.ones(2, 1, dtype=torch.bool),
            ci=0.95,
        )
    with pytest.raises(ValueError, match="valid num_questions"):
        coverage(
            alpha, obs,
            question_groups=[("q", 0, 5)],
            valid=torch.ones(3, 2, dtype=torch.bool),
            ci=0.95,
        )


