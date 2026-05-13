"""Unit tests for galaxy_vit.inference.outliers (S-3).

Toy-tensor tests pinned against analytically tractable cases. No
checkpoint, no parquet, no shards. Skipped on the [dev]-only CI install
when torch is missing.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from galaxy_vit.inference.outliers import (  # noqa: E402
    OutlierEntry,
    bald_total,
    predictive_entropy_total,
    topk_indices,
    volunteer_disagreement,
)

# Two-question toy schema: 3 + 2 answers per question.
TOY_GROUPS: list[tuple[str, int, int]] = [
    ("q0_3answers", 0, 3),
    ("q1_2answers", 3, 5),
]


def test_predictive_entropy_uniform_dirichlet_is_max() -> None:
    """Uniform Dirichlet (alpha=1 everywhere) -> predictive mean is uniform.

    Per question, H = log(K). q0 has K=3 -> ln 3; q1 has K=2 -> ln 2.
    Total = ln 3 + ln 2 = ln 6.
    """
    alpha = torch.ones(1, 5, dtype=torch.float32)
    h = predictive_entropy_total(alpha, question_groups=TOY_GROUPS)
    assert h.shape == (1,)
    assert float(h[0].item()) == pytest.approx(math.log(6), abs=1e-5)


def test_predictive_entropy_one_hot_is_zero() -> None:
    """Concentrated alpha (one answer dominates) -> predictive entropy ~= 0."""
    alpha = torch.tensor([[1000.0, 1.0, 1.0, 1000.0, 1.0]], dtype=torch.float32)
    h = predictive_entropy_total(alpha, question_groups=TOY_GROUPS)
    # Both questions are essentially one-hot; sum-of-entropies ~= 0.
    assert float(h[0].item()) < 0.05


def test_bald_is_nonnegative_for_arbitrary_alpha() -> None:
    """BALD >= 0 by construction (mutual information is non-negative)."""
    torch.manual_seed(0)
    alpha = 1.0 + torch.rand(50, 5) * 5.0  # alpha in [1, 6]
    b = bald_total(alpha, question_groups=TOY_GROUPS)
    assert (b >= -1e-6).all().item()


def test_bald_le_predictive_entropy() -> None:
    """BALD = H[p_pred] - E[H[p|alpha]] <= H[p_pred]."""
    torch.manual_seed(1)
    alpha = 1.0 + torch.rand(50, 5) * 5.0
    h = predictive_entropy_total(alpha, question_groups=TOY_GROUPS)
    b = bald_total(alpha, question_groups=TOY_GROUPS)
    assert (b <= h + 1e-6).all().item()


def test_bald_collapses_to_zero_at_high_alpha() -> None:
    """As alpha -> infinity the Dirichlet collapses to a point -> BALD -> 0."""
    alpha = torch.full((1, 5), 1e6, dtype=torch.float32)
    b = bald_total(alpha, question_groups=TOY_GROUPS)
    assert float(b[0].item()) == pytest.approx(0.0, abs=1e-3)


def test_volunteer_disagreement_zero_when_predictions_match_votes() -> None:
    """L1 disagreement is 0 when expected_fractions(alpha) matches volunteer_fracs."""
    # q0: alpha = (3, 1, 1) -> mean = (0.6, 0.2, 0.2)
    # q1: alpha = (4, 4)    -> mean = (0.5, 0.5)
    alpha = torch.tensor([[3.0, 1.0, 1.0, 4.0, 4.0]], dtype=torch.float32)
    fracs = torch.tensor(
        [[0.6, 0.2, 0.2, 0.5, 0.5]], dtype=torch.float32
    )
    valid = torch.ones(1, 2, dtype=torch.bool)
    d = volunteer_disagreement(
        alpha, fracs, valid, question_groups=TOY_GROUPS
    )
    assert d.shape == (1,)
    assert float(d[0].item()) == pytest.approx(0.0, abs=1e-6)


def test_volunteer_disagreement_max_for_one_hot_disagreement() -> None:
    """Anti-aligned one-hot pred vs vote -> per-question L1 = 2.0; mean = 2.0."""
    alpha = torch.tensor([[1000.0, 1.0, 1.0, 1000.0, 1.0]], dtype=torch.float32)
    fracs = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0, 1.0]], dtype=torch.float32
    )
    valid = torch.ones(1, 2, dtype=torch.bool)
    d = volunteer_disagreement(
        alpha, fracs, valid, question_groups=TOY_GROUPS
    )
    # Both questions: pred ~ (1, 0, 0)/(1, 0); volunteer = (0, 1, 0)/(0, 1).
    # Per-question L1 = |1-0| + |0-1| + |0-0| = 2.0; mean across 2 valid Qs = 2.0.
    assert float(d[0].item()) == pytest.approx(2.0, abs=1e-2)


def test_volunteer_disagreement_skips_invalid_questions() -> None:
    """Invalid questions don't contribute; mean is over valid count only."""
    alpha = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    # q0 disagreement L1 = 2.0; q1 invalid (zero votes) and masked off.
    fracs = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    valid = torch.tensor([[True, False]], dtype=torch.bool)
    d = volunteer_disagreement(
        alpha, fracs, valid, question_groups=TOY_GROUPS
    )
    # pred q0 = (1/3, 1/3, 1/3); L1 = |1/3-0|+|1/3-1|+|1/3-0| = 4/3
    # mean over 1 valid Q = 4/3.
    assert float(d[0].item()) == pytest.approx(4 / 3, abs=1e-5)


def test_volunteer_disagreement_zero_for_all_invalid() -> None:
    """Galaxy with all questions invalid -> denominator clamped, returns 0."""
    alpha = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    fracs = torch.zeros(1, 5, dtype=torch.float32)
    valid = torch.zeros(1, 2, dtype=torch.bool)
    d = volunteer_disagreement(
        alpha, fracs, valid, question_groups=TOY_GROUPS
    )
    assert float(d[0].item()) == pytest.approx(0.0, abs=1e-9)


def test_topk_indices_returns_descending() -> None:
    scores = torch.tensor([0.1, 0.5, 0.3, 0.9, 0.2], dtype=torch.float32)
    out = topk_indices(scores, k=3)
    assert isinstance(out[0], OutlierEntry)
    assert [e.idx for e in out] == [3, 1, 2]
    assert out[0].value == pytest.approx(0.9)
    assert out[-1].value == pytest.approx(0.3)


def test_topk_indices_clamps_k() -> None:
    scores = torch.tensor([1.0, 2.0, 3.0])
    out = topk_indices(scores, k=999)
    assert len(out) == 3


def test_topk_indices_is_deterministic() -> None:
    torch.manual_seed(0)
    scores = torch.rand(100)
    a = topk_indices(scores, k=10)
    b = topk_indices(scores, k=10)
    assert [(e.idx, e.value) for e in a] == [(e.idx, e.value) for e in b]


def test_disagreement_validation_rejects_shape_mismatch() -> None:
    alpha = torch.zeros(2, 5)
    fracs = torch.zeros(2, 4)
    valid = torch.zeros(2, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="shape"):
        volunteer_disagreement(
            alpha, fracs, valid, question_groups=TOY_GROUPS
        )


def test_disagreement_validation_rejects_batch_mismatch() -> None:
    alpha = torch.zeros(3, 5)
    fracs = torch.zeros(3, 5)
    valid = torch.zeros(2, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="batch"):
        volunteer_disagreement(
            alpha, fracs, valid, question_groups=TOY_GROUPS
        )


def test_topk_validation_rejects_zero_k() -> None:
    with pytest.raises(ValueError, match="k must be"):
        topk_indices(torch.zeros(5), k=0)


def test_topk_validation_rejects_2d_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        topk_indices(torch.zeros(2, 3), k=1)
