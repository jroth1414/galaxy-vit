"""T3.3 — DirichletMultinomialHead unit tests.

DEVPLAN T3.3 acceptance:

* ``test_forward_positive_alpha`` — every output alpha is at least the
  configured ``alpha_floor`` (default 1.0), regardless of input sign /
  magnitude. (Equality is reachable when ``softplus`` underflows to
  zero on extremely negative inputs; that is the documented contract
  of ``softplus + alpha_floor``.)
* ``test_backward_finite_grad`` — gradient flows through softplus and
  the linear layer without producing NaN / Inf.

Plus shape sanity, alpha_floor parameterization, parameter-count sanity,
and the wiring contract that ``build_zoobot_dirichlet`` exposes
``.alpha`` on the forward output.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from galaxy_vit.models.dirichlet_head import (  # noqa: E402
    DEFAULT_ALPHA_FLOOR,
    NUM_DR8_ANSWERS,
    build_dirichlet_head,
)

BATCH_SIZE = 8
FEATURE_DIM = 640  # ConvNeXt-nano penultimate dim


def test_T3_3_forward_output_shape() -> None:
    """Output shape is (B, num_answers) for an arbitrary batch."""
    head = build_dirichlet_head(FEATURE_DIM)
    x = torch.randn(BATCH_SIZE, FEATURE_DIM)
    alpha = head(x)
    assert alpha.shape == (BATCH_SIZE, NUM_DR8_ANSWERS)


def test_T3_3_forward_positive_alpha() -> None:
    """Every output alpha is >= alpha_floor for any input value (DEVPLAN accept).

    Strict inequality fails by design when softplus underflows on
    extremely negative inputs (softplus(-1e3) -> 0 exactly), so the
    contract is alpha >= alpha_floor; alpha > 0 always holds because
    alpha_floor itself must be > 0 (validated at construction).
    """
    head = build_dirichlet_head(FEATURE_DIM)
    # Mix of positive, negative, zero, and extreme inputs to exercise
    # softplus's full domain.
    x = torch.cat(
        [
            torch.randn(BATCH_SIZE, FEATURE_DIM),
            torch.full((BATCH_SIZE, FEATURE_DIM), -1e3),  # softplus(-1000) ≈ 0
            torch.full((BATCH_SIZE, FEATURE_DIM), 1e3),   # softplus(1000)  ≈ 1000
            torch.zeros(BATCH_SIZE, FEATURE_DIM),          # softplus(0)    = log(2)
        ],
        dim=0,
    )
    alpha = head(x)
    assert torch.isfinite(alpha).all(), "non-finite alpha somewhere"
    assert (alpha >= DEFAULT_ALPHA_FLOOR).all(), (
        f"alpha < {DEFAULT_ALPHA_FLOOR}; min={alpha.min().item()}"
    )
    # Strict positivity always holds (alpha_floor > 0 is enforced at construction).
    assert (alpha > 0).all()


def test_T3_3_alpha_floor_respected() -> None:
    """When softplus underflows to ~0, alpha collapses to the chosen alpha_floor.

    Pin the linear layer to all zeros so the pre-activation is a fixed
    deterministic 0 regardless of input -> softplus(0) = ln(2) ~= 0.693
    -> alpha = 0.693 + alpha_floor. (Random-init weights times -1e6
    inputs blow up the pre-activation in both signs and yield alpha values
    in the millions on the positive side, which masks the floor we are
    trying to test.)
    """
    custom_floor = 0.5
    head = build_dirichlet_head(FEATURE_DIM, alpha_floor=custom_floor)
    with torch.no_grad():
        head.linear.weight.zero_()  # type: ignore[attr-defined]
        head.linear.bias.zero_()    # type: ignore[attr-defined]
    x = torch.randn(BATCH_SIZE, FEATURE_DIM)
    alpha = head(x)
    expected = math.log(2.0) + custom_floor
    assert torch.allclose(alpha, torch.full_like(alpha, expected), atol=1e-6)
    assert (alpha >= custom_floor).all()


def test_T3_3_backward_finite_grad() -> None:
    """Backward pass produces finite, non-zero gradients on encoder + head params."""
    head = build_dirichlet_head(FEATURE_DIM)
    x = torch.randn(BATCH_SIZE, FEATURE_DIM, requires_grad=True)
    alpha = head(x)
    loss = alpha.sum()
    loss.backward()
    # Input gradient
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0
    # Linear weight + bias gradients
    for name, p in head.named_parameters():
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite grad"
        assert p.grad.abs().sum() > 0, f"{name} grad is identically zero"


def test_T3_3_backward_finite_grad_extreme_inputs() -> None:
    """Softplus is numerically stable for very large +/- inputs (no NaN in backward)."""
    head = build_dirichlet_head(FEATURE_DIM)
    # Mix extremes that historically tickle naive softplus implementations.
    x = torch.cat(
        [
            torch.full((BATCH_SIZE, FEATURE_DIM), 50.0),
            torch.full((BATCH_SIZE, FEATURE_DIM), -50.0),
            torch.full((BATCH_SIZE, FEATURE_DIM), 1e2),
            torch.full((BATCH_SIZE, FEATURE_DIM), -1e2),
        ],
        dim=0,
    ).requires_grad_(True)
    alpha = head(x)
    loss = alpha.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), "non-finite grad on extreme input"
    for name, p in head.named_parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_T3_3_parameter_count() -> None:
    """Head has exactly Linear(in, num_answers) parameters: in*num_answers + num_answers."""
    head = build_dirichlet_head(FEATURE_DIM)
    n_params = sum(p.numel() for p in head.parameters())
    expected = FEATURE_DIM * NUM_DR8_ANSWERS + NUM_DR8_ANSWERS
    assert n_params == expected, f"unexpected param count: got {n_params}, want {expected}"


def test_T3_3_per_question_slices_match_schema() -> None:
    """Output slices line up with schema.question_index_groups (T3.4 loss depends on this)."""
    pytest.importorskip("galaxy_datasets")
    from galaxy_vit.data import schema

    head = build_dirichlet_head(FEATURE_DIM)
    x = torch.randn(BATCH_SIZE, FEATURE_DIM)
    alpha = head(x)
    groups = schema.question_index_groups()
    # Every slice is non-empty and produces a strictly-positive concentration
    # vector, ready for the T3.4 Dirichlet-Multinomial NLL.
    for q_name, start, end in groups:
        slice_alpha = alpha[:, start:end]
        assert slice_alpha.shape[1] >= 2, (
            f"degenerate per-question slice for {q_name!r}: width={slice_alpha.shape[1]}"
        )
        assert (slice_alpha >= DEFAULT_ALPHA_FLOOR).all()
    # Slices cover the entire output (no gaps, no overlap).
    assert groups[-1][2] == NUM_DR8_ANSWERS


def test_T3_3_in_features_validation() -> None:
    """Bad in_features raises before we waste time materializing tensors."""
    with pytest.raises(ValueError, match="in_features must be positive"):
        build_dirichlet_head(0)
    with pytest.raises(ValueError, match="in_features must be positive"):
        build_dirichlet_head(-5)


def test_T3_3_alpha_floor_validation() -> None:
    """alpha_floor <= 0 is rejected (would let alpha collapse to 0 / negative)."""
    with pytest.raises(ValueError, match="alpha_floor must be"):
        build_dirichlet_head(FEATURE_DIM, alpha_floor=0.0)
    with pytest.raises(ValueError, match="alpha_floor must be"):
        build_dirichlet_head(FEATURE_DIM, alpha_floor=-1.0)


def test_T3_3_softplus_at_zero_input_is_log_two_plus_floor() -> None:
    """Sanity check the math: softplus(0) = ln(2), so alpha = ln(2) + alpha_floor."""
    head = build_dirichlet_head(FEATURE_DIM)
    # Pin the linear layer to zero so softplus operates on identically-zero pre-activations.
    with torch.no_grad():
        head.linear.weight.zero_()  # type: ignore[attr-defined]
        head.linear.bias.zero_()    # type: ignore[attr-defined]
    x = torch.randn(BATCH_SIZE, FEATURE_DIM)
    alpha = head(x)
    expected = math.log(2.0) + DEFAULT_ALPHA_FLOOR
    assert torch.allclose(alpha, torch.full_like(alpha, expected), atol=1e-6)
