"""Unit tests for galaxy_vit.inference.attention.per_question_gradcam (A-7).

Toy-network tests over a tiny ConvNet so the assertion shape (in [0, 1],
non-zero variance, sliced gradient) is validated without depending on
the real Zoobot checkpoint.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from galaxy_vit.inference.attention import per_question_gradcam  # noqa: E402


def _build_toy_model() -> tuple[torch.nn.Module, torch.nn.Module]:
    """Tiny Conv -> avg-pool -> Linear head; returns (model, target_layer).

    The head produces a (B, 6) "alpha" tensor exposed as
    ``namespace.alpha`` so per_question_gradcam can slice it.
    """
    import torch.nn as nn

    class _Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.linear = nn.Linear(4, 6)

        def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
            f = self.relu(self.conv(pixel_values))
            pooled = f.mean(dim=(-2, -1))
            alpha = torch.nn.functional.softplus(self.linear(pooled)) + 1.0
            return SimpleNamespace(alpha=alpha)

    model = _Toy()
    return model, model.relu  # type: ignore[return-value]


def test_per_question_gradcam_returns_finite_values_in_unit_range() -> None:
    """Heatmap entries land in [0, 1] and contain no NaN/inf."""
    torch.manual_seed(0)
    model, target = _build_toy_model()
    pixel = torch.randn(1, 3, 16, 16)
    cam = per_question_gradcam(
        model, pixel, target, alpha_start=0, alpha_end=3
    )
    assert cam.shape == (1, 16, 16)
    assert torch.isfinite(cam).all().item()
    assert (cam >= 0.0).all().item()
    assert (cam <= 1.0 + 1e-6).all().item()


def test_per_question_gradcam_has_nontrivial_variance() -> None:
    """Real GradCAMs aren't uniform; variance should be noticeable."""
    torch.manual_seed(1)
    model, target = _build_toy_model()
    pixel = torch.randn(2, 3, 16, 16)
    cam = per_question_gradcam(
        model, pixel, target, alpha_start=2, alpha_end=5
    )
    # min-max normalisation guarantees range = 1; std > 0.05 indicates
    # at least a real spatial signal.
    assert float(cam.std().item()) > 0.05


def test_per_question_gradcam_differs_across_questions() -> None:
    """Two different alpha slices produce different heatmaps.

    Toy-net qualitative check: backprop from different alpha indices
    -> different per-channel weights -> distinct CAMs.
    """
    torch.manual_seed(2)
    model, target = _build_toy_model()
    pixel = torch.randn(1, 3, 16, 16)
    cam_a = per_question_gradcam(
        model, pixel, target, alpha_start=0, alpha_end=2
    )
    cam_b = per_question_gradcam(
        model, pixel, target, alpha_start=4, alpha_end=6
    )
    # Heatmaps shouldn't be identical -- different gradient targets.
    diff = float((cam_a - cam_b).abs().mean().item())
    assert diff > 1e-3, (
        f"per-question GradCAMs are identical (mean abs diff = {diff}); "
        "back-prop targets aren't being honoured"
    )


def test_per_question_gradcam_rejects_empty_slice() -> None:
    """alpha_start >= alpha_end should raise ValueError."""
    model, target = _build_toy_model()
    pixel = torch.randn(1, 3, 8, 8)
    with pytest.raises(ValueError, match="non-empty"):
        per_question_gradcam(
            model, pixel, target, alpha_start=2, alpha_end=2
        )


def test_per_question_gradcam_rejects_out_of_bounds_slice() -> None:
    """alpha_end larger than the model's alpha dim should raise."""
    model, target = _build_toy_model()
    pixel = torch.randn(1, 3, 8, 8)
    with pytest.raises(ValueError, match="alpha shape"):
        per_question_gradcam(
            model, pixel, target, alpha_start=0, alpha_end=99
        )
