"""T1.6 — interpretability acceptance tests.

Hermetic — uses fresh-init models (no HF Hub download) so the tests run
locally and would also run in any CI that has torch + transformers
installed (currently CI runs [dev] alone so the whole module skips).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from galaxy_vit.inference.attention import attention_rollout, gradcam  # noqa: E402


def test_attention_shape() -> None:
    """T1.6 acceptance: ViT attention rollout yields (B, 14, 14) on 224x224 input.

    Uses a freshly-initialised ViTConfig (no HF Hub download) so the test
    is hermetic; we only check shape, not semantics.
    """
    from transformers import ViTConfig, ViTForImageClassification

    config = ViTConfig(image_size=224, num_labels=10)
    model = ViTForImageClassification(config)

    x = torch.randn(2, 3, 224, 224)
    rollout = attention_rollout(model, x)

    # 224 / patch_size 16 = 14
    assert rollout.shape == (2, 14, 14), f"got shape {tuple(rollout.shape)}"
    # rollout values come from a softmax-then-product chain so are non-negative.
    assert (rollout >= 0).all()


def test_gradcam_nonzero() -> None:
    """T1.6 acceptance: GradCAM produces a heatmap with at least one nonzero value.

    Uses a tiny convnet (no HF dep) so the test is fast and self-contained.
    The wrapper mimics our model wrappers' .logits forward signature.
    """
    from types import SimpleNamespace

    from torch import nn

    class _TinyClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.head = nn.Linear(16, 10)

        def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
            feats = self.relu(self.conv(pixel_values))
            pooled = self.pool(feats).flatten(1)
            return SimpleNamespace(logits=self.head(pooled))

    model = _TinyClassifier()
    x = torch.randn(1, 3, 32, 32)

    cam = gradcam(model, x, target_layer=model.conv)

    assert cam.shape == (1, 32, 32), f"got shape {tuple(cam.shape)}"
    assert (cam >= 0).all(), "GradCAM heatmap values must be non-negative (ReLU'd)"
    assert (cam <= 1).all(), "GradCAM heatmap should be normalised to [0, 1]"
    assert float(cam.max().item()) > 0.0, "GradCAM heatmap is all zeros"
