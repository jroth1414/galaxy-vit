"""Zoobot ConvNeXt-nano encoder loader (T1.5).

Loads the galaxy-pretrained ConvNeXt-nano encoder from HF Hub via timm,
strips its original classifier (timm ``num_classes=0`` returns features),
and attaches a fresh ``nn.Linear`` head sized to ``num_classes``.

The wrapper exposes a ViT-baseline-compatible forward signature
(``model(pixel_values=x).logits``) so the trainer can call both model
families uniformly. Returns the model alongside named handles to the
encoder and head modules so the trainer can build differential-LR param
groups and freeze the encoder during the head-only finetuning stage.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only
    import torch.nn as nn

ZOOBOT_HF_ID = "mwalmsley/zoobot-encoder-convnext_nano"


def build_zoobot_finetune(
    num_classes: int = 10,
    *,
    encoder_id: str = ZOOBOT_HF_ID,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Construct a Zoobot ConvNeXt-nano encoder + fresh classifier head.

    Returns
    -------
    ``(model, encoder_module, head_module)`` — the trainer hands the
    encoder/head modules to the optimizer for differential-LR groups
    and to the freeze/unfreeze logic for the two-stage finetune.
    """
    import timm
    import torch.nn as nn_runtime

    # Inner-class definition keeps torch/timm imports lazy (so the package
    # remains importable on CI without [m1-train] / [torch-cu128]) while
    # still satisfying nn.Module subclassing.
    class _ZoobotFinetune(nn_runtime.Module):
        def __init__(self, encoder: nn_runtime.Module, head: nn_runtime.Module) -> None:
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, pixel_values: object) -> SimpleNamespace:
            feats = self.encoder(pixel_values)
            logits = self.head(feats)
            return SimpleNamespace(logits=logits)

    # num_classes=0 makes timm strip the original classifier and return raw
    # features (B, num_features) on call.
    encoder = timm.create_model(
        f"hf-hub:{encoder_id}",
        pretrained=True,
        num_classes=0,
    )
    # encoder.num_features is timm's documented int attribute; nn.Module's
    # generic __getattr__ types it as Tensor|Module, so narrow explicitly.
    num_features = getattr(encoder, "num_features", None)
    if not isinstance(num_features, int):
        raise RuntimeError(
            f"timm encoder did not expose an int num_features; got {type(num_features)}"
        )
    head_dim = num_features
    head = nn_runtime.Linear(head_dim, num_classes)
    model = _ZoobotFinetune(encoder, head)

    assert isinstance(model, nn_runtime.Module)
    assert isinstance(encoder, nn_runtime.Module)
    assert isinstance(head, nn_runtime.Module)
    return model, encoder, head
