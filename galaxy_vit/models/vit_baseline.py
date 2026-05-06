"""ViT-B/16 baseline model factory.

T1.4 lives here. Loads ``google/vit-base-patch16-224`` from the HF Hub via
``transformers.ViTForImageClassification`` and swaps the 1000-class
ImageNet head for a fresh `num_classes`-way classifier.

Heavy deps (torch / transformers) are imported lazily inside the factory so
the module is importable on any machine that has the base package
installed; the ``[torch-cu128]`` + ``[m1-train]`` extras are required to
actually instantiate the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — typing-only
    import torch.nn as nn

VIT_BASELINE_HF_ID = "google/vit-base-patch16-224"


def build_vit_baseline(
    num_classes: int = 10,
    *,
    encoder_id: str = VIT_BASELINE_HF_ID,
) -> nn.Module:
    """Construct ``ViTForImageClassification`` with a freshly initialised head.

    The pretrained 1000-way head is discarded
    (``ignore_mismatched_sizes=True``) and replaced with a `num_classes`-way
    ``nn.Linear``. Encoder weights are kept at their ImageNet pretraining.
    Use ``model(pixel_values=x).logits`` to obtain raw class scores.
    """
    import torch.nn as nn
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(
        encoder_id,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    assert isinstance(model, nn.Module)
    return model


def split_param_groups(
    model: nn.Module,
    *,
    encoder_lr: float,
    head_lr: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Split params into encoder vs classifier head groups for differential LRs.

    The HF ``ViTForImageClassification`` exposes the head as ``model.classifier``
    and the encoder as ``model.vit``. Returns a list shaped for
    ``torch.optim.AdamW(param_groups, ...)``.
    """
    import torch.nn as nn

    # nn.Module's __getattr__ is typed `Tensor | Module`; narrow to Module.
    encoder = model.vit
    head = model.classifier
    assert isinstance(encoder, nn.Module)
    assert isinstance(head, nn.Module)

    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    head_params = [p for p in head.parameters() if p.requires_grad]
    return [
        {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
    ]
