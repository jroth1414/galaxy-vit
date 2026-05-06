"""Single-image inference helpers for the FastAPI server (T1.7).

:class:`GalaxyClassifier` loads a trained checkpoint, applies the eval
transform pipeline (T1.3 + T1.2 normalisation), and provides:

* :meth:`predict` — top-k softmax probabilities for an input PIL image.
* :meth:`gradcam_overlay` — PIL image with the GradCAM heatmap blended
  via the T1.6 :func:`overlay_heatmap_on_image`.

:func:`resolve_target_layer` selects the right GradCAM target module for
the configured encoder kind (``zoobot_convnext`` -> final ConvNeXt
stage; ``vit_baseline`` -> last transformer layer). Shared by the
serve endpoint and ``scripts/render_interpretability.py``.

The classifier is built once at FastAPI lifespan startup and held in
module-global state. CPU inference is the target for HF Spaces; the
class accepts a device override for local GPU benchmarking.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from galaxy_vit.data.transforms import build_eval_transform, load_normalization
from galaxy_vit.inference.attention import gradcam, overlay_heatmap_on_image
from galaxy_vit.training.trainer import TrainerConfig, build_model_and_split

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from PIL.Image import Image as PILImage


def resolve_target_layer(encoder: torch.nn.Module, kind: str) -> torch.nn.Module:
    """Return the GradCAM target module for ``encoder`` of the given kind."""
    if kind == "zoobot_convnext":
        stages: Any = getattr(encoder, "stages", None)
        if stages is None:
            raise RuntimeError(
                "Zoobot encoder doesn't expose `.stages`; timm ConvNeXt API may have changed"
            )
        layer = stages[-1]
    elif kind == "vit_baseline":
        vit_encoder: Any = getattr(encoder, "encoder", None)
        if vit_encoder is None:
            raise RuntimeError("ViT model doesn't expose `.encoder`; unexpected layout")
        layer = vit_encoder.layer[-1]
    else:
        raise ValueError(f"unknown model kind for GradCAM target: {kind!r}")
    assert isinstance(layer, torch.nn.Module)
    return layer


class GalaxyClassifier:
    """Encapsulates a trained checkpoint + eval transform + GradCAM target."""

    def __init__(self, ckpt_path: Path, *, device: str = "cpu") -> None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        cfg = TrainerConfig.model_validate(ckpt["config"])

        model, encoder, _head = build_model_and_split(cfg.model)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()

        mean, std = load_normalization(cfg.data.normalization)
        transform = build_eval_transform(
            image_size=cfg.data.image_size, mean=mean, std=std
        )

        target_layer = resolve_target_layer(encoder, cfg.model.kind)

        self.cfg = cfg
        self.model = model
        self.encoder = encoder
        self.transform = transform
        self.target_layer = target_layer
        self.device = device

    @torch.no_grad()
    def predict(
        self, image: PILImage, *, top_k: int = 3
    ) -> list[tuple[int, float]]:
        """Return ``[(class_id, probability), ...]`` sorted by descending prob."""
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(pixel_values=x).logits  # (1, num_classes)
        probs = F.softmax(logits, dim=-1)[0]
        topv, topi = torch.topk(probs, k=min(top_k, probs.numel()))
        return [(int(c.item()), float(p.item())) for c, p in zip(topi, topv, strict=True)]

    def gradcam_overlay(self, image: PILImage) -> PILImage:
        """Return the PIL image with a GradCAM heatmap overlay."""
        x = self.transform(image).unsqueeze(0).to(self.device)
        cam = gradcam(self.model, x, target_layer=self.target_layer)[0]
        return overlay_heatmap_on_image(image, cam)
