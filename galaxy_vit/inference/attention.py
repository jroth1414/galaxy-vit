"""Saliency / attention overlays for trained Galaxy-ViT models (T1.6).

Two complementary techniques live here:

* :func:`attention_rollout` — Abnar & Zuidema 2020 attention-rollout,
  defined for transformer encoders that expose per-layer attention
  matrices (HF ``ViTForImageClassification`` does so via
  ``output_attentions=True``). Returns the patch-grid heatmap that, when
  upsampled to input resolution, shows where the CLS token "looks".
* :func:`gradcam` — Selvaraju et al. 2017 GradCAM, target-layer agnostic.
  Used for the Zoobot ConvNeXt-nano encoder where attention rollout
  doesn't apply (no multi-head attention). Works for any architecture
  with a 4-D feature map at some chosen layer.

:func:`overlay_heatmap_on_image` blends an arbitrary 2-D heatmap onto a
PIL image with a matplotlib colormap and returns a PIL RGB image — the
common output stage for both techniques.

Heavy deps (torch, matplotlib, PIL) are imported lazily inside functions
so the module is importable on any machine that has the base package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from PIL.Image import Image as PILImage
    from torch import Tensor

EPS = 1.0e-8


def attention_rollout(
    model: Any,
    pixel_values: Tensor,
) -> Tensor:
    """ViT attention-rollout heatmap (Abnar & Zuidema 2020).

    Computes the per-layer mean-over-heads attention with a residual
    connection (``A_l + I``, row-normalised), multiplies through all
    layers, and returns the CLS-token-to-patch-tokens row reshaped to
    the patch grid.

    Parameters
    ----------
    model:
        A HF transformer head that returns ``outputs.attentions`` when
        called with ``output_attentions=True``. **Requires the eager
        attention backend** — transformers 5.x defaults to scaled-dot-
        product attention (SDPA) which does not expose attention
        weights. Construct with ``attn_implementation="eager"`` (or set
        ``model.config._attn_implementation = "eager"``).
    pixel_values:
        ``(B, 3, H, W)`` float tensor in the model's expected
        normalisation.

    Returns
    -------
    ``(B, h, w)`` float tensor, where ``h * w == N - 1`` for ``N`` total
    tokens. For 224x224 ViT-B/16 (patch=16) the grid is ``(B, 14, 14)``.
    """
    import torch

    # Force eager attention so output_attentions actually fires; SDPA
    # (default in transformers 5.x) skips materialising the attention
    # matrix and returns an empty tuple instead.
    if hasattr(model, "config"):
        model.config._attn_implementation = "eager"

    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, output_attentions=True)
    attentions = outputs.attentions  # tuple of (B, heads, N, N)

    if not attentions:
        raise ValueError(
            "model did not produce attentions; construct the model with "
            "attn_implementation='eager' so output_attentions=True is honoured"
        )

    b, _, n, _ = attentions[0].shape
    device = attentions[0].device
    rollout = torch.eye(n, device=device).expand(b, n, n).clone()

    for attn in attentions:
        attn_mean = attn.mean(dim=1)  # (B, N, N) — average over heads
        with_residual = attn_mean + torch.eye(n, device=device)
        with_residual = with_residual / with_residual.sum(dim=-1, keepdim=True)
        rollout = with_residual @ rollout

    cls_to_patches = rollout[:, 0, 1:]  # (B, N-1)
    n_patches = cls_to_patches.shape[1]
    side = round(n_patches**0.5)
    if side * side != n_patches:
        raise ValueError(
            f"non-square patch grid: {n_patches} patches don't form a square"
        )

    grid = cls_to_patches.reshape(b, side, side)
    assert isinstance(grid, torch.Tensor)
    return grid


def gradcam(
    model: Any,
    pixel_values: Tensor,
    target_layer: Any,
    target_class: int | None = None,
) -> Tensor:
    """GradCAM heatmap (Selvaraju et al. 2017) at ``target_layer``.

    Forward through the model, backward on the target-class logit, then
    weight the captured activations by the channel-wise mean of their
    gradients, ReLU, and min-max normalise to ``[0, 1]``.

    Parameters
    ----------
    model:
        Any model whose forward returns a namespace with ``.logits``
        (matches our trainer convention for both ViT and Zoobot wrappers).
    pixel_values:
        ``(B, 3, H, W)`` input.
    target_layer:
        The ``nn.Module`` whose output activations + gradients we capture.
        For Zoobot (timm ConvNeXt-nano) this is typically
        ``model.encoder.stages[-1]``.
    target_class:
        Class index to backprop against. Defaults to the model's argmax
        prediction for batch element 0.

    Returns
    -------
    ``(B, h, w)`` float tensor in ``[0, 1]``, where ``(h, w)`` is the
    spatial resolution of the captured activation map (e.g. 7x7 for
    ConvNeXt-nano stage 3 on 224x224 input). Caller is expected to
    upsample to image resolution for overlay.
    """
    import torch
    import torch.nn.functional as F

    activations: list[Tensor] = []
    gradients: list[Tensor] = []

    def fwd_hook(_module: Any, _inp: Any, out: Tensor) -> None:
        activations.append(out)
        # Register a per-tensor grad hook on the output activation. This
        # avoids `register_full_backward_hook`'s "no inputs require
        # gradients" warning when target_layer is near the input edge.
        out.register_hook(  # type: ignore[no-untyped-call]
            lambda grad: gradients.append(grad)
        )

    fwd_handle = target_layer.register_forward_hook(fwd_hook)

    try:
        model.eval()
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # (B, num_classes)

        cls = (
            int(logits.argmax(dim=-1)[0].item())
            if target_class is None
            else int(target_class)
        )

        score = logits[:, cls].sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        if not activations or not gradients:
            raise RuntimeError(
                "GradCAM hooks did not fire; check that target_layer is on "
                "the forward path through model"
            )

        act = activations[0]  # (B, C, H, W)
        grad = gradients[0]  # (B, C, H, W)
        if act.ndim != 4 or grad.ndim != 4:
            raise ValueError(
                f"GradCAM expects 4-D activations/grads; got {act.shape} / {grad.shape}"
            )

        weights = grad.mean(dim=(-2, -1), keepdim=True)  # (B, C, 1, 1)
        cam = F.relu((weights * act).sum(dim=1))  # (B, H, W)
        cam_min = cam.amin(dim=(-2, -1), keepdim=True)
        cam_max = cam.amax(dim=(-2, -1), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + EPS)
    finally:
        fwd_handle.remove()

    detached = cam.detach()
    assert isinstance(detached, torch.Tensor)
    return detached


def per_question_gradcam(
    model: Any,
    pixel_values: Tensor,
    target_layer: Any,
    *,
    alpha_start: int,
    alpha_end: int,
) -> Tensor:
    """A-7: per-GZ-DESI-question GradCAM against the Dirichlet head.

    Drop-in variant of :func:`gradcam` that backpropagates from the
    sum of one question's alpha slice instead of a single class
    logit. Lets the caller produce a per-question heatmap that
    visualises "where the model is looking when answering this
    question."

    Why sum (not mean / softmax / argmax):
      * sum keeps the gradient signal proportional to the magnitude
        of the alpha slice, which dominates the Dirichlet posterior
        mean. The resulting heatmap is the most numerically stable
        per-question target the head exposes.
      * the alpha values are positive (softplus + alpha_floor) so
        the gradient w.r.t. each pixel is well-signed for the
        per-channel GradCAM weighting.

    Parameters
    ----------
    model:
        A model whose forward returns ``namespace.alpha`` of shape
        ``(B, num_answers)`` (the Dirichlet head wrapper).
    pixel_values:
        ``(B, 3, H, W)`` input.
    target_layer:
        Same kind of ``nn.Module`` you would pass to :func:`gradcam`
        for the Galaxy10 model -- the captured 4-D activation map.
        For Zoobot ConvNeXt-nano this is
        ``model.encoder.stages[-1]``.
    alpha_start, alpha_end:
        Half-open slice into ``alpha`` for the chosen question
        (e.g. ``(7, 10)`` for the bar question per
        :func:`galaxy_vit.data.schema.question_index_groups`).

    Returns
    -------
    ``(B, h, w)`` float tensor in ``[0, 1]`` -- the captured
    activation-map resolution. Caller upsamples to image resolution
    for blending.
    """
    import torch
    import torch.nn.functional as F

    if alpha_end <= alpha_start:
        raise ValueError(
            f"alpha slice must be non-empty; got [{alpha_start}, {alpha_end})"
        )

    activations: list[Tensor] = []
    gradients: list[Tensor] = []

    def fwd_hook(_module: Any, _inp: Any, out: Tensor) -> None:
        activations.append(out)
        out.register_hook(  # type: ignore[no-untyped-call]
            lambda grad: gradients.append(grad)
        )

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    try:
        model.eval()
        outputs = model(pixel_values=pixel_values)
        alpha = outputs.alpha  # (B, num_answers)
        if alpha.ndim != 2 or alpha.shape[1] < alpha_end:
            raise ValueError(
                f"unexpected alpha shape {tuple(alpha.shape)} for "
                f"slice [{alpha_start}, {alpha_end})"
            )
        score = alpha[:, alpha_start:alpha_end].sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        if not activations or not gradients:
            raise RuntimeError(
                "per_question_gradcam hooks did not fire; check that "
                "target_layer is on the forward path through model"
            )

        act = activations[0]
        grad = gradients[0]
        if act.ndim != 4 or grad.ndim != 4:
            raise ValueError(
                f"per_question_gradcam expects 4-D activations/grads; "
                f"got {act.shape} / {grad.shape}"
            )

        weights = grad.mean(dim=(-2, -1), keepdim=True)
        cam = F.relu((weights * act).sum(dim=1))
        cam_min = cam.amin(dim=(-2, -1), keepdim=True)
        cam_max = cam.amax(dim=(-2, -1), keepdim=True)
        cam = (cam - cam_min) / (cam_max - cam_min + EPS)
    finally:
        fwd_handle.remove()

    detached = cam.detach()
    assert isinstance(detached, torch.Tensor)
    return detached


def overlay_heatmap_on_image(
    image: PILImage,
    heatmap: Tensor,
    *,
    alpha: float = 0.45,
    colormap: str = "viridis",
) -> PILImage:
    """Blend a ``(h, w)`` (or ``(1, h, w)``) heatmap onto a PIL image.

    Heatmap is upsampled bilinearly to the image's resolution, mapped to
    RGB through a matplotlib colormap, and alpha-blended with the source
    image. Returns a fresh PIL RGB image.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch.nn.functional as F
    from PIL import Image as PILImage_

    img_arr = np.array(image.convert("RGB"))  # (H, W, 3)
    img_h, img_w = img_arr.shape[:2]

    hm = heatmap
    while hm.ndim < 4:
        hm = hm.unsqueeze(0)
    hm_resized = F.interpolate(
        hm.float(), size=(img_h, img_w), mode="bilinear", align_corners=False
    )
    hm_arr = hm_resized.squeeze().cpu().numpy()

    cmap = plt.get_cmap(colormap)
    hm_rgb = (cmap(hm_arr)[..., :3] * 255).astype(np.uint8)

    blended = ((1 - alpha) * img_arr + alpha * hm_rgb).astype(np.uint8)
    out = PILImage_.fromarray(blended)
    assert isinstance(out, PILImage_.Image)
    return out
