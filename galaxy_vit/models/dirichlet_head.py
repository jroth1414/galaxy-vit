"""Dirichlet-Multinomial concentration head (T3.3).

Replaces the T2.3 plurality-cross-entropy ``nn.Linear`` head with a
concentration-parameter head that outputs a strictly-positive 34-dim
vector ``alpha`` per galaxy. Each per-question slice
``alpha[..., start:end]`` (per :func:`galaxy_vit.data.schema.question_index_groups`)
is the concentration of a Dirichlet distribution whose Dirichlet-Multinomial
marginal is the volunteer-vote likelihood the T3.4 loss will optimize.

Output activation per DEVPLAN T3.3::

    alpha = softplus(linear(x)) + alpha_floor      # alpha_floor = 1.0

The ``+ alpha_floor`` term keeps every concentration at least 1
(equality is reached only when softplus underflows on extremely
negative pre-activations), which:

* Avoids the numerical edge of ``log Gamma(alpha)`` near alpha = 0.
* Keeps the per-question Dirichlet posterior unimodal (alpha ≥ 1 across the
  simplex), simplifying T3.7's analytic Beta-marginal CIs.

The head exposes the concentration as ``model(pixel_values=x).alpha`` so
the T3.6 trainer's loss + the T3.7 posterior module both see the same
canonical attribute.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only
    import torch.nn as nn

DEFAULT_ALPHA_FLOOR = 1.0
NUM_DR8_ANSWERS = 34


def build_dirichlet_head(
    in_features: int,
    *,
    num_answers: int = NUM_DR8_ANSWERS,
    alpha_floor: float = DEFAULT_ALPHA_FLOOR,
) -> nn.Module:
    """Construct a fresh ``DirichletMultinomialHead`` ready for training.

    Lazy-imports torch so the package stays importable without the
    ``[torch-*]`` extra. Returns the head module; caller wires it onto
    the encoder of choice.
    """
    import torch.nn as nn_runtime
    import torch.nn.functional as F_runtime

    if in_features <= 0:
        raise ValueError(f"in_features must be positive, got {in_features}")
    if num_answers <= 0:
        raise ValueError(f"num_answers must be positive, got {num_answers}")
    if alpha_floor <= 0:
        raise ValueError(f"alpha_floor must be > 0, got {alpha_floor}")

    class DirichletMultinomialHead(nn_runtime.Module):
        """Concentration head: Linear projection then softplus + alpha_floor."""

        def __init__(self) -> None:
            super().__init__()
            self.linear = nn_runtime.Linear(in_features, num_answers)
            self.alpha_floor = alpha_floor
            self.in_features = in_features
            self.num_answers = num_answers

        def forward(self, x: object) -> object:
            raw = self.linear(x)
            return F_runtime.softplus(raw) + self.alpha_floor

    head = DirichletMultinomialHead()
    assert isinstance(head, nn_runtime.Module)
    return head


def build_zoobot_dirichlet(
    *,
    num_answers: int = NUM_DR8_ANSWERS,
    alpha_floor: float = DEFAULT_ALPHA_FLOOR,
    encoder_id: str | None = None,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    """Construct a Zoobot ConvNeXt-nano encoder + DirichletMultinomialHead.

    Mirrors :func:`galaxy_vit.models.zoobot_encoder.build_zoobot_finetune`
    but swaps the plurality-cross-entropy head for the Dirichlet
    concentration head. Forward output exposes ``.alpha`` (canonical) so
    downstream code (T3.4 loss, T3.7 posterior) can read the
    concentration directly.

    Returns
    -------
    ``(model, encoder, head)`` — same triplet shape T2.3's trainer
    expected, so the param-group / freeze logic carries over unchanged.
    """
    import timm
    import torch.nn as nn_runtime

    from galaxy_vit.models.zoobot_encoder import ZOOBOT_HF_ID

    encoder = timm.create_model(
        f"hf-hub:{encoder_id or ZOOBOT_HF_ID}",
        pretrained=True,
        num_classes=0,
    )
    num_features = getattr(encoder, "num_features", None)
    if not isinstance(num_features, int):
        raise RuntimeError(
            f"timm encoder did not expose an int num_features; got {type(num_features)}"
        )
    head = build_dirichlet_head(
        num_features, num_answers=num_answers, alpha_floor=alpha_floor
    )

    class _ZoobotDirichlet(nn_runtime.Module):
        def __init__(
            self, encoder_mod: nn_runtime.Module, head_mod: nn_runtime.Module
        ) -> None:
            super().__init__()
            self.encoder = encoder_mod
            self.head = head_mod

        def forward(self, pixel_values: object) -> SimpleNamespace:
            feats = self.encoder(pixel_values)
            alpha = self.head(feats)
            return SimpleNamespace(alpha=alpha)

    model = _ZoobotDirichlet(encoder, head)
    assert isinstance(model, nn_runtime.Module)
    assert isinstance(encoder, nn_runtime.Module)
    assert isinstance(head, nn_runtime.Module)
    return model, encoder, head
