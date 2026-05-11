"""Single-image Dirichlet posterior predictor for the Posteriors tab (T4.3).

Loads a T3.6 Zoobot+Dirichlet checkpoint, optionally applies the
post-hoc temperature calibration produced by
``scripts/calibrate_dirichlet.py``, and computes a per-question
posterior summary for an arbitrary PIL image:

* posterior_mean[i] = alpha_i / sum(alpha_q) per answer
* lower / upper = Beta(alpha_i, A_q - alpha_i) 95% CI per answer
* active: True iff the question's parent gating answer matches the
  predicted plurality of the parent. Always-asked questions
  (smooth-or-featured, merging) are always active.

The "active" flag drives the T4.3 frontend's parent-dependency greyout
(inactive questions render greyed out to indicate the volunteer
decision tree would never reach them given the model's prediction).

Designed for low CPU latency (matches the FastAPI server's HF Spaces
target). The encoder runs once per request; the rest is small-tensor
math on the resulting 34-d alpha vector.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from galaxy_vit.data.schema import (
    always_asked_questions,
    get_dependencies,
    get_questions,
    question_index_groups,
)
from galaxy_vit.data.transforms import build_eval_transform, load_normalization
from galaxy_vit.inference.posterior import credible_interval, posterior_mean
from galaxy_vit.models.dirichlet_head import build_zoobot_dirichlet

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from PIL.Image import Image as PILImage

DEFAULT_NORMALIZATION = Path("configs/normalization.json")
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CI = 0.95


@dataclass
class PosteriorAnswer:
    """One answer's posterior summary within a question."""

    name: str
    mean: float
    ci_lower: float
    ci_upper: float


@dataclass
class PosteriorQuestion:
    """One question's full posterior summary."""

    question: str
    answers: list[PosteriorAnswer]
    plurality_answer: str
    plurality_index: int
    n_effective: float
    active: bool
    parent_question: str | None
    parent_answer: str | None


def _load_temperature(calibrated_path: Path | None) -> tuple[float, str]:
    """Read the chosen calibration temperature from a calibrated_metrics.json.

    Picks the single-T value (matches the T3.6 acceptance test's
    "winning regime" selection). If the file is missing OR has no
    single_T entry, returns (1.0, "none") — raw, uncalibrated.
    """
    if calibrated_path is None or not calibrated_path.is_file():
        return 1.0, "none"
    payload = json.loads(calibrated_path.read_text(encoding="utf-8"))
    single = payload.get("single_T")
    if not single or "T" not in single:
        return 1.0, "none"
    return float(single["T"]), "single_T"


class DirichletPosteriorPredictor:
    """Encapsulates the T3.6 model + calibration + the eval transform."""

    def __init__(
        self,
        ckpt_path: Path,
        *,
        calibrated_metrics_path: Path | None = None,
        normalization_path: Path = DEFAULT_NORMALIZATION,
        image_size: int = DEFAULT_IMAGE_SIZE,
        device: str = "cpu",
        encoder_id: str | None = None,
        alpha_floor: float = 1.0,
        num_answers: int = 34,
    ) -> None:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        self.device = torch.device(device)

        model, _encoder, _head = build_zoobot_dirichlet(
            num_answers=num_answers,
            alpha_floor=alpha_floor,
            encoder_id=encoder_id,
        )
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        self.model = model.to(self.device)

        mean, std = load_normalization(normalization_path)
        self.transform = build_eval_transform(image_size=image_size, mean=mean, std=std)

        self.temperature, self.calibration_regime = _load_temperature(
            calibrated_metrics_path
        )

    @torch.no_grad()
    def _predict_alpha(self, image: PILImage) -> torch.Tensor:
        """Run image -> (1, 34) calibrated alpha tensor on CPU."""
        x: torch.Tensor = self.transform(image).to(self.device).unsqueeze(0)
        alpha = self.model(pixel_values=x).alpha.float().cpu()
        if self.temperature != 1.0:
            alpha = alpha / self.temperature
        assert isinstance(alpha, torch.Tensor)
        return alpha

    def predict_posterior(
        self, image: PILImage, *, ci: float = DEFAULT_CI
    ) -> list[PosteriorQuestion]:
        """Return the per-question posterior summary for one image.

        The returned list is in canonical question order. Each entry
        contains:

        * The per-answer posterior mean + ``ci`` credible interval.
        * The argmax plurality answer (drives the parent-dependency
          greyout for child questions).
        * ``active`` flag: True iff the volunteer decision tree would
          reach this question given the model's predicted pluralities.
        """
        alpha = self._predict_alpha(image)  # (1, 34)
        groups = question_index_groups()
        means = posterior_mean(alpha, question_groups=groups)  # (1, 34)
        lower, upper = credible_interval(
            alpha, question_groups=groups, ci=ci
        )  # (1, 34) each

        questions = get_questions()
        deps = get_dependencies()
        always_asked = set(always_asked_questions())

        # Step 1: compute pluralities per question (needed for greyout).
        pluralities: dict[str, int] = {}
        for q_name, start, end in groups:
            pluralities[q_name] = int(means[0, start:end].argmax().item())

        # Step 2: build per-question summaries with greyout cascade.
        out: list[PosteriorQuestion] = []
        active_cache: dict[str, bool] = {}

        def is_active(q: str) -> bool:
            if q in active_cache:
                return active_cache[q]
            if q in always_asked:
                active_cache[q] = True
                return True
            parent_spec = deps[q]
            if parent_spec is None:
                active_cache[q] = True
                return True
            parent_q, gating_a = parent_spec
            if not is_active(parent_q):
                active_cache[q] = False
                return False
            parent_answers = questions[parent_q]
            parent_pred_idx = pluralities[parent_q]
            parent_pred_answer = parent_answers[parent_pred_idx]
            active_cache[q] = parent_pred_answer == gating_a
            return active_cache[q]

        for q_name, start, end in groups:
            answers_list = questions[q_name]
            plurality_idx = pluralities[q_name]
            slice_alpha = alpha[0, start:end]
            n_effective = float(slice_alpha.sum().item())
            answers: list[PosteriorAnswer] = []
            for i, a in enumerate(answers_list):
                answers.append(
                    PosteriorAnswer(
                        name=a,
                        mean=float(means[0, start + i].item()),
                        ci_lower=float(lower[0, start + i].item()),
                        ci_upper=float(upper[0, start + i].item()),
                    )
                )
            parent_spec = deps[q_name]
            out.append(
                PosteriorQuestion(
                    question=q_name,
                    answers=answers,
                    plurality_answer=answers_list[plurality_idx],
                    plurality_index=plurality_idx,
                    n_effective=n_effective,
                    active=is_active(q_name),
                    parent_question=parent_spec[0] if parent_spec else None,
                    parent_answer=parent_spec[1] if parent_spec else None,
                )
            )
        return out


def posterior_to_payload(posteriors: list[PosteriorQuestion]) -> list[dict[str, Any]]:
    """Convert the dataclass list to a JSON-serializable list of dicts."""
    return [
        {
            "question": q.question,
            "answers": [
                {
                    "name": a.name,
                    "mean": a.mean,
                    "ci_lower": a.ci_lower,
                    "ci_upper": a.ci_upper,
                }
                for a in q.answers
            ],
            "plurality_answer": q.plurality_answer,
            "plurality_index": q.plurality_index,
            "n_effective": q.n_effective,
            "active": q.active,
            "parent_question": q.parent_question,
            "parent_answer": q.parent_answer,
        }
        for q in posteriors
    ]
