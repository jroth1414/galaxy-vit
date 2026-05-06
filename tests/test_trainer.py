"""T1.4 — trainer / metrics / model factory acceptance tests.

* Metrics: pure-stdlib tests on tiny tensors (no torch import beyond what
  the metrics functions use internally).
* ViT smoke: builds a fresh-init ViT-B/16 from a ``ViTConfig`` (no Hub
  download) and verifies a forward pass returns the right logit shape.
* Threshold gate: skipped unless a ``runs/m1_vit_baseline/metrics.json``
  exists; when it does, asserts best val top-1 >= 0.82 and best val
  macro-F1 >= 0.78 (DEVPLAN T1.4 acceptance).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from galaxy_vit.training.metrics import macro_f1, per_class_counts, top1_accuracy  # noqa: E402

METRICS_FILE = Path("runs/m1_vit_baseline/metrics.json")


# ------------------------------------------------------------------ metrics


def test_top1_accuracy_basic() -> None:
    preds = torch.tensor([0, 1, 2, 1, 0])
    labels = torch.tensor([0, 1, 2, 0, 1])
    # 3 correct out of 5
    assert top1_accuracy(preds, labels) == pytest.approx(0.6)


def test_top1_accuracy_empty() -> None:
    preds = torch.tensor([], dtype=torch.long)
    labels = torch.tensor([], dtype=torch.long)
    assert top1_accuracy(preds, labels) == 0.0


def test_macro_f1_perfect() -> None:
    preds = torch.tensor([0, 1, 2, 0, 1, 2])
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    assert macro_f1(preds, labels, num_classes=3) == pytest.approx(1.0)


def test_macro_f1_handcrafted() -> None:
    """3-class case with a known macro-F1 value."""
    preds = torch.tensor([0, 1, 2, 1, 0])
    labels = torch.tensor([0, 1, 2, 0, 1])
    # class 0: tp=1, fp=1, fn=1 -> P=R=0.5, F1=0.5
    # class 1: tp=1, fp=1, fn=1 -> P=R=0.5, F1=0.5
    # class 2: tp=1, fp=0, fn=0 -> P=R=1.0, F1=1.0
    # macro F1 = (0.5 + 0.5 + 1.0) / 3 = 2/3
    assert macro_f1(preds, labels, num_classes=3) == pytest.approx(2 / 3)


def test_per_class_counts_shape_and_support() -> None:
    preds = torch.tensor([0, 1, 2, 1, 0])
    labels = torch.tensor([0, 1, 2, 0, 1])
    counts = per_class_counts(preds, labels, num_classes=3)
    assert set(counts.keys()) == {"tp", "fp", "fn", "support"}
    for v in counts.values():
        assert len(v) == 3
    # support per class equals the number of true labels in that class
    assert counts["support"] == [2, 2, 1]


# --------------------------------------------------------------- ViT smoke


def test_T1_4_vit_forward_pass_shape() -> None:
    """Fresh-init ViT-B/16 with 10-class head accepts (B, 3, 224, 224) -> (B, 10).

    Uses ``ViTConfig`` to avoid the HF Hub download in CI; structural smoke
    only.
    """
    from transformers import ViTConfig, ViTForImageClassification

    config = ViTConfig(image_size=224, num_labels=10)
    model = ViTForImageClassification(config)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(pixel_values=x)
    assert out.logits.shape == (2, 10), f"got {tuple(out.logits.shape)}"


def test_T1_4_param_groups_split() -> None:
    """split_param_groups partitions encoder vs head params with separate LRs."""
    from transformers import ViTConfig, ViTForImageClassification

    from galaxy_vit.models.vit_baseline import split_param_groups

    config = ViTConfig(image_size=224, num_labels=10)
    model = ViTForImageClassification(config)

    groups = split_param_groups(
        model, encoder_lr=1.0e-5, head_lr=1.0e-3, weight_decay=0.05
    )
    assert len(groups) == 2
    assert groups[0]["lr"] == 1.0e-5
    assert groups[1]["lr"] == 1.0e-3
    assert all(g["weight_decay"] == 0.05 for g in groups)
    n_enc = sum(p.numel() for p in groups[0]["params"])
    n_head = sum(p.numel() for p in groups[1]["params"])
    # head is one Linear(768 -> 10); encoder has ~85M params for ViT-B/16.
    assert n_enc > n_head, f"encoder {n_enc} should exceed head {n_head}"
    assert n_head == 768 * 10 + 10  # weight + bias


# ---------------------------------------------------- T1.4 threshold gate


@pytest.mark.skipif(
    not METRICS_FILE.is_file(),
    reason=(
        "run trainer first: "
        "python -m galaxy_vit.training.trainer --config configs/m1_vit_baseline.yaml"
    ),
)
def test_T1_4_metrics_threshold_met() -> None:
    """T1.4 acceptance: best val top-1 >= 0.82 AND best val macro-F1 >= 0.78."""
    payload = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    best = payload["best_val"]
    top1 = best["top1"]
    f1 = best["macro_f1"]

    assert top1 is not None, "best_val.top1 missing in metrics.json"
    assert f1 is not None, "best_val.macro_f1 missing in metrics.json"
    assert top1 >= 0.82, f"val top-1 {top1:.4f} below 0.82 threshold"
    assert f1 >= 0.78, f"val macro-F1 {f1:.4f} below 0.78 threshold"
