"""T2.3 — multi-question loss + accuracy tests."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from galaxy_vit.data.gz_desi_labels import (  # noqa: E402
    NUM_QUESTIONS,
    extract_plurality_labels,
    question_index_groups,
)
from galaxy_vit.training.multi_question import (  # noqa: E402
    MultiQuestionAccumulator,
    multi_question_loss,
    multi_question_top1,
)


def test_question_index_groups_canonical() -> None:
    groups = question_index_groups()
    assert len(groups) == NUM_QUESTIONS
    # First entry is smooth-or-featured covering 0..3 (3 answers).
    assert groups[0] == ("smooth-or-featured", 0, 3)
    # Last entry is merging covering 30..34 (4 answers).
    assert groups[-1] == ("merging", 30, 34)
    # All slices are contiguous + total to 34.
    cursor = 0
    for _q, start, end in groups:
        assert start == cursor
        cursor = end
    assert cursor == 34


def test_extract_plurality_labels_simple() -> None:
    metadata = {
        "smooth-or-featured_smooth": 8,
        "smooth-or-featured_featured-or-disk": 2,
        "smooth-or-featured_artifact": 0,
        "smooth-or-featured_total-votes": 10,
        "merging_none": 1,
        "merging_minor-disturbance": 0,
        "merging_major-disturbance": 0,
        "merging_merger": 0,
        "merging_total-votes": 1,
    }
    labels, valid = extract_plurality_labels(metadata, min_votes=5)
    # smooth-or-featured: smooth (idx 0) is plurality, total=10 ok.
    assert labels[0] == 0
    assert valid[0] is True
    # merging: none (idx 0) plurality, total=1 < 5 -> invalid.
    assert labels[-1] == 0
    assert valid[-1] is False


def test_multi_question_loss_zero_when_all_invalid() -> None:
    """Loss is 0 when every question has zero valid samples in the batch."""
    logits = torch.randn(4, 34)
    plurality = torch.zeros(4, NUM_QUESTIONS, dtype=torch.long)
    valid = torch.zeros(4, NUM_QUESTIONS, dtype=torch.bool)
    loss = multi_question_loss(
        logits, plurality, valid, question_groups=question_index_groups()
    )
    assert loss.item() == 0.0


def test_multi_question_loss_positive_with_signal() -> None:
    """Loss is positive when at least one question has valid samples."""
    torch.manual_seed(0)
    logits = torch.randn(4, 34)
    plurality = torch.zeros(4, NUM_QUESTIONS, dtype=torch.long)
    valid = torch.zeros(4, NUM_QUESTIONS, dtype=torch.bool)
    valid[:, 0] = True  # only smooth-or-featured valid
    loss = multi_question_loss(
        logits, plurality, valid, question_groups=question_index_groups()
    )
    assert loss.item() > 0.0


def test_multi_question_top1_per_question() -> None:
    """Per-question accuracy is correct when we control the logits."""
    # 4-sample batch, smooth-or-featured (idx 0) only.
    logits = torch.zeros(4, 34)
    # Sample 0: argmax=smooth (slice 0..3 -> idx 0)
    logits[0, 0] = 5.0
    # Sample 1: argmax=featured (idx 1)
    logits[1, 1] = 5.0
    # Sample 2: argmax=artifact (idx 2)
    logits[2, 2] = 5.0
    # Sample 3: argmax=smooth (idx 0)
    logits[3, 0] = 5.0

    plurality = torch.zeros(4, NUM_QUESTIONS, dtype=torch.long)
    plurality[0, 0] = 0  # truth: smooth
    plurality[1, 0] = 1  # truth: featured
    plurality[2, 0] = 1  # truth: featured (model wrong, said artifact)
    plurality[3, 0] = 0  # truth: smooth

    valid = torch.zeros(4, NUM_QUESTIONS, dtype=torch.bool)
    valid[:, 0] = True

    out = multi_question_top1(
        logits, plurality, valid, question_groups=question_index_groups()
    )
    # 3 of 4 correct on smooth-or-featured.
    assert out["smooth-or-featured"]["top1"] == pytest.approx(3 / 4)
    assert out["smooth-or-featured"]["n_valid"] == 4
    # Other questions had no valid samples.
    assert out["merging"]["n_valid"] == 0


def test_multi_question_accumulator_aggregates() -> None:
    """Accumulator correctly tracks correct/valid across multiple updates."""
    acc = MultiQuestionAccumulator(question_index_groups())

    # Batch 1: 2 samples, smooth-or-featured valid.
    logits1 = torch.zeros(2, 34)
    logits1[0, 0] = 10.0  # smooth
    logits1[1, 1] = 10.0  # featured
    plurality1 = torch.zeros(2, NUM_QUESTIONS, dtype=torch.long)
    plurality1[0, 0] = 0  # truth: smooth -> correct
    plurality1[1, 0] = 0  # truth: smooth -> wrong (model said featured)
    valid1 = torch.zeros(2, NUM_QUESTIONS, dtype=torch.bool)
    valid1[:, 0] = True
    acc.update(logits1, plurality1, valid1)

    # Batch 2: 1 sample, both smooth-or-featured + merging valid.
    logits2 = torch.zeros(1, 34)
    logits2[0, 2] = 10.0  # artifact
    logits2[0, 32] = 10.0  # merging slice 30..34, idx 32 -> major-disturbance (idx 2)
    plurality2 = torch.zeros(1, NUM_QUESTIONS, dtype=torch.long)
    plurality2[0, 0] = 2  # truth: artifact -> correct
    plurality2[0, 9] = 2  # truth: major-disturbance -> correct
    valid2 = torch.zeros(1, NUM_QUESTIONS, dtype=torch.bool)
    valid2[:, 0] = True
    valid2[:, 9] = True
    acc.update(logits2, plurality2, valid2)

    result = acc.result()
    # smooth-or-featured: 2 of 3 correct.
    assert result["smooth-or-featured"]["n_valid"] == 3
    assert result["smooth-or-featured"]["top1"] == pytest.approx(2 / 3)
    # merging: 1 of 1 correct.
    assert result["merging"]["n_valid"] == 1
    assert result["merging"]["top1"] == 1.0
    # macro: average of per-question top-1 across active questions.
    assert acc.macro_top1 == pytest.approx((2 / 3 + 1.0) / 2)
