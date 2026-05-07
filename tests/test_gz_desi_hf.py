"""T2.3 — HF mwalmsley/gz_desi_wds adapter tests."""

from __future__ import annotations

from galaxy_vit.data.gz_desi import (
    GZ_DESI_QUESTIONS,
    expected_vote_count_columns,
    expected_vote_total_columns,
)
from galaxy_vit.data.gz_desi_hf import has_any_dr8_votes, hf_labels_to_canonical


def _hf_label_dict(per_question_dr8: dict[str, dict[str, int]]) -> dict[str, float]:
    """Construct an HF-shaped label dict from a per-question mapping.

    All non-dr8 columns are zeroed; only the requested dr8 entries get
    non-zero values. Caller passes ``{question_name: {answer_name: count, ...}}``.
    """
    out: dict[str, float] = {}
    for q, answers in GZ_DESI_QUESTIONS.items():
        for a in answers:
            for survey in ("dr5", "dr8", "dr12"):
                out[f"{q}-{survey}_{a}"] = 0.0
        # The HF data also has -dr12 with different answer sets — fake a few.
    # Apply the per-question DR8 votes.
    for q, votes in per_question_dr8.items():
        for a, c in votes.items():
            out[f"{q}-dr8_{a}"] = float(c)
    return out


def test_hf_to_canonical_smooth_only() -> None:
    """Galaxy with only smooth-or-featured DR8 votes: 5 smooth, 0 elsewhere."""
    hf = _hf_label_dict({"smooth-or-featured": {"smooth": 5}})
    canonical = hf_labels_to_canonical(hf)

    # Schema: every expected column present.
    for col in expected_vote_count_columns():
        assert col in canonical, f"missing {col}"
    for col in expected_vote_total_columns():
        assert col in canonical, f"missing {col}"

    # Smooth-or-featured: smooth=5, others=0, total=5.
    assert canonical["smooth-or-featured_smooth"] == 5
    assert canonical["smooth-or-featured_featured-or-disk"] == 0
    assert canonical["smooth-or-featured_artifact"] == 0
    assert canonical["smooth-or-featured_total-votes"] == 5
    # Other questions: zero.
    assert canonical["bar_strong"] == 0
    assert canonical["bar_total-votes"] == 0


def test_hf_to_canonical_multi_question() -> None:
    """Galaxy with answers across several DR8 questions."""
    hf = _hf_label_dict(
        {
            "smooth-or-featured": {"featured-or-disk": 8, "artifact": 1},
            "disk-edge-on": {"yes": 6, "no": 3},
            "bar": {"strong": 2, "weak": 4, "no": 1},
            "merging": {"none": 7, "minor-disturbance": 2},
        }
    )
    canonical = hf_labels_to_canonical(hf)

    assert canonical["smooth-or-featured_featured-or-disk"] == 8
    assert canonical["smooth-or-featured_artifact"] == 1
    assert canonical["smooth-or-featured_total-votes"] == 9

    assert canonical["disk-edge-on_yes"] == 6
    assert canonical["disk-edge-on_no"] == 3
    assert canonical["disk-edge-on_total-votes"] == 9

    assert canonical["bar_strong"] == 2
    assert canonical["bar_weak"] == 4
    assert canonical["bar_no"] == 1
    assert canonical["bar_total-votes"] == 7

    assert canonical["merging_none"] == 7
    assert canonical["merging_minor-disturbance"] == 2
    assert canonical["merging_major-disturbance"] == 0
    assert canonical["merging_merger"] == 0
    assert canonical["merging_total-votes"] == 9


def test_hf_to_canonical_ignores_dr5_dr12() -> None:
    """Non-dr8 votes don't leak into the canonical dict."""
    hf: dict[str, float] = {
        "smooth-or-featured-dr5_smooth": 5.0,
        "smooth-or-featured-dr12_smooth": 3.0,
        "smooth-or-featured-dr8_smooth": 2.0,
    }
    canonical = hf_labels_to_canonical(hf)
    assert canonical["smooth-or-featured_smooth"] == 2  # only dr8 read


def test_has_any_dr8_votes_true() -> None:
    hf = _hf_label_dict({"smooth-or-featured": {"smooth": 1}})
    assert has_any_dr8_votes(hf)


def test_has_any_dr8_votes_false_when_dr5_only() -> None:
    """Galaxy voted only under DR5 should be filtered out."""
    hf: dict[str, float] = {
        f"{q}-dr5_{a}": 5.0 for q, answers in GZ_DESI_QUESTIONS.items() for a in answers
    }
    # Add zero dr8 entries.
    for q, answers in GZ_DESI_QUESTIONS.items():
        for a in answers:
            hf[f"{q}-dr8_{a}"] = 0.0
    assert not has_any_dr8_votes(hf)


def test_has_any_dr8_votes_false_when_all_zero() -> None:
    hf = _hf_label_dict({})
    assert not has_any_dr8_votes(hf)
