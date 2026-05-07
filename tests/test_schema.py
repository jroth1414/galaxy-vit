"""T3.1 — GZ DESI schema acceptance tests.

DEVPLAN T3.1 acceptance:

* total answers = sum of per-question counts
* every dependency key exists in the question list
* every dependency parent (question, answer) refers to a real question
  and a real answer in that question

Plus extra sanity tests that protect the existing T2.3 checkpoint:

* Upstream-derived ordering matches the legacy hardcoded
  ``GZ_DESI_QUESTIONS`` constant (catastrophic if it ever drifts).
* Always-asked set matches ``gz_desi.ALWAYS_ASKED``.
* The flat-tensor index groups match
  ``gz_desi_labels.question_index_groups`` (the layout T2.3's 34-logit
  Linear head is keyed on).

Skipped when ``galaxy-datasets`` is not installed (it ships with the
``[m1-train]`` extra). Bare ``[dev]`` CI sees the skip; the parity
guarantee is still enforced any time the heavy extra is present.
"""

from __future__ import annotations

import pytest

galaxy_datasets = pytest.importorskip(
    "galaxy_datasets",
    reason="install galaxy_vit[m1-train] (galaxy-datasets) to run schema tests",
)

from galaxy_vit.data import schema  # noqa: E402  (after importorskip is intentional)
from galaxy_vit.data.gz_desi import ALWAYS_ASKED, GZ_DESI_QUESTIONS  # noqa: E402
from galaxy_vit.data.gz_desi_labels import (  # noqa: E402
    question_index_groups as legacy_question_index_groups,
)


def test_T3_1_total_answers_matches_sum_per_question() -> None:
    """DEVPLAN T3.1 acceptance: NUM_ANSWERS == sum(len(answers) for answers in QUESTIONS)."""
    questions = schema.get_questions()
    total = sum(len(answers) for answers in questions.values())
    assert total == schema.num_answers()
    # GZ DESI is fixed at 10 questions / 34 answers; pin the constant so
    # any future upstream change becomes immediately obvious in CI.
    assert schema.num_questions() == 10
    assert schema.num_answers() == 34


def test_T3_1_every_dependency_key_is_a_known_question() -> None:
    """DEVPLAN T3.1 acceptance: every key in DEPENDENCIES exists in QUESTIONS."""
    questions = schema.get_questions()
    deps = schema.get_dependencies()
    missing = [q for q in deps if q not in questions]
    assert not missing, f"dependency keys not in question list: {missing}"
    # And every question must have a dependency entry (None or otherwise).
    missing_dep = [q for q in questions if q not in deps]
    assert not missing_dep, f"questions with no dependency entry: {missing_dep}"


def test_T3_1_every_dependency_parent_resolves() -> None:
    """Every non-null parent must reference a real ``(question, answer)`` pair."""
    questions = schema.get_questions()
    deps = schema.get_dependencies()
    failures: list[str] = []
    for q, parent in deps.items():
        if parent is None:
            continue
        parent_q, parent_a = parent
        if parent_q not in questions:
            failures.append(f"{q}: parent question {parent_q!r} unknown")
            continue
        if parent_a not in questions[parent_q]:
            failures.append(
                f"{q}: parent answer {parent_a!r} not in answers for {parent_q!r} "
                f"(have {questions[parent_q]})"
            )
    assert not failures, "dependency parents fail to resolve:\n  " + "\n  ".join(failures)


def test_T3_1_upstream_ordering_matches_legacy_constants() -> None:
    """Upstream schema must agree with hardcoded GZ_DESI_QUESTIONS ordering.

    A drift here would invalidate the T2.3 checkpoint (its 34-logit head
    layout is determined by the dict insertion order).
    """
    upstream = schema.get_questions()
    assert list(upstream.keys()) == list(GZ_DESI_QUESTIONS.keys())
    for q, upstream_answers in upstream.items():
        assert upstream_answers == GZ_DESI_QUESTIONS[q], (
            f"answer drift for {q!r}: upstream={upstream_answers}, "
            f"legacy={GZ_DESI_QUESTIONS[q]}"
        )


def test_T3_1_always_asked_match() -> None:
    """Always-asked questions (parent is None) match the legacy ALWAYS_ASKED tuple."""
    upstream_always_asked = schema.always_asked_questions()
    assert set(upstream_always_asked) == set(ALWAYS_ASKED)


def test_T3_1_question_index_groups_match_legacy() -> None:
    """The flat 34-logit slice layout must agree across schema.py and gz_desi_labels."""
    upstream_groups = schema.question_index_groups()
    legacy_groups = legacy_question_index_groups()
    assert upstream_groups == legacy_groups


def test_T3_1_vote_columns_round_trip() -> None:
    """vote_count_columns + vote_total_columns produce the catalog column names."""
    count_cols = schema.vote_count_columns()
    total_cols = schema.vote_total_columns()
    # Length checks.
    assert len(count_cols) == 34
    assert len(total_cols) == 10
    # Sample the catalog convention (matches docs/SCHEMA.md and gz_desi.py).
    assert "smooth-or-featured_smooth" in count_cols
    assert "smooth-or-featured_featured-or-disk" in count_cols
    assert "smooth-or-featured_artifact" in count_cols
    assert "smooth-or-featured_total-votes" in total_cols
    assert "merging_total-votes" in total_cols


def test_T3_1_parent_of_known_questions() -> None:
    """Spot-check parent_of for the GZ DESI decision tree's branch points."""
    assert schema.parent_of("smooth-or-featured") is None
    assert schema.parent_of("merging") is None
    assert schema.parent_of("disk-edge-on") == ("smooth-or-featured", "featured-or-disk")
    assert schema.parent_of("how-rounded") == ("smooth-or-featured", "smooth")
    assert schema.parent_of("edge-on-bulge") == ("disk-edge-on", "yes")
    # These three are the "featured-or-disk -> not edge-on" branch:
    assert schema.parent_of("has-spiral-arms") == ("disk-edge-on", "no")
    assert schema.parent_of("bar") == ("disk-edge-on", "no")
    assert schema.parent_of("bulge-size") == ("disk-edge-on", "no")
    # Spiral sub-questions are gated by has-spiral-arms=yes.
    assert schema.parent_of("spiral-winding") == ("has-spiral-arms", "yes")
    assert schema.parent_of("spiral-arm-count") == ("has-spiral-arms", "yes")


def test_T3_1_parent_of_unknown_raises() -> None:
    """parent_of with an unknown question name raises KeyError."""
    with pytest.raises(KeyError, match="unknown GZ DESI question"):
        schema.parent_of("not-a-real-question")
