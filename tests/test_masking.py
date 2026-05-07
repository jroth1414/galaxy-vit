"""T3.2 — Synthetic-galaxy masking tests (acceptance suite).

Originally written as a TDD gate against a NotImplementedError stub
(xfail decorators); flipped to live tests at T3.4 once HITL #2 approved
the semantics encoded here and ``compute_question_mask`` got its body.

The 5 DEVPLAN cases (one per question situation):

1. ``all-full``        — every question reachable in the decision tree
                         has plenty of votes; mask True everywhere
                         that's reachable, False where the decision
                         tree gates the question off.
2. ``below-min-votes`` — a reachable question has fewer than
                         ``min_votes`` answer counts; that question
                         masks False even though gating is satisfied.
3. ``zero-on-dependent`` — gated child question has zero answer counts
                         (volunteers were never routed there); masks
                         False because total < min_votes.
4. ``parent-lost``     — parent question's plurality picked a different
                         answer than the gating one; child masks False
                         and the cascade propagates to grandchildren.
5. ``parent-tie``      — parent has a tied plurality. Both ``tie_policy``
                         values are exercised (default ``"argmax"`` picks
                         the lowest-index winner per ``torch.argmax``;
                         ``"drop"`` disqualifies all descendants).

Plus boundary checks:

* Always-asked questions (``smooth-or-featured``, ``merging``) ignore
  the parent chain entirely.
* Cascading invalidation through 2 levels (e.g., spiral-winding when
  has-spiral-arms is itself unreachable).
* The ``tie_policy="argmax"`` default matches Zoobot 2.0 conventions —
  this is load-bearing for the T3.5 loss-parity acceptance test.
"""

from __future__ import annotations

import pytest

galaxy_datasets = pytest.importorskip(
    "galaxy_datasets",
    reason="install galaxy_vit[m1-train] to exercise the schema-driven masking",
)

from galaxy_vit.data.masking import compute_question_mask  # noqa: E402


def _featured_disk_galaxy_full() -> dict[str, dict[str, int]]:
    """A bona-fide 'featured-or-disk, not-edge-on, has-spirals' galaxy.

    Every question reachable along this branch has well above min_votes=5;
    questions on the unreachable branches (how-rounded, edge-on-bulge)
    have zero votes (volunteers never got routed there).
    """
    return {
        "smooth-or-featured": {"smooth": 2, "featured-or-disk": 22, "artifact": 1},
        # featured-or-disk -> ask disk-edge-on
        "disk-edge-on": {"yes": 3, "no": 18},
        # disk-edge-on=no -> ask has-spiral-arms, bar, bulge-size
        "has-spiral-arms": {"yes": 16, "no": 4},
        "bar": {"strong": 4, "weak": 6, "no": 10},
        "bulge-size": {"dominant": 1, "large": 3, "moderate": 9, "small": 6, "none": 1},
        # has-spiral-arms=yes -> ask spiral-winding, spiral-arm-count
        "spiral-winding": {"tight": 3, "medium": 9, "loose": 4},
        "spiral-arm-count": {"1": 0, "2": 11, "3": 4, "4": 1, "more-than-4": 0, "cant-tell": 0},
        # smooth-or-featured=smooth path NOT taken; how-rounded gets 0 votes.
        "how-rounded": {"round": 0, "in-between": 0, "cigar-shaped": 0},
        # disk-edge-on=yes path NOT taken; edge-on-bulge gets 0 votes.
        "edge-on-bulge": {"boxy": 0, "none": 0, "rounded": 0},
        # merging is always-asked.
        "merging": {"none": 19, "minor-disturbance": 4, "major-disturbance": 1, "merger": 0},
    }


def _smooth_galaxy_full() -> dict[str, dict[str, int]]:
    """A bona-fide 'smooth, round' galaxy on the alternative branch."""
    return {
        "smooth-or-featured": {"smooth": 21, "featured-or-disk": 3, "artifact": 0},
        # smooth-or-featured=featured-or-disk path NOT taken.
        "disk-edge-on": {"yes": 0, "no": 0},
        "has-spiral-arms": {"yes": 0, "no": 0},
        "bar": {"strong": 0, "weak": 0, "no": 0},
        "bulge-size": {"dominant": 0, "large": 0, "moderate": 0, "small": 0, "none": 0},
        "spiral-winding": {"tight": 0, "medium": 0, "loose": 0},
        "spiral-arm-count": {"1": 0, "2": 0, "3": 0, "4": 0, "more-than-4": 0, "cant-tell": 0},
        "edge-on-bulge": {"boxy": 0, "none": 0, "rounded": 0},
        # smooth-or-featured=smooth -> ask how-rounded
        "how-rounded": {"round": 17, "in-between": 4, "cigar-shaped": 1},
        "merging": {"none": 20, "minor-disturbance": 2, "major-disturbance": 0, "merger": 0},
    }


# ---------------------------------------------------------------------------
# Case 1 — all-full
# ---------------------------------------------------------------------------


def test_T3_2_all_full_featured_branch() -> None:
    """Featured-or-disk galaxy: reachable questions True, off-branch questions False."""
    votes = _featured_disk_galaxy_full()
    mask = compute_question_mask(votes, min_votes=5)
    assert mask["smooth-or-featured"] is True
    assert mask["merging"] is True
    assert mask["disk-edge-on"] is True
    # disk-edge-on=no branch:
    assert mask["has-spiral-arms"] is True
    assert mask["bar"] is True
    assert mask["bulge-size"] is True
    # has-spiral-arms=yes branch:
    assert mask["spiral-winding"] is True
    assert mask["spiral-arm-count"] is True
    # Off-branch (smooth-or-featured=smooth):
    assert mask["how-rounded"] is False
    # Off-branch (disk-edge-on=yes):
    assert mask["edge-on-bulge"] is False


def test_T3_2_all_full_smooth_branch() -> None:
    """Smooth galaxy: only smooth-or-featured + how-rounded + merging are True."""
    votes = _smooth_galaxy_full()
    mask = compute_question_mask(votes, min_votes=5)
    assert mask["smooth-or-featured"] is True
    assert mask["how-rounded"] is True
    assert mask["merging"] is True
    assert mask["disk-edge-on"] is False
    assert mask["has-spiral-arms"] is False
    assert mask["bar"] is False
    assert mask["bulge-size"] is False
    assert mask["edge-on-bulge"] is False
    assert mask["spiral-winding"] is False
    assert mask["spiral-arm-count"] is False


# ---------------------------------------------------------------------------
# Case 2 — below-min-votes
# ---------------------------------------------------------------------------


def test_T3_2_below_min_votes_disqualifies_own_question() -> None:
    """Reachable child with total < min_votes masks False even if gating succeeded."""
    votes = _featured_disk_galaxy_full()
    # Drop bar's vote count below min_votes.
    votes["bar"] = {"strong": 1, "weak": 1, "no": 1}  # total=3 < 5
    mask = compute_question_mask(votes, min_votes=5)
    assert mask["bar"] is False
    # Sibling questions (has-spiral-arms, bulge-size) are unaffected.
    assert mask["has-spiral-arms"] is True
    assert mask["bulge-size"] is True
    # Grandchildren of has-spiral-arms still valid.
    assert mask["spiral-winding"] is True


def test_T3_2_below_min_votes_on_parent_cascades() -> None:
    """If a parent question itself has total < min_votes, all descendants mask False."""
    votes = _featured_disk_galaxy_full()
    # Drop has-spiral-arms below min_votes.
    votes["has-spiral-arms"] = {"yes": 2, "no": 1}  # total=3 < 5
    mask = compute_question_mask(votes, min_votes=5)
    assert mask["has-spiral-arms"] is False
    # Both children of has-spiral-arms must cascade-fail.
    assert mask["spiral-winding"] is False
    assert mask["spiral-arm-count"] is False


# ---------------------------------------------------------------------------
# Case 3 — zero-on-dependent
# ---------------------------------------------------------------------------


def test_T3_2_zero_votes_on_dependent_masks_false() -> None:
    """Reachable child with zero answer counts masks False (total=0 < min_votes)."""
    votes = _featured_disk_galaxy_full()
    # Volunteers were routed to bar (gating succeeded) but no answers cast.
    votes["bar"] = {"strong": 0, "weak": 0, "no": 0}
    mask = compute_question_mask(votes, min_votes=5)
    assert mask["bar"] is False
    # Other reachable questions still True.
    assert mask["has-spiral-arms"] is True
    assert mask["bulge-size"] is True


# ---------------------------------------------------------------------------
# Case 4 — parent-lost (gating answer didn't win the parent's plurality)
# ---------------------------------------------------------------------------


def test_T3_2_parent_lost_masks_descendants_false() -> None:
    """If the parent's plurality is not the gating answer, child masks False."""
    votes = _featured_disk_galaxy_full()
    # Flip disk-edge-on to predominantly 'yes' (so gating for has-spiral-arms,
    # bar, bulge-size — which require disk-edge-on=no — fails).
    votes["disk-edge-on"] = {"yes": 17, "no": 4}
    # Provide enough votes on edge-on-bulge so its OWN total >= min_votes.
    votes["edge-on-bulge"] = {"boxy": 4, "none": 11, "rounded": 2}
    mask = compute_question_mask(votes, min_votes=5)
    # Now edge-on-bulge (gated by disk-edge-on=yes) is True instead.
    assert mask["edge-on-bulge"] is True
    # disk-edge-on=no children all cascade False because parent's plurality
    # is "yes", not the required "no".
    assert mask["has-spiral-arms"] is False
    assert mask["bar"] is False
    assert mask["bulge-size"] is False
    # Grandchildren of has-spiral-arms also False (cascade).
    assert mask["spiral-winding"] is False
    assert mask["spiral-arm-count"] is False


# ---------------------------------------------------------------------------
# Case 5 — parent-tie (tie_policy: 'argmax' default vs 'drop')
# ---------------------------------------------------------------------------


def _tied_parent_galaxy() -> dict[str, dict[str, int]]:
    """A galaxy where smooth-or-featured ties between 'smooth' and 'featured-or-disk'.

    Counts: smooth=10, featured-or-disk=10, artifact=2 (total 22, well above
    min_votes). With ``tie_policy='argmax'`` the lowest-index answer
    ('smooth') wins, so how-rounded is reachable. With ``tie_policy='drop'``
    no descendant is reachable.

    All downstream questions get plenty of votes so the ONLY thing
    determining their mask is the parent-tie semantic.
    """
    return {
        "smooth-or-featured": {"smooth": 10, "featured-or-disk": 10, "artifact": 2},
        "disk-edge-on": {"yes": 4, "no": 11},  # featured-or-disk path
        "has-spiral-arms": {"yes": 9, "no": 2},
        "bar": {"strong": 2, "weak": 4, "no": 5},
        "bulge-size": {"dominant": 1, "large": 2, "moderate": 5, "small": 3, "none": 0},
        "how-rounded": {"round": 7, "in-between": 3, "cigar-shaped": 1},  # smooth path
        "edge-on-bulge": {"boxy": 0, "none": 0, "rounded": 0},
        "spiral-winding": {"tight": 2, "medium": 4, "loose": 1},
        "spiral-arm-count": {"1": 0, "2": 6, "3": 1, "4": 0, "more-than-4": 0, "cant-tell": 0},
        "merging": {"none": 19, "minor-disturbance": 2, "major-disturbance": 1, "merger": 0},
    }


def test_T3_2_parent_tie_argmax_picks_lowest_index() -> None:
    """Default tie_policy='argmax' picks the lowest-index answer (matches torch.argmax)."""
    votes = _tied_parent_galaxy()
    mask = compute_question_mask(votes, min_votes=5)  # default tie_policy='argmax'
    # 'smooth' is the lowest-index answer in (smooth, featured-or-disk, artifact),
    # so the smooth branch is taken.
    assert mask["how-rounded"] is True
    # ...and the featured-or-disk branch is NOT taken.
    assert mask["disk-edge-on"] is False
    # Cascading off disk-edge-on:
    assert mask["has-spiral-arms"] is False
    assert mask["bar"] is False
    assert mask["bulge-size"] is False
    assert mask["spiral-winding"] is False
    assert mask["spiral-arm-count"] is False
    # Always-asked questions unaffected.
    assert mask["smooth-or-featured"] is True
    assert mask["merging"] is True


def test_T3_2_parent_tie_drop_disqualifies_all_descendants() -> None:
    """tie_policy='drop' masks every descendant False on a tied parent."""
    votes = _tied_parent_galaxy()
    mask = compute_question_mask(votes, min_votes=5, tie_policy="drop")
    # Both branches' immediate children fail.
    assert mask["how-rounded"] is False
    assert mask["disk-edge-on"] is False
    # Cascade descendants also False.
    assert mask["has-spiral-arms"] is False
    assert mask["bar"] is False
    assert mask["bulge-size"] is False
    assert mask["spiral-winding"] is False
    assert mask["spiral-arm-count"] is False
    assert mask["edge-on-bulge"] is False
    # Always-asked questions unaffected.
    assert mask["smooth-or-featured"] is True
    assert mask["merging"] is True


# ---------------------------------------------------------------------------
# Boundary cases
# ---------------------------------------------------------------------------


def test_T3_2_always_asked_ignores_parent_chain() -> None:
    """smooth-or-featured + merging are True iff their own total >= min_votes."""
    votes = _featured_disk_galaxy_full()
    votes["smooth-or-featured"] = {"smooth": 1, "featured-or-disk": 1, "artifact": 0}  # 2<5
    votes["merging"] = {"none": 1, "minor-disturbance": 0, "major-disturbance": 0, "merger": 0}
    mask = compute_question_mask(votes, min_votes=5)
    assert mask["smooth-or-featured"] is False
    assert mask["merging"] is False
    # And every downstream question cascades False because
    # smooth-or-featured (their root parent) is invalid.
    assert mask["disk-edge-on"] is False
    assert mask["how-rounded"] is False


def test_T3_2_two_level_cascade_invalidation() -> None:
    """spiral-winding (grandchild) cascades from disk-edge-on going wrong."""
    votes = _featured_disk_galaxy_full()
    # Make has-spiral-arms vote pattern itself fine, but flip disk-edge-on
    # to predominantly 'yes' so has-spiral-arms is unreachable.
    votes["disk-edge-on"] = {"yes": 18, "no": 3}
    votes["edge-on-bulge"] = {"boxy": 4, "none": 11, "rounded": 2}
    mask = compute_question_mask(votes, min_votes=5)
    assert mask["edge-on-bulge"] is True
    assert mask["has-spiral-arms"] is False  # parent (disk-edge-on) chose wrong answer
    assert mask["spiral-winding"] is False  # grandparent of cascade
    assert mask["spiral-arm-count"] is False


def test_T3_2_returns_entry_for_every_question() -> None:
    """Output dict has exactly one bool per GZ DESI question, regardless of input."""
    from galaxy_vit.data import schema

    votes = _smooth_galaxy_full()
    mask = compute_question_mask(votes, min_votes=5)
    expected_questions = set(schema.get_questions().keys())
    assert set(mask.keys()) == expected_questions
    for q, v in mask.items():
        assert isinstance(v, bool), f"{q} mask is not bool: {type(v)}"


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_T3_2_unknown_tie_policy_raises_value_error() -> None:
    """tie_policy is restricted to the two literals."""
    with pytest.raises(ValueError, match="unknown tie_policy"):
        compute_question_mask(
            _smooth_galaxy_full(),
            min_votes=5,
            tie_policy="majority",  # type: ignore[arg-type]
        )


def test_T3_2_negative_min_votes_raises_value_error() -> None:
    """min_votes < 0 is nonsense and rejected at the call site."""
    with pytest.raises(ValueError, match="min_votes must be"):
        compute_question_mask(_smooth_galaxy_full(), min_votes=-1)
