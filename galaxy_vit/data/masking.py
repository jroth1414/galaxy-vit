"""Per-question validity mask honoring min_votes + the GZ DESI dependency tree.

A galaxy's vote on question ``q`` is **valid** iff:

1. The question's own ``total_votes`` (sum of per-answer counts) is at least
   ``min_votes``.
2. AND, recursively, the parent question (per
   :func:`galaxy_vit.data.schema.parent_of`) is itself valid AND its plurality
   answer matches the parent gating answer.

For always-asked questions (``smooth-or-featured``, ``merging``) the parent
chain bottoms out at ``None``; only condition (1) applies.

Tie semantics on the parent's plurality vote — handled by ``tie_policy``:

* ``"argmax"`` (default): lowest-index answer wins, matching ``torch.argmax``
  / ``numpy.argmax`` convention. This is the convention the upstream Zoobot
  2.0 pipeline uses, so the loss-parity check at T3.5 requires this default.
* ``"drop"``: any tied parent disqualifies all of its descendants. More
  conservative; useful for sensitivity analyses in T3.6.

T3.4 implements the body. The TDD gate (T3.2) wrote 11 xfailed tests
against the stub; T3.4 strips those decorators after this implementation
lands. HITL #2 approved the semantics encoded here:

* tie_policy default ``"argmax"`` for Zoobot / T3.5 compatibility
* below-min-votes on a parent cascades to all descendants
* zero-votes-on-dependent and parent-lost are distinct failure modes
  (both produce False masks); both are intentional
* always-asked questions ignore the parent chain entirely
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Literal

from galaxy_vit.data.schema import (
    ParentSpec,
    get_dependencies,
    get_questions,
)

TiePolicy = Literal["argmax", "drop"]


@lru_cache(maxsize=1)
def _schema_snapshot() -> tuple[
    tuple[tuple[str, tuple[str, ...]], ...],
    tuple[tuple[str, ParentSpec], ...],
]:
    """Cache the immutable schema as nested tuples for fast repeated access.

    The schema doesn't change at runtime, so we snapshot it once. ``lru_cache``
    on a no-arg function gives us thread-safe lazy initialization.
    """
    questions = tuple((q, tuple(answers)) for q, answers in get_questions().items())
    deps = tuple((q, parent) for q, parent in get_dependencies().items())
    return questions, deps


def _question_total(votes: Mapping[str, Mapping[str, int]], q: str) -> int:
    counts = votes.get(q, {})
    return sum(int(v) for v in counts.values())


def _plurality_winner(
    votes: Mapping[str, Mapping[str, int]],
    question: str,
    answers: tuple[str, ...],
    *,
    tie_policy: TiePolicy,
) -> str | None:
    """Return the plurality answer for ``question``, honoring ``tie_policy``.

    Returns ``None`` only when ``tie_policy='drop'`` and the top-class
    count is shared by 2+ answers. With ``'argmax'`` the canonical
    answer-list order breaks ties: the lowest-index tied answer wins
    (matches ``torch.argmax`` / ``numpy.argmax``).
    """
    counts = votes.get(question, {})
    paired = [(a, int(counts.get(a, 0))) for a in answers]
    max_count = max(c for _, c in paired)
    winners = [a for a, c in paired if c == max_count]
    if len(winners) > 1 and tie_policy == "drop":
        return None
    # 'argmax' path (or single winner): canonical answer-list order means
    # winners[0] is the lowest-index tied answer.
    return winners[0]


def compute_question_mask(
    votes: Mapping[str, Mapping[str, int]],
    *,
    min_votes: int = 5,
    tie_policy: TiePolicy = "argmax",
) -> dict[str, bool]:
    """Per-question validity mask for a single galaxy's vote-count dict.

    Parameters
    ----------
    votes:
        ``{question_name: {answer_name: count}}``. Missing answers default
        to 0; missing questions default to all-zero counts. Caller is
        responsible for using the canonical question / answer names from
        :func:`galaxy_vit.data.schema.get_questions`.
    min_votes:
        Minimum ``sum(answer_counts)`` for a question to be considered
        validly answered. Catalog default in T2.3 is 5.
    tie_policy:
        How to break ties in the parent's plurality answer when computing
        whether a child question's gating succeeded. ``"argmax"`` (default)
        picks the lowest-index answer (matches ``torch.argmax`` / Zoobot);
        ``"drop"`` treats the tied parent as disqualifying all descendants.

    Returns
    -------
    ``{question_name: bool}`` — one entry for every GZ DESI question, in
    the canonical order of :func:`galaxy_vit.data.schema.get_questions`.
    """
    if tie_policy not in ("argmax", "drop"):
        raise ValueError(
            f"unknown tie_policy {tie_policy!r}; expected 'argmax' or 'drop'"
        )
    if min_votes < 0:
        raise ValueError(f"min_votes must be >= 0, got {min_votes}")

    questions_tuple, deps_tuple = _schema_snapshot()
    answers_by_q: dict[str, tuple[str, ...]] = dict(questions_tuple)
    deps_by_q: dict[str, ParentSpec] = dict(deps_tuple)

    mask: dict[str, bool] = {}

    def _is_valid(q: str) -> bool:
        if q in mask:
            return mask[q]

        # Step 1: own-question vote-count threshold.
        if _question_total(votes, q) < min_votes:
            mask[q] = False
            return False

        # Step 2: parent gating chain.
        parent = deps_by_q[q]
        if parent is None:
            mask[q] = True
            return True

        parent_q, required_a = parent
        if not _is_valid(parent_q):
            # Parent is itself invalid (low votes OR its own gating failed);
            # cascade the failure to this child.
            mask[q] = False
            return False

        # Step 3: parent's plurality must equal the gating answer.
        parent_answers = answers_by_q[parent_q]
        winner = _plurality_winner(
            votes, parent_q, parent_answers, tie_policy=tie_policy
        )
        if winner is None:
            # tie_policy='drop' on a tied parent.
            mask[q] = False
            return False
        mask[q] = winner == required_a
        return mask[q]

    # Walk every question in canonical order; recursion via _is_valid will
    # populate parent entries as a side effect, but iterating here ensures
    # the output dict has all 10 keys regardless of which questions get
    # touched by the recursion.
    for q, _answers in questions_tuple:
        _is_valid(q)
    return mask
