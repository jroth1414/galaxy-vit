"""Per-question validity mask honoring min_votes + the GZ DESI dependency tree (T3.2 stub).

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

T3.2 status — STUB. The DEVPLAN T3.2 task is the TDD gate for this masking
logic: the unit tests in ``tests/test_masking.py`` are written first and
marked ``@pytest.mark.xfail(strict=True)`` against this NotImplementedError
stub. T3.4 implements the body and removes the xfail markers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

TiePolicy = Literal["argmax", "drop"]


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

    Notes
    -----
    T3.2 STUB — raises ``NotImplementedError``. Implementation lands at
    T3.4 once the HITL #2 review confirms the test cases capture the
    intended semantics.
    """
    if tie_policy not in ("argmax", "drop"):
        raise ValueError(
            f"unknown tie_policy {tie_policy!r}; expected 'argmax' or 'drop'"
        )
    if min_votes < 0:
        raise ValueError(f"min_votes must be >= 0, got {min_votes}")
    raise NotImplementedError(
        "compute_question_mask: T3.2 TDD stub — implementation lands at T3.4 "
        "after HITL #2 reviews the test cases in tests/test_masking.py"
    )
