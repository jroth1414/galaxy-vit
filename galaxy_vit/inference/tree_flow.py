"""Decision-tree reach probabilities for the Sankey diagram (A-5).

Given a per-galaxy Dirichlet ``alpha`` (shape ``(34,)``) and the
canonical GZ DESI question/dependency schema, compute the **reach
probability** of every node in the decision tree. Two layers of nodes
are surfaced:

* **Question nodes** — one per GZ DESI question. Reach probability is
  the probability that a volunteer would be asked this question given
  the model's posterior on the parent. Always-asked questions
  (``smooth-or-featured``, ``merging``) have reach 1.
* **Answer nodes** — one per ``(question, answer)`` pair. Reach
  probability is ``reach(question) * posterior_mean(answer | question)``.

Math (multiplicative cascade):

    reach(Q)           = reach(parent(Q)) * P(parent answer = gate | parent)
    reach((Q, A))      = reach(Q)         * P(answer = A | Q)

with always-asked questions seeded at reach=1.

The posterior mean within a question slice is the Dirichlet point
estimate ``alpha_i / sum(alpha_q)``, identical to what
:func:`galaxy_vit.losses.dirichlet_mn.expected_fractions` produces.

This module is heavyweight-dep-free at import time; torch is lazy-
imported inside :func:`compute_tree_flow` so unit tests can run on
the ``[dev]``-only CI install.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover -- typing-only
    from torch import Tensor

EPS = 1e-12


@dataclass(frozen=True)
class TreeNode:
    """A single Sankey node (either a question or an answer)."""

    id: str
    label: str
    kind: str  # "question" | "answer"
    question: str
    answer: str | None  # None for question nodes
    reach: float  # probability volunteers visit this node, in [0, 1]
    parent_question: str | None  # question this gates from, if any
    parent_answer: str | None  # the answer on the parent that gates this


def _posterior_mean_per_question(
    alpha: list[float],
    *,
    question_groups: list[tuple[str, int, int]],
) -> dict[str, list[float]]:
    """Compute per-question posterior-mean fractions for one galaxy.

    Returns ``{q_name: [p_0, p_1, ..., p_{K-1}]}`` where each slice
    sums to 1. Zero-sum slices (shouldn't happen with the head's
    softplus + alpha_floor, but cheap to defend) become uniform.
    """
    out: dict[str, list[float]] = {}
    for q, start, end in question_groups:
        slice_alpha = alpha[start:end]
        s = sum(slice_alpha)
        if s <= EPS:
            k = end - start
            out[q] = [1.0 / k] * k
        else:
            out[q] = [a / s for a in slice_alpha]
    return out


def compute_tree_flow(
    alpha: list[float] | Tensor,
    *,
    question_groups: list[tuple[str, int, int]] | None = None,
    questions: dict[str, tuple[str, ...]] | None = None,
    dependencies: dict[str, tuple[str, str] | None] | None = None,
    always_asked: tuple[str, ...] | None = None,
) -> list[TreeNode]:
    """Return every node's reach probability for one galaxy's alpha.

    Parameters
    ----------
    alpha :
        Length-34 vector of Dirichlet concentrations (list or 1-D tensor).
    question_groups :
        ``[(q_name, start, end)]`` slices into alpha. Default: load from
        :func:`galaxy_vit.data.schema.question_index_groups`.
    questions :
        ``{q_name: (a_0, a_1, ...)}`` answer lists. Default: schema.
    dependencies :
        ``{q_name: (parent_q, gating_a) | None}``. Default: schema.
    always_asked :
        Tuple of question names with no parent. Default: schema.

    Returns
    -------
    Flat list of :class:`TreeNode` records — all question nodes first
    (in canonical order), then all answer nodes. Reach values are
    floats in ``[0, 1]``.
    """
    # Materialise alpha as a plain Python list so we don't need torch at
    # module load time and the toy unit tests can pass lists directly.
    alpha_list: list[float] = (
        list(alpha.tolist()) if hasattr(alpha, "tolist") else list(alpha)
    )

    if question_groups is None or questions is None or dependencies is None:
        from galaxy_vit.data.schema import (
            always_asked_questions,
            get_dependencies,
            get_questions,
            question_index_groups,
        )

        question_groups = question_groups or question_index_groups()
        questions = questions or get_questions()
        dependencies = dependencies or get_dependencies()
        if always_asked is None:
            always_asked = always_asked_questions()
    elif always_asked is None:
        always_asked = tuple(
            q for q, parent in dependencies.items() if parent is None
        )

    expected_len = sum(end - start for _, start, end in question_groups)
    if len(alpha_list) != expected_len:
        raise ValueError(
            f"alpha length {len(alpha_list)} != expected {expected_len} "
            f"(from question_groups)"
        )

    per_q_mean = _posterior_mean_per_question(
        alpha_list, question_groups=question_groups
    )
    answer_index: dict[str, dict[str, int]] = {
        q: {a: i for i, a in enumerate(answers)}
        for q, answers in questions.items()
    }

    # Resolve question reach via memoised recursion (cycles are
    # impossible by GZ DESI schema construction).
    q_reach: dict[str, float] = {}
    always_set = set(always_asked)

    def _resolve_q_reach(q: str) -> float:
        if q in q_reach:
            return q_reach[q]
        parent = dependencies.get(q)
        if q in always_set or parent is None:
            q_reach[q] = 1.0
            return 1.0
        parent_q, gating_a = parent
        parent_reach = _resolve_q_reach(parent_q)
        a_idx = answer_index[parent_q].get(gating_a)
        if a_idx is None:
            raise ValueError(
                f"dependency cites unknown answer: {parent_q}_{gating_a}"
            )
        p_gate = per_q_mean[parent_q][a_idx]
        q_reach[q] = parent_reach * p_gate
        return q_reach[q]

    for q, _start, _end in question_groups:
        _resolve_q_reach(q)

    nodes: list[TreeNode] = []
    for q, _start, _end in question_groups:
        parent_spec = dependencies.get(q)
        nodes.append(
            TreeNode(
                id=f"q:{q}",
                label=q,
                kind="question",
                question=q,
                answer=None,
                reach=q_reach[q],
                parent_question=parent_spec[0] if parent_spec else None,
                parent_answer=parent_spec[1] if parent_spec else None,
            )
        )

    for q, _start, _end in question_groups:
        means = per_q_mean[q]
        parent_spec = dependencies.get(q)
        for a_name, p_a in zip(questions[q], means, strict=True):
            nodes.append(
                TreeNode(
                    id=f"a:{q}_{a_name}",
                    label=a_name,
                    kind="answer",
                    question=q,
                    answer=a_name,
                    reach=q_reach[q] * p_a,
                    parent_question=parent_spec[0] if parent_spec else None,
                    parent_answer=parent_spec[1] if parent_spec else None,
                )
            )

    return nodes


def tree_flow_to_payload(nodes: list[TreeNode]) -> list[dict[str, object]]:
    """Convert dataclass list to JSON-serialisable dicts for the API."""
    return [
        {
            "id": n.id,
            "label": n.label,
            "kind": n.kind,
            "question": n.question,
            "answer": n.answer,
            "reach": n.reach,
            "parent_question": n.parent_question,
            "parent_answer": n.parent_answer,
        }
        for n in nodes
    ]
