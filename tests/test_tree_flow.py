"""Unit tests for galaxy_vit.inference.tree_flow (A-5).

Tree flow reach probabilities are tested against analytically tractable
toy schemas (1 parent + 1 child) and the real GZ DESI schema (10
questions, 34 answers). No checkpoint, no API — these run anywhere
torch is installed.
"""

from __future__ import annotations

import math

import pytest

from galaxy_vit.inference.tree_flow import TreeNode, compute_tree_flow

TOY_GROUPS = [("q0", 0, 2), ("q1", 2, 4)]
TOY_QUESTIONS = {"q0": ("yes", "no"), "q1": ("a", "b")}
TOY_DEPS: dict[str, tuple[str, str] | None] = {
    "q0": None,
    "q1": ("q0", "yes"),
}
TOY_ALWAYS_ASKED = ("q0",)


def _toy_flow(alpha: list[float]) -> dict[str, TreeNode]:
    nodes = compute_tree_flow(
        alpha,
        question_groups=TOY_GROUPS,
        questions=TOY_QUESTIONS,
        dependencies=TOY_DEPS,
        always_asked=TOY_ALWAYS_ASKED,
    )
    return {n.id: n for n in nodes}


def test_always_asked_question_has_reach_one() -> None:
    nodes = _toy_flow([3.0, 1.0, 1.0, 1.0])
    assert nodes["q:q0"].reach == pytest.approx(1.0, abs=1e-9)


def test_child_reach_equals_parent_gating_probability() -> None:
    """For q1 (gated on q0=yes) with alpha_q0=(3,1) -> P(yes)=0.75 -> reach(q1)=0.75."""
    nodes = _toy_flow([3.0, 1.0, 1.0, 1.0])
    assert nodes["q:q1"].reach == pytest.approx(0.75, abs=1e-9)


def test_answer_reach_equals_question_reach_times_answer_fraction() -> None:
    """reach(q1, a) = reach(q1) * P(a | q1) = 0.75 * 0.5 = 0.375."""
    nodes = _toy_flow([3.0, 1.0, 1.0, 1.0])
    assert nodes["a:q1_a"].reach == pytest.approx(0.375, abs=1e-9)
    assert nodes["a:q1_b"].reach == pytest.approx(0.375, abs=1e-9)


def test_per_question_reach_sums_to_question_reach() -> None:
    """Sum of answer-reach over a question = reach of the question itself."""
    nodes = _toy_flow([3.0, 1.0, 2.0, 3.0])
    reach_q0 = nodes["q:q0"].reach
    reach_q1 = nodes["q:q1"].reach
    sum_q0 = nodes["a:q0_yes"].reach + nodes["a:q0_no"].reach
    sum_q1 = nodes["a:q1_a"].reach + nodes["a:q1_b"].reach
    assert sum_q0 == pytest.approx(reach_q0, abs=1e-9)
    assert sum_q1 == pytest.approx(reach_q1, abs=1e-9)


def test_validation_rejects_wrong_alpha_length() -> None:
    with pytest.raises(ValueError, match="alpha length"):
        compute_tree_flow(
            [1.0, 1.0, 1.0],  # length 3, expected 4
            question_groups=TOY_GROUPS,
            questions=TOY_QUESTIONS,
            dependencies=TOY_DEPS,
            always_asked=TOY_ALWAYS_ASKED,
        )


# --- Real-schema acceptance tests ---


def _real_flow(alpha: list[float]) -> dict[str, TreeNode]:
    nodes = compute_tree_flow(alpha)  # schema loaded from galaxy-datasets
    return {n.id: n for n in nodes}


def test_real_schema_smooth_gated_subtree_matches_smooth_prob() -> None:
    """DEVPLAN acceptance: P(smooth) > 0.9 -> how-rounded reach ≈ 0.9.

    Build an alpha where smooth-or-featured strongly prefers 'smooth':
    alpha_0=18, alpha_1=1, alpha_2=1 -> P(smooth) = 0.9.
    """
    pytest.importorskip("galaxy_datasets")
    # 34 answers total in canonical GZ DESI schema.
    alpha = [1.0] * 34
    alpha[0] = 18.0  # smooth
    nodes = _real_flow(alpha)
    p_smooth = 18.0 / (18.0 + 1.0 + 1.0)
    assert nodes["q:how-rounded"].reach == pytest.approx(p_smooth, abs=1e-3)


def test_real_schema_featured_gated_subtree_matches_featured_prob() -> None:
    """For P(featured) ≈ 0.05: disk-edge-on reach ≈ 0.05.

    The gating answer for disk-edge-on is 'featured-or-disk'.
    """
    pytest.importorskip("galaxy_datasets")
    alpha = [1.0] * 34
    alpha[0] = 18.0  # smooth
    alpha[1] = 1.0  # featured-or-disk
    alpha[2] = 1.0  # artifact
    nodes = _real_flow(alpha)
    p_featured = 1.0 / 20.0
    assert nodes["q:disk-edge-on"].reach == pytest.approx(
        p_featured, abs=1e-3
    )


def test_real_schema_leaf_reach_bounded_by_parent_reach() -> None:
    """Monotonicity: every node's reach <= its parent's reach."""
    pytest.importorskip("galaxy_datasets")
    # Random plausible alpha (uniform high).
    alpha = [1.5] * 34
    nodes_list = compute_tree_flow(alpha)
    nodes = {n.id: n for n in nodes_list}
    for n in nodes_list:
        if n.kind == "answer":
            parent = nodes[f"q:{n.question}"]
            assert n.reach <= parent.reach + 1e-9
        if n.kind == "question" and n.parent_question and n.parent_answer:
            parent_answer = nodes[f"a:{n.parent_question}_{n.parent_answer}"]
            assert n.reach <= parent_answer.reach + 1e-9


def test_real_schema_always_asked_questions_have_reach_one() -> None:
    pytest.importorskip("galaxy_datasets")
    alpha = [1.5] * 34
    nodes = _real_flow(alpha)
    # 'smooth-or-featured' and 'merging' are the always-asked questions
    # per the GZ DESI schema.
    assert nodes["q:smooth-or-featured"].reach == pytest.approx(1.0)
    assert nodes["q:merging"].reach == pytest.approx(1.0)


def test_real_schema_returns_44_nodes_total() -> None:
    """10 question nodes + 34 answer nodes = 44 total."""
    pytest.importorskip("galaxy_datasets")
    alpha = [1.0] * 34
    nodes = compute_tree_flow(alpha)
    question_nodes = [n for n in nodes if n.kind == "question"]
    answer_nodes = [n for n in nodes if n.kind == "answer"]
    assert len(question_nodes) == 10
    assert len(answer_nodes) == 34


def test_real_schema_node_ids_are_unique() -> None:
    pytest.importorskip("galaxy_datasets")
    alpha = [1.0] * 34
    nodes = compute_tree_flow(alpha)
    ids = [n.id for n in nodes]
    assert len(ids) == len(set(ids))


def test_tree_flow_accepts_tensor() -> None:
    """Torch tensors are accepted in addition to plain lists."""
    pytest.importorskip("galaxy_datasets")
    torch = pytest.importorskip("torch")
    alpha = torch.ones(34, dtype=torch.float32)
    nodes = compute_tree_flow(alpha)
    assert len(nodes) == 44
    # Uniform alpha -> uniform per-question fractions -> sanity check.
    smooth_or_featured = next(
        n for n in nodes if n.id == "a:smooth-or-featured_smooth"
    )
    assert math.isclose(smooth_or_featured.reach, 1 / 3, abs_tol=1e-6)
