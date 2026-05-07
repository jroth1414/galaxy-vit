"""Canonical GZ DESI schema sourced from upstream ``galaxy-datasets`` (T3.1).

Until T3.1 the question / answer ordering lived as a hardcoded dict in
:mod:`galaxy_vit.data.gz_desi`. From T3.1 forward the canonical source
of truth is :mod:`galaxy_datasets.shared.label_metadata` — specifically
``desi_pairs`` (question -> answer list) and ``desi_dependencies``
(question -> parent gating ``"<question>_<answer>"`` string, or
``None`` for always-asked questions).

This module:

* Loads the upstream schema lazily on first access (so the package
  remains importable in environments that only have the bare ``[dev]``
  extra installed without ``galaxy-datasets``).
* Strips the leading ``_`` from each upstream answer name so the
  resulting answers (``smooth``, ``featured-or-disk``, ...) match the
  catalog column convention ``<question>_<answer>``.
* Asserts at load time that the derived schema is exactly compatible
  with the legacy hardcoded :data:`galaxy_vit.data.gz_desi.GZ_DESI_QUESTIONS`
  — guards the existing T2.3 checkpoint against any silent reordering
  in a future ``galaxy-datasets`` release.

Dependency parents are exposed as parsed ``(parent_question, parent_answer)``
tuples (instead of the raw upstream ``"<q>_<a>"`` string) so the T3.4
masking logic can route through the GZ DESI decision tree without
re-parsing on every call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only
    pass

# Re-export legacy hardcoded constants for convenience; downstream code
# should treat :data:`QUESTIONS` (loaded from upstream) as canonical and
# fall back to :data:`galaxy_vit.data.gz_desi.GZ_DESI_QUESTIONS` only as
# an offline-friendly mirror.
from galaxy_vit.data.gz_desi import GZ_DESI_QUESTIONS as _LEGACY_QUESTIONS

ParentSpec = tuple[str, str] | None


def _strip_leading_underscore(answer: str) -> str:
    """Convert upstream ``_smooth`` → catalog convention ``smooth``."""
    return answer[1:] if answer.startswith("_") else answer


def _parse_dependency(spec: str | None) -> ParentSpec:
    """Parse upstream ``"<question>_<answer>"`` into ``(question, answer)`` or None."""
    if spec is None:
        return None
    # GZ DESI question names contain hyphens but never underscores; the FIRST
    # underscore separates question from answer (e.g.
    # ``"smooth-or-featured_featured-or-disk"`` → ``("smooth-or-featured",
    # "featured-or-disk")``).
    if "_" not in spec:
        raise ValueError(f"malformed dependency spec (missing _): {spec!r}")
    parent_q, parent_a = spec.split("_", 1)
    return (parent_q, parent_a)


def _load_upstream_schema() -> tuple[
    dict[str, tuple[str, ...]],
    dict[str, ParentSpec],
]:
    """Pull ``desi_pairs`` and ``desi_dependencies`` from galaxy-datasets.

    Lazy import keeps galaxy_vit.data.schema importable in dev / CI
    environments without the heavyweight ``[m1-train]`` extra.
    """
    from galaxy_datasets.shared import label_metadata as lm

    raw_pairs: dict[str, list[str]] = lm.desi_pairs
    raw_deps: dict[str, str | None] = lm.desi_dependencies

    questions: dict[str, tuple[str, ...]] = {
        q: tuple(_strip_leading_underscore(a) for a in answers)
        for q, answers in raw_pairs.items()
    }
    dependencies: dict[str, ParentSpec] = {
        q: _parse_dependency(spec) for q, spec in raw_deps.items()
    }
    return questions, dependencies


def _validate_canonical_compatibility(
    upstream: dict[str, tuple[str, ...]],
    legacy: dict[str, tuple[str, ...]],
) -> None:
    """Loud-fail if upstream schema drifted from the T2.3-trained ordering.

    A silent reorder would invalidate any existing checkpoint (the head
    is a flat 34-logit Linear and its slice-per-question layout is
    determined by the dict iteration order).
    """
    if list(upstream.keys()) != list(legacy.keys()):
        raise RuntimeError(
            "GZ DESI question ordering drift between galaxy-datasets and "
            f"legacy hardcoded constants:\n  upstream={list(upstream.keys())}\n"
            f"  legacy  ={list(legacy.keys())}"
        )
    for q, upstream_answers in upstream.items():
        legacy_answers = legacy[q]
        if upstream_answers != legacy_answers:
            raise RuntimeError(
                f"answer-list drift for question {q!r}:\n  upstream={upstream_answers}\n"
                f"  legacy  ={legacy_answers}"
            )


_QUESTIONS_CACHE: dict[str, tuple[str, ...]] | None = None
_DEPENDENCIES_CACHE: dict[str, ParentSpec] | None = None


def _ensure_loaded() -> None:
    global _QUESTIONS_CACHE, _DEPENDENCIES_CACHE
    if _QUESTIONS_CACHE is not None and _DEPENDENCIES_CACHE is not None:
        return
    upstream_q, upstream_d = _load_upstream_schema()
    _validate_canonical_compatibility(upstream_q, _LEGACY_QUESTIONS)
    _QUESTIONS_CACHE = upstream_q
    _DEPENDENCIES_CACHE = upstream_d


def get_questions() -> dict[str, tuple[str, ...]]:
    """Question -> answer-tuple dict, sourced from upstream ``galaxy-datasets``."""
    _ensure_loaded()
    assert _QUESTIONS_CACHE is not None
    return _QUESTIONS_CACHE


def get_dependencies() -> dict[str, ParentSpec]:
    """Question -> ``(parent_question, parent_answer)`` or ``None`` for always-asked."""
    _ensure_loaded()
    assert _DEPENDENCIES_CACHE is not None
    return _DEPENDENCIES_CACHE


def num_questions() -> int:
    return len(get_questions())


def num_answers() -> int:
    return sum(len(answers) for answers in get_questions().values())


def question_index_groups() -> list[tuple[str, int, int]]:
    """``[(question_name, start_idx, end_idx)]`` slices into the flat 34-logit head.

    Mirrors :func:`galaxy_vit.data.gz_desi_labels.question_index_groups` but
    walks the upstream-derived schema. T3.4's masking + per-question
    softmax logic uses these slices.
    """
    groups: list[tuple[str, int, int]] = []
    cursor = 0
    for q, answers in get_questions().items():
        groups.append((q, cursor, cursor + len(answers)))
        cursor += len(answers)
    return groups


def parent_of(question: str) -> ParentSpec:
    """Return the gating ``(question, answer)`` parent, or ``None`` for always-asked."""
    deps = get_dependencies()
    if question not in deps:
        raise KeyError(f"unknown GZ DESI question: {question!r}")
    return deps[question]


def always_asked_questions() -> tuple[str, ...]:
    """Tuple of questions with ``None`` parent (smooth-or-featured + merging)."""
    return tuple(q for q, parent in get_dependencies().items() if parent is None)


def vote_count_columns() -> list[str]:
    """Flat list of the 34 ``<question>_<answer>`` vote-count column names."""
    cols: list[str] = []
    for q, answers in get_questions().items():
        cols.extend(f"{q}_{a}" for a in answers)
    return cols


def vote_total_columns() -> list[str]:
    """Flat list of the 10 ``<question>_total-votes`` column names."""
    return [f"{q}_total-votes" for q in get_questions()]
