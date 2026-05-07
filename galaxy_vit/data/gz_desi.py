"""Galaxy Zoo DESI decision-tree schema (T2.1, Phase 2 entry point).

Hardcoded from Walmsley+23's actual catalog format on Zenodo 8331338
(``docs/SCHEMA.md`` was wrong about the column suffix — see commit msg).
The 10-question / 34-answer tree maps to flat-tensor indices the same
way the upstream Zoobot 2.0 pipeline does — DEVPLAN T3.1 will replace
these constants with imports from ``galaxy-datasets``'s ``gz_desi_pairs``
and ``gz_desi_dependencies``, keeping the same surface area.

Catalog column naming convention (verified against
``gz_desi_gzd8_volunteer_core_catalog.parquet`` from Zenodo 8331338):

* ``<question>_<answer>``           — integer vote count for that answer
* ``<question>_total-votes``        — total votes cast on that question
* ``<question>_<answer>_fraction``  — debiased vote fraction in [0, 1]

Note: the catalog also has an auxiliary ``anything-odd`` question with
``yes/no`` answers that we ignore here (it's not part of the GZ DESI
decision tree the Dirichlet head will model). Extra columns are
tolerated by ``validate_schema``.
"""

from __future__ import annotations

from collections.abc import Iterable

# Insertion order is canonical and load-bearing — the flat tensor index of
# answer ``a`` of question ``q`` is determined by walking this dict and the
# answer lists in order. Reordering breaks all saved checkpoints.
GZ_DESI_QUESTIONS: dict[str, tuple[str, ...]] = {
    "smooth-or-featured": ("smooth", "featured-or-disk", "artifact"),
    "disk-edge-on": ("yes", "no"),
    "has-spiral-arms": ("yes", "no"),
    "bar": ("strong", "weak", "no"),
    "bulge-size": ("dominant", "large", "moderate", "small", "none"),
    "how-rounded": ("round", "in-between", "cigar-shaped"),
    "edge-on-bulge": ("boxy", "none", "rounded"),
    "spiral-winding": ("tight", "medium", "loose"),
    "spiral-arm-count": ("1", "2", "3", "4", "more-than-4", "cant-tell"),
    "merging": ("none", "minor-disturbance", "major-disturbance", "merger"),
}

NUM_QUESTIONS = len(GZ_DESI_QUESTIONS)
NUM_ANSWERS = sum(len(answers) for answers in GZ_DESI_QUESTIONS.values())

# Always-asked questions (no parent dependency); the others are gated by
# the GZ DESI dependency tree (see docs/SCHEMA.md §3 — wired up in T3.x).
ALWAYS_ASKED = ("smooth-or-featured", "merging")


def expected_vote_count_columns() -> list[str]:
    """Return the 34 ``<question>_<answer>`` integer-count column names."""
    return [
        f"{question}_{answer}"
        for question, answers in GZ_DESI_QUESTIONS.items()
        for answer in answers
    ]


def expected_vote_total_columns() -> list[str]:
    """Return the 10 ``<question>_total-votes`` column names (one per question)."""
    return [f"{question}_total-votes" for question in GZ_DESI_QUESTIONS]


def expected_vote_fraction_columns() -> list[str]:
    """Return the 34 ``<question>_<answer>_fraction`` column names in canonical order."""
    return [
        f"{question}_{answer}_fraction"
        for question, answers in GZ_DESI_QUESTIONS.items()
        for answer in answers
    ]


def validate_schema(columns: Iterable[str]) -> None:
    """Raise ``ValueError`` if any expected vote-count or per-question total column is missing.

    Extra columns are tolerated (the catalog has many auxiliary fields:
    ra, dec, dr8_id, fractions, the ``anything-odd`` question, etc.) —
    the validator only insists on the 34 vote-count columns and the 10
    per-question totals being present.
    """
    expected = set(expected_vote_count_columns()) | set(expected_vote_total_columns())
    actual = set(columns)
    missing = expected - actual
    if missing:
        sorted_missing = sorted(missing)
        head = ", ".join(sorted_missing[:5])
        more = (
            f" (+{len(sorted_missing) - 5} more)"
            if len(sorted_missing) > 5
            else ""
        )
        raise ValueError(
            f"missing {len(sorted_missing)}/{NUM_ANSWERS + NUM_QUESTIONS} expected "
            f"GZ DESI columns: {head}{more}"
        )
