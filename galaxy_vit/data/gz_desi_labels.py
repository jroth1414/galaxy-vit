"""Plurality-answer label extraction for GZ DESI multi-question training (T2.3).

Given a per-galaxy vote-count dict (from the catalog parquet or a tar-shard
JSON sidecar), produce:

* :func:`question_index_groups` — flat 34-logit head layout: per-question
  ``(name, start, end)`` slices into the model output. Canonical order
  matches :data:`galaxy_vit.data.gz_desi.GZ_DESI_QUESTIONS`.
* :func:`extract_plurality_labels` — per-question integer plurality
  answer (argmax of vote counts) + per-question validity mask
  (``total-votes >= min_votes``).
* :func:`stratification_label` — coarse 3-class label
  (smooth / featured-or-disk / artifact) for stratified-split keying.

T3.x will replace the plurality-target training with the full
Dirichlet-Multinomial likelihood; T2.3 is the calibration step that
checks our per-question argmax matches Walmsley+23's published table.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from galaxy_vit.data.gz_desi import GZ_DESI_QUESTIONS, NUM_ANSWERS

DEFAULT_MIN_VOTES = 5
QUESTION_NAMES = tuple(GZ_DESI_QUESTIONS.keys())
NUM_QUESTIONS = len(QUESTION_NAMES)


def question_index_groups() -> list[tuple[str, int, int]]:
    """Return ``[(question, start_idx, end_idx)]`` slices into a 34-logit vector."""
    groups: list[tuple[str, int, int]] = []
    cursor = 0
    for question, answers in GZ_DESI_QUESTIONS.items():
        groups.append((question, cursor, cursor + len(answers)))
        cursor += len(answers)
    assert cursor == NUM_ANSWERS, f"sum of answers != NUM_ANSWERS ({cursor} != {NUM_ANSWERS})"
    return groups


def extract_plurality_labels(
    metadata: Mapping[str, Any],
    *,
    min_votes: int = DEFAULT_MIN_VOTES,
) -> tuple[list[int], list[bool]]:
    """Per-question plurality answer index + ``total-votes >= min_votes`` mask.

    NaN / null total-votes (galaxies that were never shown that question
    because the parent answer disqualified them) are treated as 0 and
    therefore mark the question invalid.

    Parameters
    ----------
    metadata:
        Dict with ``<question>_<answer>`` integer counts and
        ``<question>_total-votes`` totals, as produced by T2.1's catalog
        builder + the tar-shard JSON sidecars.
    min_votes:
        Minimum total votes for a question to be considered valid.

    Returns
    -------
    ``(labels, valid)`` — both length ``NUM_QUESTIONS`` (10 for GZ DESI):

    * ``labels[i]`` = argmax of vote counts for question ``i`` (always
      defined; 0 if no votes were cast).
    * ``valid[i]`` = ``True`` if the question received >= min_votes; the
      trainer uses this to mask out per-question loss + accuracy.
    """
    labels: list[int] = []
    valid: list[bool] = []
    for question, answers in GZ_DESI_QUESTIONS.items():
        counts = []
        for answer in answers:
            v = metadata.get(f"{question}_{answer}", 0)
            counts.append(0 if v is None else int(v))
        total_v = metadata.get(f"{question}_total-votes", 0)
        total = 0 if total_v is None else int(total_v)
        plurality_idx = max(range(len(counts)), key=lambda i: counts[i])
        labels.append(plurality_idx)
        valid.append(total >= min_votes)
    return labels, valid


def stratification_label(
    metadata: Mapping[str, Any],
    *,
    min_votes: int = DEFAULT_MIN_VOTES,
) -> int:
    """Coarse 3-class label for stratified-split keying.

    Uses the plurality answer of the always-asked ``smooth-or-featured``
    question (``0=smooth, 1=featured-or-disk, 2=artifact``). Galaxies
    where ``smooth-or-featured`` has fewer than ``min_votes`` total are
    flagged with class ``-1`` (caller should drop these from
    stratification, which the T2.1 filter already did).
    """
    answers = GZ_DESI_QUESTIONS["smooth-or-featured"]
    counts = [int(metadata.get(f"smooth-or-featured_{a}", 0) or 0) for a in answers]
    total = int(metadata.get("smooth-or-featured_total-votes", 0) or 0)
    if total < min_votes:
        return -1
    return int(max(range(len(counts)), key=lambda i: counts[i]))
