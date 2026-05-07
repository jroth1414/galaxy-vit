"""HF mwalmsley/gz_desi_wds dataset adapter (T2.3 pivot from Legacy Survey).

The HF dataset's labels JSON has 98 columns: each of the 10 GZ DESI
questions appears in 3 versions (``-dr5``, ``-dr8``, ``-dr12``) for the
three Galaxy Zoo campaigns over DECaLS imaging, with slightly different
answer sets per campaign. Each galaxy was voted on by exactly one
campaign — the other two have all-zero counts.

Walmsley+23's reproduction baseline trains only on the DR8 subset, which
is exactly the same 34-answer schema my T2.1 catalog already encodes
(just stored under ``<question>-dr8_<answer>`` keys here vs.
``<question>_<answer>`` in the T2.1 parquet). This module bridges the
two: :func:`hf_labels_to_canonical` turns an HF labels dict into the
canonical 34-column dict the rest of the pipeline already understands,
and :func:`has_any_dr8_votes` filters out galaxies that were voted on
under a different campaign.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from galaxy_vit.data.gz_desi import GZ_DESI_QUESTIONS

DR8_INFIX = "-dr8_"


def hf_labels_to_canonical(hf_labels: Mapping[str, Any]) -> dict[str, int]:
    """Project an HF labels dict to the canonical 34-column GZ DESI dr8 schema.

    Reads only ``<question>-dr8_<answer>`` columns; ignores ``-dr5_`` and
    ``-dr12_``. Returns a dict with:

    * ``<question>_<answer>`` -> int vote count (matching T2.1 schema)
    * ``<question>_total-votes`` -> sum of per-answer counts for that question

    Float counts in the source (HF stores as float64) are coerced to int.
    Missing or null entries are treated as zero.
    """
    out: dict[str, int] = {}
    for question, answers in GZ_DESI_QUESTIONS.items():
        total = 0
        for answer in answers:
            hf_key = f"{question}{DR8_INFIX}{answer}"
            v = hf_labels.get(hf_key, 0)
            count = 0 if v is None else int(v)
            out[f"{question}_{answer}"] = count
            total += count
        out[f"{question}_total-votes"] = total
    return out


def has_any_dr8_votes(hf_labels: Mapping[str, Any]) -> bool:
    """Return True if the galaxy has at least one DR8 vote across all questions."""
    for question, answers in GZ_DESI_QUESTIONS.items():
        for answer in answers:
            v = hf_labels.get(f"{question}{DR8_INFIX}{answer}", 0)
            if v is not None and float(v) > 0:
                return True
    return False
