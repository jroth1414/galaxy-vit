"""T2.1 — GZ DESI schema constants + validator + produced-catalog tests.

The schema-constant tests are pure stdlib — no pandas, no pyarrow, no
network. The catalog acceptance test loads the produced parquet and
skips if it isn't built yet (same pattern as the T1.4 / T1.5 metrics
gates).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from galaxy_vit.data.gz_desi import (
    ALWAYS_ASKED,
    GZ_DESI_QUESTIONS,
    NUM_ANSWERS,
    NUM_QUESTIONS,
    expected_vote_count_columns,
    expected_vote_fraction_columns,
    expected_vote_total_columns,
    validate_schema,
)

CATALOG_PATH = Path("data/gz_desi_volunteer_decals.parquet")
T2_1_MIN_ROWS = 80_000  # docs/SCHEMA.md §5b
T2_1_MAX_ROWS = 150_000


def test_gz_desi_question_count_matches_schema_doc() -> None:
    """docs/SCHEMA.md §2 lists 10 questions for GZ DESI."""
    assert NUM_QUESTIONS == 10
    assert len(GZ_DESI_QUESTIONS) == 10


def test_gz_desi_answer_count_matches_schema_doc() -> None:
    """3+2+2+3+5+3+3+3+6+4 = 34 total answers across the 10 questions."""
    assert NUM_ANSWERS == 34
    by_question = {q: len(a) for q, a in GZ_DESI_QUESTIONS.items()}
    assert by_question == {
        "smooth-or-featured": 3,
        "disk-edge-on": 2,
        "has-spiral-arms": 2,
        "bar": 3,
        "bulge-size": 5,
        "how-rounded": 3,
        "edge-on-bulge": 3,
        "spiral-winding": 3,
        "spiral-arm-count": 6,
        "merging": 4,
    }


def test_always_asked_questions() -> None:
    """smooth-or-featured and merging have no parent dependency."""
    assert set(ALWAYS_ASKED) == {"smooth-or-featured", "merging"}


def test_expected_vote_count_columns_complete_and_named() -> None:
    cols = expected_vote_count_columns()
    assert len(cols) == NUM_ANSWERS
    # Catalog convention: bare <question>_<answer>, NO _count suffix.
    assert all("_" in c for c in cols)
    assert all(not c.endswith("_count") for c in cols)
    assert all(not c.endswith("_fraction") for c in cols)
    # Spot-check the canonical names from Walmsley+23's catalog.
    assert "smooth-or-featured_smooth" in cols
    assert "smooth-or-featured_featured-or-disk" in cols
    assert "merging_merger" in cols
    assert "spiral-arm-count_more-than-4" in cols
    assert "spiral-arm-count_cant-tell" in cols


def test_expected_vote_total_columns() -> None:
    totals = expected_vote_total_columns()
    assert len(totals) == NUM_QUESTIONS
    assert all(c.endswith("_total-votes") for c in totals)
    assert "smooth-or-featured_total-votes" in totals
    assert "merging_total-votes" in totals


def test_expected_vote_fraction_columns_parallel_to_counts() -> None:
    counts = expected_vote_count_columns()
    fractions = expected_vote_fraction_columns()
    assert len(counts) == len(fractions)
    for c, f in zip(counts, fractions, strict=True):
        # Fraction column has the same question_answer prefix plus _fraction.
        assert f == f"{c}_fraction"


def test_validate_schema_passes_on_complete_columns() -> None:
    full = expected_vote_count_columns() + expected_vote_total_columns()
    validate_schema(full)
    # Extra columns are fine — the catalog has many auxiliary fields.
    extras = [
        "ra",
        "dec",
        "dr8_id",
        "anything-odd_yes",
        "anything-odd_no",
        "anything-odd_total-votes",
        *expected_vote_fraction_columns(),
    ]
    validate_schema(full + extras)


def test_validate_schema_rejects_missing_columns() -> None:
    full = expected_vote_count_columns() + expected_vote_total_columns()
    truncated = full[:-3]  # drop last 3
    expected_total = NUM_ANSWERS + NUM_QUESTIONS
    with pytest.raises(ValueError, match=rf"missing 3/{expected_total}"):
        validate_schema(truncated)


def test_validate_schema_error_lists_specific_missing() -> None:
    """Validator's error mentions the specific missing column names."""
    full = expected_vote_count_columns() + expected_vote_total_columns()
    # Drop a known column — error should name it.
    drop = "smooth-or-featured_smooth"
    with pytest.raises(ValueError, match=drop):
        validate_schema([c for c in full if c != drop])


# --------------------------------------------------- T2.1 catalog gate


@pytest.mark.skipif(
    not CATALOG_PATH.is_file(),
    reason=(
        "run `python scripts/build_gz_desi_catalog.py` to download Zenodo 8331338 "
        "and produce the volunteer catalog first"
    ),
)
def test_T2_1_catalog_row_count_and_schema() -> None:
    """T2.1 acceptance: produced parquet has 80-150k rows and the GZ DESI vote schema.

    Adjusted from the DEVPLAN's original 400-600k figure — see
    docs/SCHEMA.md §5b for the rationale (Zenodo 8331338's volunteer
    subset is ~100k galaxies; the 8.67M-row ``deep_learning_catalog``
    parquets contain only model-predicted fractions).
    """
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(CATALOG_PATH)
    n_rows = pf.metadata.num_rows
    columns = [field.name for field in pf.schema_arrow]

    assert T2_1_MIN_ROWS <= n_rows <= T2_1_MAX_ROWS, (
        f"row count {n_rows:,} outside [{T2_1_MIN_ROWS:,}, {T2_1_MAX_ROWS:,}]"
    )
    # Every expected GZ DESI vote-count + per-question total column is present.
    validate_schema(columns)
