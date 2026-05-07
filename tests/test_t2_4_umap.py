"""T2.4 — UMAP penultimate-feature silhouette acceptance gate.

Reads the lightweight metrics summary written by ``scripts/extract_umap.py``
and asserts:

* The companion PNG figure exists.
* The silhouette score across the three smooth-or-featured classes
  (``smooth``, ``featured-or-disk``, ``artifact``) clears the project
  floor (default 0.15, set in DEVPLAN T2.4).
* The label distribution covers all three classes (catches a degenerate
  run where the test split somehow loses a class).
* The metric reports a non-trivial sample count (catches "extraction
  silently terminated after one batch" regressions).

Skipped when the run hasn't been done; re-run with::

    python -m scripts.extract_umap \\
        --config configs/m2_w23_reproduction.yaml \\
        --checkpoint runs/m2_w23_reproduction/best.pt
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

METRICS_PATH = Path("artifacts/umap_metrics.json")
PNG_PATH = Path("artifacts/umap_penultimate.png")
SILHOUETTE_FLOOR = 0.15
MIN_SAMPLES = 1000  # well below the ~25k DR8 test population


@pytest.mark.skipif(
    not METRICS_PATH.is_file(),
    reason=(
        "run UMAP extractor first: "
        "python -m scripts.extract_umap "
        "--config configs/m2_w23_reproduction.yaml "
        "--checkpoint runs/m2_w23_reproduction/best.pt"
    ),
)
def test_T2_4_umap_artifacts_and_silhouette() -> None:
    """T2.4 acceptance: PNG present, silhouette clears the 0.15 floor."""
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    assert PNG_PATH.is_file(), f"missing UMAP figure: {PNG_PATH}"

    silhouette = float(metrics["silhouette_score"])
    assert silhouette >= SILHOUETTE_FLOOR, (
        f"silhouette {silhouette:.4f} below T2.4 floor {SILHOUETTE_FLOOR}; "
        f"see {METRICS_PATH} for full context"
    )

    n_total = int(metrics["n_samples_total"])
    assert n_total >= MIN_SAMPLES, (
        f"only {n_total} samples in UMAP fit (expected >= {MIN_SAMPLES})"
    )

    label_counts = metrics["label_counts"]
    expected_classes = {"smooth", "featured-or-disk", "artifact"}
    missing = expected_classes - set(label_counts.keys())
    assert not missing, (
        f"smooth-or-featured class(es) missing from UMAP run: {sorted(missing)}"
    )
    for cls, count in label_counts.items():
        assert int(count) > 0, f"class {cls!r} has zero samples in UMAP run"
