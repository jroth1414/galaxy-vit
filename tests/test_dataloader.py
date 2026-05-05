"""T1.1 — Galaxy10 stratified split acceptance test.

Hermetic: uses tests/fixtures/galaxy10_synthetic_labels.csv (a 500-row,
10-class label-only file mirroring Galaxy10 DECaLS' class imbalance
qualitatively). No network, no `datasets` import.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from galaxy_vit.data.splits import stratified_split

FIXTURE = Path(__file__).parent / "fixtures" / "galaxy10_synthetic_labels.csv"
NUM_CLASSES = 10
N = 500
SIZE_TARGETS = {"train": 350, "val": 75, "test": 75}  # 70/15/15 of 500
SIZE_TOLERANCE = 5
RATIO_TOLERANCE = 0.01


def _load_fixture_labels() -> list[int]:
    with FIXTURE.open(encoding="utf-8", newline="") as fh:
        return [int(row["label"]) for row in csv.DictReader(fh)]


def test_galaxy10_split_sizes() -> None:
    """Stratified split satisfies T1.1 acceptance: ±5 sizes, ±1% class ratios.

    On a 500-sample synthetic fixture (sizes 100/80/60/60/40/40/40/40/20/20),
    `stratified_split(seed=42)` must:

      1. Produce exactly the three keys {"train","val","test"}.
      2. Have indices that partition range(500) (covering, no duplicates).
      3. Have split sizes within ±5 of (350, 75, 75).
      4. Have per-class ratio in each split within 1% (absolute) of the
         per-class ratio in the full dataset.
    """
    labels = _load_fixture_labels()
    assert len(labels) == N, f"fixture changed size: got {len(labels)}"

    splits = stratified_split(labels, ratios=(0.70, 0.15, 0.15), seed=42)

    # 1. Keys.
    assert set(splits.keys()) == {"train", "val", "test"}

    # 2. Coverage / disjoint partition.
    flattened: list[int] = []
    for indices in splits.values():
        flattened.extend(indices)
    assert sorted(flattened) == list(range(N)), (
        "split indices must form a partition of range(N) — no missing, no duplicates"
    )

    # 3. Size acceptance: |actual - target| ≤ 5.
    for name, target in SIZE_TARGETS.items():
        actual = len(splits[name])
        assert abs(actual - target) <= SIZE_TOLERANCE, (
            f"{name} size {actual} outside ±{SIZE_TOLERANCE} of target {target}"
        )

    # 4. Per-class ratio acceptance: within RATIO_TOLERANCE absolute.
    full_counts = Counter(labels)
    for split_name, indices in splits.items():
        split_counts = Counter(labels[i] for i in indices)
        n_split = len(indices)
        for cls in range(NUM_CLASSES):
            full_ratio = full_counts[cls] / N
            split_ratio = split_counts[cls] / n_split
            diff = abs(split_ratio - full_ratio)
            assert diff <= RATIO_TOLERANCE, (
                f"class {cls} in split {split_name!r}: ratio {split_ratio:.4f} "
                f"vs full {full_ratio:.4f}, diff {diff:.4f} > {RATIO_TOLERANCE}"
            )
