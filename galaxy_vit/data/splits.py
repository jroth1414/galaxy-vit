"""Stratified train / val / test split utility.

Pure stdlib (no numpy / no datasets) so it can be imported and tested without
the m1 optional extras installed. Used by `galaxy_vit.data.galaxy10` (Month 1)
and reused for GZ DESI in Month 2.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence


def stratified_split(
    labels: Sequence[int],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    split_names: tuple[str, str, str] = ("train", "val", "test"),
) -> dict[str, list[int]]:
    """Return ``{split: [indices]}`` partitioning ``labels`` by class.

    Each class' indices are shuffled with the seeded RNG and partitioned
    proportionally to ``ratios`` using the largest-remainder method
    (Hamilton's method). Aggregating the per-class allocations preserves
    class balance across splits to within rounding.

    Parameters
    ----------
    labels:
        Per-sample integer class label, one per dataset row.
    ratios:
        Three floats that sum to 1.0; default (0.70, 0.15, 0.15) for
        train / val / test.
    seed:
        RNG seed for the per-class shuffle. Default 42 (project convention).
    split_names:
        Names used as keys in the returned mapping. Order matches ``ratios``.

    Returns
    -------
    A dict with keys equal to ``split_names`` and values equal to lists of
    indices (each index in ``range(len(labels))`` appears in exactly one
    split). The union of values equals ``set(range(len(labels)))``.

    Raises
    ------
    ValueError
        If ``ratios`` doesn't have length 3, doesn't sum to ~1.0, has any
        non-positive entry, or if ``split_names`` doesn't have length 3.
    """
    if len(ratios) != 3:
        raise ValueError(f"ratios must have length 3; got {len(ratios)}")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0; got {ratios} (sum={sum(ratios)})")
    if min(ratios) <= 0:
        raise ValueError(f"all ratios must be positive; got {ratios}")
    if len(split_names) != 3:
        raise ValueError(f"split_names must have length 3; got {split_names}")
    if len(set(split_names)) != 3:
        raise ValueError(f"split_names must be unique; got {split_names}")

    rng = random.Random(seed)

    by_label: dict[int, list[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        by_label[lab].append(idx)

    result: dict[str, list[int]] = {name: [] for name in split_names}

    # Sort by label for deterministic iteration order regardless of input order.
    for _label, indices in sorted(by_label.items()):
        rng.shuffle(indices)
        counts = _largest_remainder(len(indices), ratios)
        cursor = 0
        for split_i, count in enumerate(counts):
            result[split_names[split_i]].extend(indices[cursor : cursor + count])
            cursor += count

    return result


def _largest_remainder(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Allocate ``n`` items across 3 buckets via Hamilton's largest-remainder method.

    Floor each ideal allocation, then distribute the remainder one at a time
    to the buckets with the largest fractional parts (ties broken by bucket
    index — the earlier bucket wins, which gives ``train`` a slight edge as
    the natural "default" for any leftover sample on small classes).
    """
    ideals = [n * r for r in ratios]
    floors = [int(x) for x in ideals]
    remainder = n - sum(floors)
    # Sort bucket indices by descending fractional part; stable so ties prefer
    # the lower bucket index (train > val > test under the default ratios).
    order = sorted(range(3), key=lambda i: -(ideals[i] - floors[i]))
    for i in range(remainder):
        floors[order[i]] += 1
    return (floors[0], floors[1], floors[2])
