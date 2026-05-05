"""Galaxy10 DECaLS dataset loader (HF Hub: matthieulel/galaxy10_decals).

Thin wrapper around `datasets.load_dataset` that:

- Caches under ``DATA_DIR / huggingface`` so the canonical settings entrypoint
  controls cache location.
- Imports `datasets` lazily so that `galaxy_vit.data.galaxy10` remains
  importable in environments without the m1 optional extra installed
  (e.g. CI lint/test). Calling the loader functions still requires the dep.
- Exposes a `make_split(...)` helper that combines load + stratified_split
  and returns a list of (index, label, split) rows ready for CSV writing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from galaxy_vit.data.splits import stratified_split

if TYPE_CHECKING:
    from collections.abc import Iterable

GALAXY10_HF_ID = "matthieulel/galaxy10_decals"
GALAXY10_NUM_CLASSES = 10


def load_galaxy10(data_dir: Path) -> Any:
    """Return the full Galaxy10 DECaLS DatasetDict (downloads on first call).

    The HF ``datasets`` library handles caching, integrity checks, and
    HF_TOKEN auth implicitly via the user's process environment.
    """
    import datasets

    cache_dir = data_dir / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return datasets.load_dataset(GALAXY10_HF_ID, cache_dir=str(cache_dir))


def extract_labels(rows: Iterable[dict[str, Any]]) -> list[int]:
    """Flatten an iterable of dataset rows into a list of integer labels.

    Galaxy10's HF dataset uses the column name ``label`` for the integer
    class id (0..9). We coerce to ``int`` defensively in case the underlying
    arrow column is a Class typed feature exposed as numpy.int64.
    """
    return [int(row["label"]) for row in rows]


def make_split(
    data_dir: Path,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> list[tuple[int, int, str]]:
    """End-to-end: load Galaxy10, extract labels, stratified-split, return rows.

    The returned rows are ``(global_index, label, split_name)`` tuples sorted
    by ``global_index`` so the CSV is a stable function of (dataset version,
    seed) for reproducibility audits.

    ``global_index`` is the row offset within the dataset's *concatenated*
    splits (HF Galaxy10 ships ``train`` + ``test``; we treat the union as a
    single index space and re-split ourselves). Column-level access is used
    instead of row iteration so HF doesn't trigger the image decoder — the
    splitter only needs labels and Pillow remains out of the m1 dep set.
    """
    ds = load_galaxy10(data_dir)
    labels: list[int] = []
    for split_key in sorted(ds.keys()):
        labels.extend(int(x) for x in ds[split_key]["label"])

    splits = stratified_split(labels, ratios=ratios, seed=seed)

    rows: list[tuple[int, int, str]] = []
    for split_name, indices in splits.items():
        for idx in indices:
            rows.append((idx, labels[idx], split_name))
    rows.sort(key=lambda r: r[0])
    return rows
