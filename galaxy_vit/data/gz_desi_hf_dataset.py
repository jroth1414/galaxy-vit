"""IterableDatasets bridging HF gz_desi_wds shards into the trainer loops.

Two parallel variants share the same shard-walk + DR8-only filter +
worker-stride sharding logic, differing only in what each sample
yields:

* :func:`build_gz_desi_hf_dataset` (T2.3 plurality-CE path) yields
  ``(image, plurality, valid)``:
  - ``plurality``: ``(num_questions,)`` int64, argmax answer per question
  - ``valid``:     ``(num_questions,)`` bool, total >= ``min_votes``
* :func:`build_gz_desi_hf_dataset_for_dirichlet` (T3.6 Dirichlet-MN path)
  yields ``(image, counts, valid)``:
  - ``counts``:    ``(num_answers,)`` int64, per-answer raw vote counts
                    in canonical flat order (matches the head's 34-D layout)

Both filter galaxies with no DR8 votes (they were classified under a
different GZ campaign — DR5 or DR12 — and don't belong in either
training run). Multi-worker safe via ``torch.utils.data.get_worker_info``.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from galaxy_vit.data.gz_desi import GZ_DESI_QUESTIONS, NUM_ANSWERS
from galaxy_vit.data.gz_desi_hf import has_any_dr8_votes, hf_labels_to_canonical
from galaxy_vit.data.gz_desi_labels import (
    DEFAULT_MIN_VOTES,
    NUM_QUESTIONS,
    extract_plurality_labels,
)
from galaxy_vit.data.gz_desi_streaming import _iter_samples_from_shard


def _extract_count_vector(canonical: dict[str, int]) -> list[int]:
    """Flatten a canonical ``{<q>_<a>: count}`` dict to the 34-element answer vector.

    Walks ``GZ_DESI_QUESTIONS`` in canonical order (matches the head's
    flat-tensor layout) so ``counts[i]`` aligns with the ``i``-th alpha.
    """
    out: list[int] = []
    for question, answers in GZ_DESI_QUESTIONS.items():
        for answer in answers:
            v = canonical.get(f"{question}_{answer}", 0)
            out.append(0 if v is None else int(v))
    return out


def _extract_valid_mask(
    canonical: dict[str, int], *, min_votes: int
) -> list[bool]:
    """Per-question validity mask: True iff that question's total >= min_votes."""
    out: list[bool] = []
    for question in GZ_DESI_QUESTIONS:
        total_v = canonical.get(f"{question}_total-votes", 0)
        total = 0 if total_v is None else int(total_v)
        out.append(total >= min_votes)
    return out

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from PIL.Image import Image as PILImage
    from torch import Tensor


def build_gz_desi_hf_dataset(
    shard_paths: list[Path],
    transform: Callable[[PILImage], Tensor],
    *,
    shuffle_buffer: int = 1000,
    shardshuffle: bool = True,
    min_votes: int = DEFAULT_MIN_VOTES,
    seed: int | None = None,
) -> Any:
    """Construct an IterableDataset over a list of HF gz_desi_wds tar shards.

    ``shard_paths`` is the explicit ordered list of tar files for the
    desired split (caller filters HF shard inventory by filename
    pattern). The dataset filters DR8-only at iteration time and yields
    ``(image_tensor, plurality, valid)`` triples.
    """
    from torch.utils.data import IterableDataset, get_worker_info

    if not shard_paths:
        raise ValueError("build_gz_desi_hf_dataset requires at least one shard path")

    class _GZDesiHFDataset(IterableDataset[Any]):
        def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
            import torch

            worker_info = get_worker_info()
            if worker_info is None:
                worker_paths = list(shard_paths)
                rng_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
            else:
                worker_paths = list(shard_paths)[worker_info.id :: worker_info.num_workers]
                rng_seed = (seed or 0) + worker_info.id
            rng = random.Random(rng_seed)
            if shardshuffle:
                rng.shuffle(worker_paths)

            buffer: list[tuple[Tensor, Tensor, Tensor]] = []
            for shard in worker_paths:
                for img, hf_labels in _iter_samples_from_shard(shard):
                    if not has_any_dr8_votes(hf_labels):
                        continue
                    canonical = hf_labels_to_canonical(hf_labels)
                    plurality_list, valid_list = extract_plurality_labels(
                        canonical, min_votes=min_votes
                    )
                    image_tensor = transform(img)
                    plurality = torch.tensor(plurality_list, dtype=torch.long)
                    valid = torch.tensor(valid_list, dtype=torch.bool)

                    triple = (image_tensor, plurality, valid)
                    if len(buffer) < shuffle_buffer:
                        buffer.append(triple)
                    else:
                        idx = rng.randrange(0, len(buffer))
                        yield buffer[idx]
                        buffer[idx] = triple
            rng.shuffle(buffer)
            yield from buffer

    return _GZDesiHFDataset()


def collate_multi_question(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Stack a list of (image, plurality, valid) triples into batched tensors."""
    import torch

    images = torch.stack([item[0] for item in batch], dim=0)
    plurality = torch.stack([item[1] for item in batch], dim=0)
    valid = torch.stack([item[2] for item in batch], dim=0)
    assert plurality.shape[1] == NUM_QUESTIONS
    return images, plurality, valid


def build_gz_desi_hf_dataset_for_dirichlet(
    shard_paths: list[Path],
    transform: Callable[[PILImage], Tensor],
    *,
    shuffle_buffer: int = 1000,
    shardshuffle: bool = True,
    min_votes: int = DEFAULT_MIN_VOTES,
    seed: int | None = None,
) -> Any:
    """T3.6 variant: yields ``(image, counts, valid)`` for the Dirichlet trainer.

    Same shard-walk + DR8-only filter + worker-stride logic as
    :func:`build_gz_desi_hf_dataset`; only the per-sample yield differs.
    ``counts`` is the flat ``(num_answers,)`` int64 vote-count vector
    aligned with the head's 34-D alpha output.
    """
    from torch.utils.data import IterableDataset, get_worker_info

    if not shard_paths:
        raise ValueError(
            "build_gz_desi_hf_dataset_for_dirichlet requires at least one shard path"
        )

    class _GZDesiHFDirichletDataset(IterableDataset[Any]):
        def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
            import torch

            worker_info = get_worker_info()
            if worker_info is None:
                worker_paths = list(shard_paths)
                rng_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
            else:
                worker_paths = list(shard_paths)[worker_info.id :: worker_info.num_workers]
                rng_seed = (seed or 0) + worker_info.id
            rng = random.Random(rng_seed)
            if shardshuffle:
                rng.shuffle(worker_paths)

            buffer: list[tuple[Tensor, Tensor, Tensor]] = []
            for shard in worker_paths:
                for img, hf_labels in _iter_samples_from_shard(shard):
                    if not has_any_dr8_votes(hf_labels):
                        continue
                    canonical = hf_labels_to_canonical(hf_labels)
                    counts_list = _extract_count_vector(canonical)
                    valid_list = _extract_valid_mask(canonical, min_votes=min_votes)
                    image_tensor = transform(img)
                    counts = torch.tensor(counts_list, dtype=torch.long)
                    valid = torch.tensor(valid_list, dtype=torch.bool)

                    triple = (image_tensor, counts, valid)
                    if len(buffer) < shuffle_buffer:
                        buffer.append(triple)
                    else:
                        idx = rng.randrange(0, len(buffer))
                        yield buffer[idx]
                        buffer[idx] = triple
            rng.shuffle(buffer)
            yield from buffer

    return _GZDesiHFDirichletDataset()


def collate_dirichlet(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Stack a list of (image, counts, valid) triples into batched tensors."""
    import torch

    images = torch.stack([item[0] for item in batch], dim=0)
    counts = torch.stack([item[1] for item in batch], dim=0)
    valid = torch.stack([item[2] for item in batch], dim=0)
    assert counts.shape[1] == NUM_ANSWERS, (
        f"counts shape mismatch: got {counts.shape[1]}, expected {NUM_ANSWERS}"
    )
    return images, counts, valid
