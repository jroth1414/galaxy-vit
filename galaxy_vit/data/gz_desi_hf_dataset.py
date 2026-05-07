"""IterableDataset bridging HF gz_desi_wds shards into the multi-question trainer.

Each iterated sample is a ``(image_tensor, plurality_tensor, valid_tensor)``
triple ready for the trainer's batch loop:

* ``image_tensor``: ``(3, H, W)`` float32 from the project's eval/train
  transform (T1.3) applied to a 300x300 PIL DECaLS cutout.
* ``plurality_tensor``: ``(num_questions,)`` int64, the argmax answer
  per question (output of
  :func:`galaxy_vit.data.gz_desi_labels.extract_plurality_labels`).
* ``valid_tensor``: ``(num_questions,)`` bool, ``True`` where the
  galaxy received >= ``min_votes`` DR8 votes for that question.

Filters galaxies that have no DR8 votes (those were classified under a
different GZ campaign — DR5 or DR12 — and don't belong in a W+23
reproduction). Multi-worker safe via ``torch.utils.data.get_worker_info``.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from galaxy_vit.data.gz_desi_hf import has_any_dr8_votes, hf_labels_to_canonical
from galaxy_vit.data.gz_desi_labels import (
    DEFAULT_MIN_VOTES,
    NUM_QUESTIONS,
    extract_plurality_labels,
)
from galaxy_vit.data.gz_desi_streaming import _iter_samples_from_shard

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
