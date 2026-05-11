"""Streaming dataset over GZ DESI volunteer-vote tar shards (T2.2).

Each tar shard contains paired ``<key>.jpg`` + ``<key>.json`` entries,
where the JSON sidecar carries the per-galaxy vote counts, the
per-question totals, and the ``dr8_id`` provenance. The shards are
written by ``scripts/build_gz_desi_shards.py`` (one-shot pre-fetch
via the Legacy Survey cutout API) and consumed by
:class:`GZDesiShardDataset` at training time.

Implementation note — we use a small stdlib-based ``IterableDataset``
rather than ``webdataset`` directly. ``webdataset.WebDataset`` and its
``tarfile_to_samples`` pipeline both route every shard through
``webdataset.gopen``, which mis-parses Windows absolute paths
(``urllib.parse.urlparse('C:\\foo').scheme == 'c'``). The custom
loader avoids that, supports multi-worker shard sharding via
``torch.utils.data.get_worker_info``, and is small enough to read in
one pass. Same shape contract:

  * yields ``(PIL.Image, dict)`` tuples per sample
  * shard-level shuffle on iteration start
  * sample-level reservoir shuffle with ``shuffle_buffer`` slots

T5.1 will move shards to HF Hub; at that point a thin gopen-based
fallback can be added without changing the user-facing API.

This module also provides :func:`write_synthetic_shards` which packs
deterministic random images + realistic vote-count metadata into tar
shards on disk. The T2.2 throughput benchmark and the unit tests
use it so they run hermetically — no network, no real DECaLS images
required.
"""

from __future__ import annotations

import io
import json
import random
import tarfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from galaxy_vit.data.gz_desi import (
    GZ_DESI_QUESTIONS,
    expected_vote_count_columns,
    expected_vote_total_columns,
)

if TYPE_CHECKING:  # pragma: no cover — typing-only
    from PIL.Image import Image as PILImage

DEFAULT_SHARD_PREFIX = "gz_desi_shard"
DEFAULT_IMAGE_SIZE = 256


def _shard_paths(shard_dir: Path, prefix: str = DEFAULT_SHARD_PREFIX) -> list[Path]:
    """Sorted list of tar shard paths matching ``<prefix>_*.tar``."""
    return sorted(shard_dir.glob(f"{prefix}_*.tar"))


def _iter_samples_from_shard(
    shard_path: Path,
) -> Iterator[tuple[PILImage, dict[str, Any]]]:
    """Yield decoded ``(image, metadata)`` pairs from a single tar shard.

    Tolerates two key conventions:

    * Our synthetic shards: ``<key>.jpg`` + ``<key>.json``.
    * HF mwalmsley/gz_desi_wds: ``<key>.image.jpg`` + ``<key>.labels.json``.

    Grouping is by ``name.partition(".")`` (first period), so HF's
    multi-period names land on the bare ``<key>`` and the per-sample
    extension dict gets ``"image.jpg"`` / ``"labels.json"`` keys. The
    decoder finds the right entries via suffix matching rather than
    insisting on exact key names.
    """
    from PIL import Image

    by_key: dict[str, dict[str, bytes]] = {}
    last_key: str | None = None
    with tarfile.open(shard_path, "r") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            key, _, ext = name.partition(".")
            if last_key is not None and key != last_key and last_key in by_key:
                files = by_key.pop(last_key)
                yield _decode(files, Image)
            last_key = key
            stream = tf.extractfile(member)
            if stream is None:
                continue
            by_key.setdefault(key, {})[ext.lower()] = stream.read()
    if last_key is not None and last_key in by_key:
        yield _decode(by_key.pop(last_key), Image)


def _decode(
    files: dict[str, bytes],
    image_cls: Any,
) -> tuple[PILImage, dict[str, Any]]:
    img_key = next((k for k in files if k.endswith("jpg") or k.endswith("jpeg")), None)
    json_key = next((k for k in files if k.endswith("json")), None)
    if img_key is None or json_key is None:
        raise ValueError(
            f"shard sample missing image / json (got keys {sorted(files.keys())})"
        )
    img = image_cls.open(io.BytesIO(files[img_key])).convert("RGB")
    metadata = json.loads(files[json_key].decode("utf-8"))
    return img, metadata


def build_gz_desi_dataset(
    shard_dir: Path,
    *,
    shard_prefix: str = DEFAULT_SHARD_PREFIX,
    shuffle_buffer: int = 1000,
    shardshuffle: bool = True,
    seed: int | None = None,
) -> Any:
    """Construct a torch ``IterableDataset`` over the local GZ DESI tar shards.

    Yields decoded ``(PIL.Image, dict)`` tuples. Multi-worker safe —
    each ``DataLoader`` worker gets a disjoint subset of shards via
    ``get_worker_info``.

    Raises ``FileNotFoundError`` if ``shard_dir`` contains no shards.
    """
    from torch.utils.data import IterableDataset, get_worker_info

    paths = _shard_paths(shard_dir, prefix=shard_prefix)
    if not paths:
        raise FileNotFoundError(
            f"no shards matching {shard_prefix}_*.tar in {shard_dir}; "
            f"run `scripts/build_gz_desi_shards.py` first"
        )

    class _GZDesiShardDataset(IterableDataset[Any]):
        def __iter__(self) -> Iterator[tuple[PILImage, dict[str, Any]]]:
            worker_info = get_worker_info()
            if worker_info is None:
                worker_paths = list(paths)
                rng_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
            else:
                # Stride-shard workers so each sees disjoint tars.
                worker_paths = list(paths)[worker_info.id :: worker_info.num_workers]
                rng_seed = (seed or 0) + worker_info.id
            rng = random.Random(rng_seed)
            if shardshuffle:
                rng.shuffle(worker_paths)

            buffer: list[tuple[PILImage, dict[str, Any]]] = []
            for shard in worker_paths:
                for sample in _iter_samples_from_shard(shard):
                    if len(buffer) < shuffle_buffer:
                        buffer.append(sample)
                    else:
                        idx = rng.randrange(0, len(buffer))
                        yield buffer[idx]
                        buffer[idx] = sample
            rng.shuffle(buffer)
            yield from buffer

    return _GZDesiShardDataset()


def write_synthetic_shards(
    shard_dir: Path,
    *,
    n_shards: int = 4,
    samples_per_shard: int = 3000,
    image_size: int = DEFAULT_IMAGE_SIZE,
    seed: int = 42,
    shard_prefix: str = DEFAULT_SHARD_PREFIX,
    hf_dr8_format: bool = False,
) -> list[Path]:
    """Write deterministic synthetic shards for tests + benchmarks.

    Each shard contains ``samples_per_shard`` (image, metadata) pairs.
    Images are random RGB at ``image_size`` x ``image_size``; metadata
    is a realistic GZ DESI vote-count dict (random integer counts per
    answer, totals consistent within each question). Pixel content
    is deterministic given ``seed``.

    Parameters
    ----------
    hf_dr8_format:
        When False (default), JSON keys use the canonical T2.1 schema
        (``<q>_<a>`` count + ``<q>_total-votes``), matching what
        :func:`build_gz_desi_dataset` consumes.
        When True, the metadata uses the HF ``mwalmsley/gz_desi_wds``
        schema (``<q>-dr8_<a>`` keys, no separate total field), matching
        what :func:`build_gz_desi_hf_dataset_for_dirichlet` and
        :func:`build_gz_desi_hf_dataset` consume after the
        ``has_any_dr8_votes`` filter.

    Returns the list of written shard paths.
    """
    import numpy as np
    from PIL import Image

    shard_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    written: list[Path] = []
    count_cols = expected_vote_count_columns()

    global_idx = 0
    for shard_i in range(n_shards):
        shard_path = shard_dir / f"{shard_prefix}_{shard_i:04d}.tar"
        with tarfile.open(shard_path, "w") as tf:
            for _ in range(samples_per_shard):
                key = f"{global_idx:08d}"
                img_arr = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
                buf = io.BytesIO()
                Image.fromarray(img_arr).save(buf, format="JPEG", quality=85)
                jpg_bytes = buf.getvalue()
                _add_to_tar(tf, f"{key}.jpg", jpg_bytes)

                if hf_dr8_format:
                    metadata = _synthesize_hf_dr8_vote_metadata(rng, key)
                else:
                    metadata = _synthesize_vote_metadata(rng, key, count_cols)
                meta_bytes = json.dumps(metadata).encode("utf-8")
                _add_to_tar(tf, f"{key}.json", meta_bytes)

                global_idx += 1
        written.append(shard_path)
    return written


def _synthesize_hf_dr8_vote_metadata(
    rng: Any,
    key: str,
) -> dict[str, Any]:
    """HF-format variant: keys are ``<q>-dr8_<a>`` with no total-votes field.

    Mirrors the schema ``galaxy_vit.data.gz_desi_hf.hf_labels_to_canonical``
    consumes (after stripping the ``-dr8_`` infix to recover the
    canonical ``<q>_<a>`` form).
    """
    metadata: dict[str, Any] = {"key": key, "dr8_id": f"synthetic_{key}"}
    for question, answers in GZ_DESI_QUESTIONS.items():
        total = int(rng.integers(5, 40))
        weights = rng.dirichlet([1.0] * len(answers))
        floats = weights * total
        counts = [int(x) for x in floats]
        remainder = total - sum(counts)
        order = sorted(range(len(answers)), key=lambda i: -(floats[i] - counts[i]))
        for i in range(remainder):
            counts[order[i]] += 1
        for answer, c in zip(answers, counts, strict=True):
            metadata[f"{question}-dr8_{answer}"] = int(c)
    return metadata


def _add_to_tar(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


def _synthesize_vote_metadata(
    rng: Any,
    key: str,
    count_cols: Iterable[str],
) -> dict[str, Any]:
    """Build a realistic vote-count + total dict matching the catalog schema."""
    metadata: dict[str, Any] = {"key": key, "dr8_id": f"synthetic_{key}"}
    for question, answers in GZ_DESI_QUESTIONS.items():
        total = int(rng.integers(5, 40))
        weights = rng.dirichlet([1.0] * len(answers))
        floats = weights * total
        counts = [int(x) for x in floats]
        remainder = total - sum(counts)
        order = sorted(range(len(answers)), key=lambda i: -(floats[i] - counts[i]))
        for i in range(remainder):
            counts[order[i]] += 1
        for answer, c in zip(answers, counts, strict=True):
            metadata[f"{question}_{answer}"] = int(c)
        metadata[f"{question}_total-votes"] = total
    for col in count_cols:
        if col not in metadata:
            raise RuntimeError(f"synthesizer missed column {col!r}")
    for col in expected_vote_total_columns():
        if col not in metadata:
            raise RuntimeError(f"synthesizer missed column {col!r}")
    return metadata
