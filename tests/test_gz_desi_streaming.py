"""T2.2 — WebDataset streaming pipeline acceptance tests.

Hermetic — uses :func:`write_synthetic_shards` so the throughput
benchmark and the round-trip integration test run without any real
DECaLS images. CI without ``[m1-train]`` (no webdataset / numpy / PIL)
skips the whole module.
"""

from __future__ import annotations

import json
import tarfile
import time
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")
pytest.importorskip("numpy")
pytest.importorskip("webdataset")

from galaxy_vit.data.gz_desi import (
    expected_vote_count_columns,
    expected_vote_total_columns,
)
from galaxy_vit.data.gz_desi_streaming import (
    build_gz_desi_dataset,
    write_synthetic_shards,
)

# DEVPLAN T2.2 acceptance: 10k-sample pass in <120 s on the dev box (5070 Ti).
TARGET_SAMPLES = 10_000
LATENCY_BUDGET_S = 120.0


def test_synthetic_shard_round_trip(tmp_path) -> None:
    """Synthetic-shard helper writes valid tars with the expected schema."""
    paths = write_synthetic_shards(
        tmp_path, n_shards=2, samples_per_shard=10, image_size=64, seed=7
    )
    assert len(paths) == 2

    # Inspect the first shard's contents directly via tarfile.
    members_per_shard = []
    for p in paths:
        with tarfile.open(p, "r") as tf:
            names = tf.getnames()
            members_per_shard.append(names)
            # 10 jpg + 10 json
            assert len(names) == 20
            jsons = [n for n in names if n.endswith(".json")]
            assert len(jsons) == 10
            # Inspect one metadata dict.
            with tf.extractfile(jsons[0]) as fh:  # type: ignore[union-attr]
                payload = json.loads(fh.read().decode("utf-8"))
            assert "dr8_id" in payload
            for col in expected_vote_count_columns():
                assert col in payload
                assert isinstance(payload[col], int)
                assert payload[col] >= 0
            for col in expected_vote_total_columns():
                assert col in payload
                assert isinstance(payload[col], int)
    # Keys are unique across shards (global counter).
    flat = [n for shard in members_per_shard for n in shard]
    assert len(flat) == len(set(flat))


def test_build_gz_desi_dataset_shape(tmp_path) -> None:
    """Iterating the dataset yields decoded (PIL, dict) tuples."""
    write_synthetic_shards(
        tmp_path, n_shards=1, samples_per_shard=20, image_size=32, seed=11
    )
    dataset = build_gz_desi_dataset(
        tmp_path, shuffle_buffer=4, shardshuffle=False
    )

    seen = 0
    for sample in dataset:
        # `to_tuple("jpg", "json")` -> (PIL.Image, dict)
        img, metadata = sample
        # PIL Image, 32x32 RGB.
        assert img.size == (32, 32)
        assert img.mode == "RGB"
        # Metadata has the dr8_id + at least the canonical 34 + 10 columns.
        assert "dr8_id" in metadata
        assert "smooth-or-featured_smooth" in metadata
        assert "merging_total-votes" in metadata
        seen += 1
    assert seen == 20


def test_build_gz_desi_dataset_missing_shards_errors(tmp_path) -> None:
    """Helpful error when no shards match the prefix."""
    with pytest.raises(FileNotFoundError, match="no shards matching"):
        build_gz_desi_dataset(tmp_path)


def test_T2_2_throughput_under_120s(tmp_path) -> None:
    """T2.2 acceptance: 10k-sample pass through the loader in <120 s.

    Uses synthetic 256x256 RGB images. The DataLoader batches with a
    no-op ``collate_fn`` (just returns the list of (PIL, dict) tuples)
    so we measure raw streaming + JPEG decode throughput, not a torch
    transform pipeline. T1.4's trainer attaches a real torchvision
    transform to convert PIL -> Tensor; that's the measurement we want
    later, not here.
    """
    from torch.utils.data import DataLoader

    write_synthetic_shards(
        tmp_path,
        n_shards=4,
        samples_per_shard=3_000,
        image_size=256,
        seed=42,
    )

    dataset = build_gz_desi_dataset(
        tmp_path, shuffle_buffer=128, shardshuffle=False
    )

    # Custom collate keeps PIL Images as-is (default collate can only
    # batch Tensors/dicts/numbers) and avoids the trainer-side transform
    # so we measure raw streaming throughput.
    def _identity_collate(batch: list[Any]) -> list[Any]:
        return batch

    loader: DataLoader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        collate_fn=_identity_collate,
    )

    n_seen = 0
    t0 = time.perf_counter()
    for batch in loader:
        n_seen += len(batch)
        if n_seen >= TARGET_SAMPLES:
            break
    elapsed = time.perf_counter() - t0

    print(
        f"[throughput] {n_seen} samples in {elapsed:.1f}s "
        f"({n_seen / elapsed:.0f} samples/s)"
    )
    assert n_seen >= TARGET_SAMPLES, (
        f"only consumed {n_seen} samples; loader didn't yield 10k"
    )
    assert elapsed < LATENCY_BUDGET_S, (
        f"10k-sample pass took {elapsed:.1f}s; target <{LATENCY_BUDGET_S}s "
        f"(throughput {n_seen / elapsed:.0f} samples/s)"
    )
