"""Unit tests for galaxy_vit.inference.similarity (S-1).

Synthetic-feature tests pinned against analytically computed reference
values. No checkpoint, no parquet, no encoder — runs anywhere torch is
installed (skipped on the [dev]-only CI install).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from galaxy_vit.inference.similarity import (  # noqa: E402
    SimilarityHit,
    SimilarityIndex,
)


def _make_index(n: int = 10, d: int = 8, seed: int = 0) -> SimilarityIndex:
    g = torch.Generator().manual_seed(seed)
    features = torch.randn(n, d, generator=g)
    return SimilarityIndex(features)


def test_topk_by_index_returns_self_first() -> None:
    """Query == cache row i → first hit is i with distance ≈ 0."""
    idx = _make_index()
    for i in (0, 3, 9):
        hits = idx.topk_by_index(i, k=5)
        assert isinstance(hits[0], SimilarityHit)
        assert hits[0].idx == i
        assert hits[0].distance == pytest.approx(0.0, abs=1e-6)


def test_topk_returns_exactly_k_items() -> None:
    idx = _make_index(n=20)
    for k in (1, 5, 20):
        hits = idx.topk_by_index(0, k=k)
        assert len(hits) == k


def test_topk_clamps_to_n_items() -> None:
    idx = _make_index(n=4)
    hits = idx.topk_by_index(0, k=1000)
    assert len(hits) == 4


def test_topk_distances_are_in_valid_range() -> None:
    """Cosine distance for unit-normalised features is in [0, 2]."""
    idx = _make_index()
    hits = idx.topk_by_index(0, k=10)
    for h in hits:
        assert 0.0 <= h.distance <= 2.0
        assert math.isfinite(h.distance)


def test_topk_is_sorted_ascending() -> None:
    """Hits come back sorted by distance ascending (closest first)."""
    idx = _make_index()
    hits = idx.topk_by_index(0, k=10)
    distances = [h.distance for h in hits]
    assert distances == sorted(distances)


def test_topk_is_deterministic_across_calls() -> None:
    """Two queries on the same index produce identical hit lists."""
    idx = _make_index()
    a = idx.topk_by_index(0, k=10)
    b = idx.topk_by_index(0, k=10)
    assert [(h.idx, h.distance) for h in a] == [(h.idx, h.distance) for h in b]


def test_topk_with_orthogonal_query_has_distance_one() -> None:
    """Orthogonal unit vectors → cosine_sim = 0 → distance = 1."""
    # Cache: a single unit vector along x. Query: unit vector along y.
    cache = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    query = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    idx = SimilarityIndex(cache)
    hits = idx.topk(query, k=1)
    assert len(hits) == 1
    assert hits[0].idx == 0
    assert hits[0].distance == pytest.approx(1.0, abs=1e-6)


def test_topk_with_antiparallel_query_has_distance_two() -> None:
    """Anti-parallel unit vectors → cosine_sim = -1 → distance = 2."""
    cache = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    query = torch.tensor([-1.0, 0.0, 0.0], dtype=torch.float32)
    idx = SimilarityIndex(cache)
    hits = idx.topk(query, k=1)
    assert hits[0].distance == pytest.approx(2.0, abs=1e-6)


def test_topk_with_parallel_query_has_distance_zero() -> None:
    """Parallel vectors (same direction, different magnitudes) → distance = 0."""
    cache = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    query = torch.tensor([7.5, 0.0, 0.0], dtype=torch.float32)
    idx = SimilarityIndex(cache)
    hits = idx.topk(query, k=1)
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)


def test_zero_row_does_not_crash() -> None:
    """A zero row in the cache returns distance = 1.0 against any unit query."""
    cache = torch.zeros(3, 4)
    cache[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    idx = SimilarityIndex(cache)
    hits = idx.topk_by_index(0, k=3)
    # The non-zero row is itself; the two zero rows are tied at distance = 1.
    assert hits[0].idx == 0
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)
    for h in hits[1:]:
        assert h.distance == pytest.approx(1.0, abs=1e-6)


def test_index_exposes_dims() -> None:
    features = torch.randn(7, 11)
    idx = SimilarityIndex(features)
    assert idx.n_items == 7
    assert idx.dim == 11


def test_validation_rejects_empty_cache() -> None:
    with pytest.raises(ValueError, match="empty"):
        SimilarityIndex(torch.empty(0))


def test_validation_rejects_3d_features() -> None:
    with pytest.raises(ValueError, match="2-D"):
        SimilarityIndex(torch.zeros(2, 3, 4))


def test_validation_rejects_zero_k() -> None:
    idx = _make_index()
    with pytest.raises(ValueError, match="k must be"):
        idx.topk_by_index(0, k=0)


def test_validation_rejects_out_of_range_idx() -> None:
    idx = _make_index(n=5)
    with pytest.raises(IndexError, match="out of range"):
        idx.topk_by_index(5, k=1)
    with pytest.raises(IndexError, match="out of range"):
        idx.topk_by_index(-1, k=1)


def test_validation_rejects_dim_mismatch() -> None:
    idx = _make_index(n=4, d=8)
    bad_query = torch.zeros(7)
    with pytest.raises(ValueError, match="query dim"):
        idx.topk(bad_query, k=1)


def test_validation_rejects_batch_query() -> None:
    idx = _make_index(n=4, d=8)
    bad_query = torch.zeros(3, 8)  # batch size > 1
    with pytest.raises(ValueError, match="batch size"):
        idx.topk(bad_query, k=1)


def test_from_parquet_round_trip(tmp_path: Path) -> None:
    """Write a parquet, load it back, verify the kNN matches the source."""
    pd = pytest.importorskip("pandas")
    features = torch.randn(6, 5)
    df = pd.DataFrame({"features": features.tolist()})
    out = tmp_path / "feats.parquet"
    df.to_parquet(out, index=False)

    idx = SimilarityIndex.from_parquet(out)
    assert idx.n_items == 6
    assert idx.dim == 5

    # The query from the loaded index should match the query from a
    # freshly constructed index.
    fresh = SimilarityIndex(features)
    a = idx.topk_by_index(0, k=6)
    b = fresh.topk_by_index(0, k=6)
    assert [h.idx for h in a] == [h.idx for h in b]
    for ha, hb in zip(a, b, strict=True):
        assert ha.distance == pytest.approx(hb.distance, abs=1e-5)


def test_from_parquet_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SimilarityIndex.from_parquet(tmp_path / "nope.parquet")


def test_from_parquet_rejects_missing_features_column(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    out = tmp_path / "wrong.parquet"
    pd.DataFrame({"not_features": [[1.0, 2.0]]}).to_parquet(out, index=False)
    with pytest.raises(ValueError, match="features"):
        SimilarityIndex.from_parquet(out)
