"""Cosine-kNN similarity search over cached encoder features (S-1).

Loads the per-row encoder features for the 2,462 UMAP-set test thumbnails
from ``artifacts/test_thumb_features.parquet`` (produced by
``scripts/cache_test_thumb_features.py``) into an in-memory cache, then
serves cosine-distance kNN queries for the live demo's
``/api/similar`` endpoints.

Two query modes are supported:

* ``topk_by_index(idx, k)`` — query feature is the cache row at ``idx``.
  This always returns ``idx`` first with distance 0 (sanity-check
  property used by the API tests). No encoder needed.
* ``topk(query_feature, k)`` — caller supplies a 640-D feature for an
  arbitrary image (typically encoded through the Dirichlet model's
  Zoobot encoder).

Math: features are L2-normalised at index-load time. Cosine similarity
is then the inner product ``q_norm @ cache_norm.T``; cosine distance is
``1 - cosine_sim``. For arbitrary float vectors the distance lies in
``[0, 2]``; for non-negative ReLU/softplus features it lies in
``[0, 1]``. The endpoint contract is the conservative ``[0, 2]``
bound.

The module is heavyweight-dep-free at import time (no torch, no pandas
at module scope) so it can be cheaply unit-tested with synthetic feature
matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing-only
    import torch
    from PIL.Image import Image as PILImage

EPS = 1.0e-12


@dataclass(frozen=True)
class SimilarityHit:
    """One row of a kNN response (idx into the cached thumbnail set + distance)."""

    idx: int
    distance: float


def _l2_normalize_rows(features: torch.Tensor) -> torch.Tensor:
    """Normalize each row of ``features`` to unit L2 length.

    Rows with zero norm are returned unchanged (their inner product
    with any query is 0 → cosine_sim = 0 → distance = 1, which is the
    correct "no information" answer).
    """
    import torch as torch_runtime

    if features.ndim != 2:
        raise ValueError(f"features must be 2-D; got shape {tuple(features.shape)}")
    norms = features.norm(p=2, dim=1, keepdim=True).clamp(min=EPS)
    out = features / norms
    assert isinstance(out, torch_runtime.Tensor)
    return out


def _validate_query(query: torch.Tensor, expected_dim: int) -> torch.Tensor:
    """Coerce a 1-D query to (1, D); validate its dim against the cache."""
    import torch

    if query.ndim == 1:
        query = query.unsqueeze(0)
    if query.ndim != 2:
        raise ValueError(
            f"query must be 1-D or 2-D; got shape {tuple(query.shape)}"
        )
    if query.shape[0] != 1:
        raise ValueError(
            f"query batch size must be 1; got {query.shape[0]}"
        )
    if query.shape[1] != expected_dim:
        raise ValueError(
            f"query dim {query.shape[1]} != cache dim {expected_dim}"
        )
    assert isinstance(query, torch.Tensor)
    return query


class SimilarityIndex:
    """In-memory cosine-kNN over a fixed (N, D) feature matrix.

    The cache is L2-normalised on construction; queries are L2-normalised
    on the fly. Returns lists of :class:`SimilarityHit` sorted ascending
    by distance.

    For the live demo's 2,462-row cache (~6 MB raw), all operations are
    sub-millisecond on CPU, so we don't bother with FAISS or a tree
    structure.
    """

    def __init__(self, features: torch.Tensor) -> None:
        import torch

        if features.numel() == 0:
            raise ValueError("SimilarityIndex received empty feature tensor")
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2-D (N, D); got shape {tuple(features.shape)}"
            )
        self._n, self._dim = int(features.shape[0]), int(features.shape[1])
        self._normalized = _l2_normalize_rows(features.float()).contiguous()
        assert isinstance(self._normalized, torch.Tensor)

    @property
    def n_items(self) -> int:
        return self._n

    @property
    def dim(self) -> int:
        return self._dim

    @classmethod
    def from_parquet(cls, path: Path) -> SimilarityIndex:
        """Load features from a parquet file with a ``features`` column.

        The column must contain fixed-length float arrays (one per row).
        Row order is preserved verbatim — caller is responsible for
        ensuring the cache aligns with whatever idx scheme the API uses.
        """
        import pandas as pd
        import torch

        if not path.is_file():
            raise FileNotFoundError(f"feature parquet not found: {path}")
        df = pd.read_parquet(path)
        if "features" not in df.columns:
            raise ValueError(
                f"feature parquet {path} missing required 'features' column"
            )
        rows = df["features"].tolist()
        if not rows:
            raise ValueError(f"feature parquet {path} has zero rows")
        features = torch.tensor(rows, dtype=torch.float32)
        return cls(features)

    def topk(self, query: torch.Tensor, k: int = 20) -> list[SimilarityHit]:
        """Cosine-distance kNN against the cache for an arbitrary feature.

        Parameters
        ----------
        query:
            ``(D,)`` or ``(1, D)`` float tensor.
        k:
            Number of neighbours to return. Clamped to ``[1, n_items]``.

        Returns
        -------
        ``[SimilarityHit(idx, distance), ...]`` sorted ascending by
        distance. Distance ∈ ``[0, 2]``.
        """
        import torch

        if k <= 0:
            raise ValueError(f"k must be >= 1; got {k}")
        k = min(k, self._n)
        q = _validate_query(query, expected_dim=self._dim).float()
        q_norm = _l2_normalize_rows(q)
        sims = (q_norm @ self._normalized.T).squeeze(0)  # (N,)
        # argsort descending for similarity == argsort ascending for distance.
        topk = torch.topk(sims, k=k, largest=True, sorted=True)
        distances = (1.0 - topk.values).clamp(min=0.0, max=2.0)
        return [
            SimilarityHit(idx=int(i.item()), distance=float(d.item()))
            for i, d in zip(topk.indices, distances, strict=True)
        ]

    def topk_by_index(self, idx: int, k: int = 20) -> list[SimilarityHit]:
        """kNN where the query is the cache row at ``idx``.

        Always returns ``idx`` itself as the first hit (distance ≈ 0).
        Useful for the "find similar to this thumbnail" UI flow.
        """
        if idx < 0 or idx >= self._n:
            raise IndexError(f"idx {idx} out of range [0, {self._n})")
        return self.topk(self._normalized[idx], k=k)


def encode_image_to_feature(
    model: torch.nn.Module,
    transform: object,
    image: PILImage,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Encode a PIL image through ``model.encoder`` to a (1, D) feature.

    Mirrors the same eval transform + ``model.encoder(x)`` pattern T2.4's
    ``extract_umap.py`` and the Dirichlet predictor both use, so cached
    features and live query features sit in identical feature space.
    """
    import torch

    model.eval()
    x: torch.Tensor = transform(image).to(device).unsqueeze(0)  # type: ignore[operator]
    with torch.no_grad():
        feats = model.encoder(x).float().cpu()  # type: ignore[operator]
    assert isinstance(feats, torch.Tensor)
    return feats
