"""S-3 — Precompute top-K outliers per metric for the "interesting galaxies" panel.

Three sort orders are surfaced at runtime via ``/api/outliers``:

* **entropy**       — per-galaxy predictive entropy summed across the
  10 GZ DESI questions (Dirichlet posterior mean).
* **bald**          — per-galaxy BALD score (Houlsby+11) summed across
  questions. Closed-form digamma.
* **disagreement**  — per-galaxy mean L1 distance between
  ``expected_fractions(alpha)`` and the volunteer vote fractions,
  averaged across questions where volunteers actually voted
  (i.e. ``<q>_total-votes > 0``). Galaxies with zero valid questions
  are excluded.

This script reuses ``artifacts/test_thumb_features.parquet`` (from S-1)
so it does NOT re-run the full encoder; it just pushes the 2,462 cached
640-D features through the Dirichlet head, then iterates the test
shards in the SAME deterministic order ``scripts/build_test_thumbs.py``
used to recover the volunteer-vote counts row-by-row.

The output ``artifacts/outliers.json`` is small (top-100 per metric x
~3 metrics) and committed:

    {
      "n_galaxies": 2462,
      "n_with_volunteer_votes": <int>,
      "metrics": {
        "entropy":       [{"idx": 1873, "value": 6.21}, ...],
        "bald":          [{"idx": 1054, "value": 0.83}, ...],
        "disagreement":  [{"idx":  927, "value": 0.49}, ...]
      },
      "median": {"entropy": ..., "bald": ..., "disagreement": ...}
    }

The ``median`` block lets the frontend / acceptance test compare top-K
values against the population median (acceptance criterion: outlier
galaxies have visibly higher uncertainty than the median).

Invocation::

    python -m scripts.build_outlier_indices
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from galaxy_vit.data.gz_desi import GZ_DESI_QUESTIONS
from galaxy_vit.data.gz_desi_hf import has_any_dr8_votes, hf_labels_to_canonical
from galaxy_vit.data.gz_desi_streaming import _iter_samples_from_shard
from galaxy_vit.data.schema import question_index_groups
from galaxy_vit.inference.outliers import (
    bald_total,
    predictive_entropy_total,
    topk_indices,
    volunteer_disagreement,
)
from galaxy_vit.inference.similarity import SimilarityIndex
from galaxy_vit.models.dirichlet_head import build_zoobot_dirichlet
from galaxy_vit.training.dirichlet_trainer import DirichletConfig
from galaxy_vit.training.multi_question_trainer import _resolve_shards

DEFAULT_CONFIG = Path("configs/m3_dirichlet.yaml")
DEFAULT_CKPT = Path("runs/m3_dirichlet/best.pt")
DEFAULT_FEATURES = Path("artifacts/test_thumb_features.parquet")
DEFAULT_OUT = Path("artifacts/outliers.json")
DEFAULT_TOP_K = 100


def _alpha_from_features(
    ckpt_path: Path, features: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    """Run cached 640-D features through the Dirichlet head only.

    Loads the full model, copies the head's weights, then discards the
    encoder. Faster (no convolutional forward pass) and cleaner than
    rebuilding the head module manually.
    """
    model, _encoder, head = build_zoobot_dirichlet()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    head = head.to(device)
    head.eval()
    with torch.no_grad():
        alpha = head(features.to(device)).float().cpu()
    assert isinstance(alpha, torch.Tensor)
    return alpha


def _gather_vote_counts(
    *,
    test_paths: list[Path],
    min_votes: int,
    expected_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-iterate the test shards to collect per-thumb vote counts + valid mask.

    Filter chain is byte-identical to ``scripts/build_test_thumbs.py``:
    DR8 votes filter + ``smooth-or-featured`` total >= ``min_votes``.
    Yields ``(counts (N, 34) int64, valid (N, Q) bool)`` where ``Q`` is
    the number of GZ DESI questions (10).
    """
    groups = question_index_groups()
    num_answers = sum(end - start for _, start, end in groups)
    num_questions = len(groups)

    sof_answers = GZ_DESI_QUESTIONS["smooth-or-featured"]
    sof_keys = [f"smooth-or-featured_{a}" for a in sof_answers]

    counts = torch.zeros((expected_n, num_answers), dtype=torch.int64)
    valid = torch.zeros((expected_n, num_questions), dtype=torch.bool)
    row_idx = 0
    for shard in test_paths:
        for _img, hf_labels in _iter_samples_from_shard(shard):
            if not has_any_dr8_votes(hf_labels):
                continue
            canonical = hf_labels_to_canonical(hf_labels)
            sof_total = sum(int(canonical.get(k, 0) or 0) for k in sof_keys)
            if sof_total < min_votes:
                continue
            if row_idx >= expected_n:
                break
            for q_idx, (q_name, start, _end) in enumerate(groups):
                q_total = 0
                for a_idx, a_name in enumerate(GZ_DESI_QUESTIONS[q_name]):
                    c = int(canonical.get(f"{q_name}_{a_name}", 0) or 0)
                    counts[row_idx, start + a_idx] = c
                    q_total += c
                valid[row_idx, q_idx] = q_total > 0
            row_idx += 1
            if row_idx % 500 == 0:
                print(f"[outliers] gathered votes for {row_idx} rows", flush=True)
        if row_idx >= expected_n:
            break
    if row_idx != expected_n:
        raise RuntimeError(
            f"vote-count iteration produced {row_idx} rows; expected "
            f"{expected_n} (cache + shard filters must be in lock-step)"
        )
    return counts, valid


def _per_question_fractions(counts: torch.Tensor) -> torch.Tensor:
    """Convert per-answer counts to per-question fractions.

    For each (galaxy, question), divide the per-answer counts by the
    per-question total. Questions with zero total stay zero (no
    division by zero).
    """
    groups = question_index_groups()
    fracs = torch.zeros_like(counts, dtype=torch.float32)
    for _q_name, start, end in groups:
        slice_counts = counts[:, start:end].float()
        denom = slice_counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
        # Avoid 0-division by checking total > 0 mask explicitly so
        # zero-total questions remain zero (clamp would otherwise let a
        # tiny "1.0" denominator leak in).
        nonzero = slice_counts.sum(dim=-1, keepdim=True) > 0
        fracs[:, start:end] = torch.where(
            nonzero,
            slice_counts / denom,
            torch.zeros_like(slice_counts),
        )
    return fracs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"[outliers] missing checkpoint: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.features.is_file():
        print(
            f"[outliers] missing feature cache: {args.features}; "
            "run `python -m scripts.cache_test_thumb_features` first",
            file=sys.stderr,
        )
        return 2

    print(f"[outliers] loading cached features from {args.features}", flush=True)
    index = SimilarityIndex.from_parquet(args.features)
    import pandas as pd

    df = pd.read_parquet(args.features)
    features = torch.tensor(df["features"].tolist(), dtype=torch.float32)
    n_galaxies = int(features.shape[0])
    print(
        f"[outliers] {n_galaxies} galaxies x {index.dim}-D features",
        flush=True,
    )

    device = torch.device(args.device)
    print(f"[outliers] device={device}", flush=True)
    print("[outliers] running head on cached features", flush=True)
    alpha = _alpha_from_features(args.checkpoint, features, device=device)
    print(f"[outliers] alpha shape={tuple(alpha.shape)}", flush=True)

    groups = question_index_groups()

    cfg = DirichletConfig.from_yaml(args.config)
    import os

    from galaxy_vit.config import Settings

    settings = Settings()  # type: ignore[call-arg]
    os.environ["HF_TOKEN"] = settings.HF_TOKEN.get_secret_value()
    _, _, test_paths = _resolve_shards(cfg.data.shards)
    print(
        f"[outliers] gathering volunteer votes from {len(test_paths)} test shards",
        flush=True,
    )
    counts, valid = _gather_vote_counts(
        test_paths=test_paths,
        min_votes=cfg.data.min_votes,
        expected_n=n_galaxies,
    )
    vote_fracs = _per_question_fractions(counts)
    n_with_votes = int(valid.any(dim=1).sum().item())
    print(
        f"[outliers] {n_with_votes}/{n_galaxies} galaxies have >= 1 valid question",
        flush=True,
    )

    print("[outliers] scoring metrics", flush=True)
    entropy = predictive_entropy_total(alpha, question_groups=groups)
    bald = bald_total(alpha, question_groups=groups)
    disagreement_raw = volunteer_disagreement(
        alpha, vote_fracs, valid, question_groups=groups
    )
    # Mask out galaxies with zero valid questions from the disagreement
    # ranking entirely (assign -inf so topk never picks them).
    any_valid = valid.any(dim=1)
    disagreement = torch.where(
        any_valid, disagreement_raw, torch.full_like(disagreement_raw, -1.0)
    )

    print(
        "[outliers] medians:"
        f" entropy={float(entropy.median().item()):.3f}"
        f" bald={float(bald.median().item()):.3f}"
        f" disagreement={float(disagreement_raw[any_valid].median().item()):.3f}",
        flush=True,
    )

    payload: dict[str, object] = {
        "n_galaxies": n_galaxies,
        "n_with_volunteer_votes": n_with_votes,
        "top_k": args.top_k,
        "metrics": {
            "entropy": [
                {"idx": e.idx, "value": e.value}
                for e in topk_indices(entropy, k=args.top_k)
            ],
            "bald": [
                {"idx": e.idx, "value": e.value}
                for e in topk_indices(bald, k=args.top_k)
            ],
            "disagreement": [
                {"idx": e.idx, "value": e.value}
                for e in topk_indices(disagreement, k=args.top_k)
                if e.value >= 0.0  # drop the masked-out -1.0 sentinel rows
            ],
        },
        "median": {
            "entropy": float(entropy.median().item()),
            "bald": float(bald.median().item()),
            "disagreement": float(disagreement_raw[any_valid].median().item()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[outliers] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
