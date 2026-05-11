"""T4.1 - Active-learning experiment driver.

Runs the cached-feature, head-only active-learning loop across 3 seeds
and 3 acquisitions (entropy, BALD, random), writes the curves +
metrics summary to ``artifacts/active_learning_metrics.json`` and a
visualization to ``artifacts/active_learning_curves.png``.

Acceptance gate (DEVPLAN T4.1): entropy acquisition reaches 90% of the
full-data MAE in <= 60% of labels across all 3 seeds. Checked
post-script by ``tests/test_active_learning.py``.

Strategy
--------

1. Load the T3.6 best.pt (Dirichlet head + Zoobot encoder).
2. Iterate the val + test loaders ONCE through the frozen encoder;
   cache the 640-d penultimate features per galaxy. The val pool
   becomes the "unlabeled" pool we'll acquire from; the test set
   stays held-out for MAE evaluation.
3. For each (seed, method) pair:
   - Initialize labeled subset with 5% of the pool (random).
   - For each AL round: fit a FRESH head on the labeled features (no
     encoder updates), evaluate on test, acquire next batch from the
     unlabeled pool by ``method``.
   - Record per-round (n_labeled, fraction, test MAE, test coverage).
4. Compute the full-data MAE (head trained on the entire pool) once,
   shared across seeds.
5. Plot mae-vs-fraction curves (3 acquisitions, mean +/- std across
   seeds).

Invocation::

    python -m scripts.run_active_learning \\
        --config configs/m3_dirichlet.yaml \\
        --checkpoint runs/m3_dirichlet/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from galaxy_vit.config import Settings
from galaxy_vit.data.gz_desi_hf_dataset import (
    build_gz_desi_hf_dataset_for_dirichlet,
    collate_dirichlet,
)
from galaxy_vit.data.gz_desi_labels import question_index_groups
from galaxy_vit.data.transforms import build_eval_transform, load_normalization
from galaxy_vit.inference.posterior import credible_interval
from galaxy_vit.losses.dirichlet_mn import (
    dirichlet_multinomial_nll,
    expected_fractions,
)
from galaxy_vit.models.dirichlet_head import build_dirichlet_head, build_zoobot_dirichlet
from galaxy_vit.training.active_learning import (
    ALRound,
    run_active_learning_loop,
)
from galaxy_vit.training.dirichlet_trainer import DirichletConfig
from galaxy_vit.training.multi_question_trainer import _resolve_shards

DEFAULT_CONFIG = Path("configs/m3_dirichlet.yaml")
DEFAULT_CHECKPOINT = Path("runs/m3_dirichlet/best.pt")
DEFAULT_OUT_DIR = Path("artifacts")
N_ROUNDS = 9                  # 10 evaluations: round 0 (5%) + 9 acquisitions
INIT_FRACTION = 0.05          # 5% labeled at round 0
HEAD_TRAIN_STEPS = 200        # head fits in ~10s on cached features
HEAD_LR = 5.0e-3
HEAD_WEIGHT_DECAY = 1.0e-4
SEEDS = (42, 43, 44)
METHODS: tuple[str, ...] = ("entropy", "bald", "random")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def _cache_features(
    encoder: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode an entire loader once; return (features, counts, valid) on CPU."""
    encoder.eval()
    feats: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []
    for x, c, v in loader:
        x = x.to(device, non_blocking=True)
        f = encoder(x).float().cpu()
        feats.append(f)
        counts.append(c.cpu())
        valids.append(v.cpu())
    return torch.cat(feats), torch.cat(counts), torch.cat(valids)


def _train_head(
    head: torch.nn.Module,
    features: torch.Tensor,
    counts: torch.Tensor,
    valid: torch.Tensor,
    question_groups: list[tuple[str, int, int]],
) -> None:
    """Fit the head in-place on cached features. Pure CPU, ~10s for 200 steps."""
    head.train()
    optim = torch.optim.Adam(head.parameters(), lr=HEAD_LR, weight_decay=HEAD_WEIGHT_DECAY)
    counts_f = counts.float()
    n = features.shape[0]
    batch_size = min(256, max(32, n))
    for _step in range(HEAD_TRAIN_STEPS):
        # Random mini-batch.
        idx = torch.randint(0, n, (batch_size,))
        f = features[idx]
        c = counts_f[idx]
        v = valid[idx]
        alpha = head(f)
        loss = dirichlet_multinomial_nll(
            alpha, c, v, question_groups=question_groups
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optim.step()


def _evaluate_head(
    head: torch.nn.Module,
    features: torch.Tensor,
    counts: torch.Tensor,
    valid: torch.Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
) -> tuple[float, float]:
    """Return (mae_macro, coverage_macro) on the given features."""
    head.eval()
    with torch.no_grad():
        alpha = head(features).float()
        pred = expected_fractions(alpha, question_groups=question_groups)
        obs = expected_fractions(counts.float(), question_groups=question_groups)
        lower, upper = credible_interval(
            alpha, question_groups=question_groups, ci=0.95
        )
    mae_per_q: list[float] = []
    cov_per_q: list[float] = []
    for q_idx, (_q, start, end) in enumerate(question_groups):
        q_valid = valid[:, q_idx]
        if not q_valid.any():
            continue
        mae = float((pred[q_valid, start:end] - obs[q_valid, start:end]).abs().mean().item())
        inside = (
            (obs[q_valid, start:end] >= lower[q_valid, start:end])
            & (obs[q_valid, start:end] <= upper[q_valid, start:end])
        ).float()
        cov = float(inside.mean().item())
        mae_per_q.append(mae)
        cov_per_q.append(cov)
    mae_macro = sum(mae_per_q) / max(1, len(mae_per_q))
    cov_macro = sum(cov_per_q) / max(1, len(cov_per_q))
    return mae_macro, cov_macro


def _save_curves_figure(
    payload: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot mae-vs-fraction-labeled curves (mean +/- std across seeds)."""
    import matplotlib.pyplot as plt

    methods = list(payload["seeds"][next(iter(payload["seeds"]))].keys())
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=140)
    colors = {"entropy": "#0072B2", "bald": "#009E73", "random": "#666666"}
    for method in methods:
        # Stack the per-seed curves by round.
        seeds = list(payload["seeds"].keys())
        fractions = [r["fraction_labeled"] for r in payload["seeds"][seeds[0]][method]["curve"]]
        maes = []
        for s in seeds:
            seed_curve = [r["test_mae_macro"] for r in payload["seeds"][s][method]["curve"]]
            maes.append(seed_curve)
        import numpy as np
        m = np.array(maes)
        mean = m.mean(axis=0)
        std = m.std(axis=0)
        c = colors.get(method, "#222222")
        ax.plot(fractions, mean, color=c, label=method, linewidth=2)
        ax.fill_between(fractions, mean - std, mean + std, color=c, alpha=0.18)
    full = float(payload["full_data_mae"])
    target = full / 0.9
    ax.axhline(full, color="#888888", linestyle=":", linewidth=1.0, label=f"full-data MAE = {full:.4f}")
    ax.axhline(target, color="#CC79A7", linestyle="--", linewidth=1.0,
               label=f"T4.1 target MAE = {target:.4f}")
    ax.axvline(0.60, color="#CC79A7", linestyle="--", linewidth=1.0)
    ax.set_xlabel("fraction of pool labeled")
    ax.set_ylabel("test MAE (macro, per-answer fractions)")
    ax.set_title(
        f"Active learning curves (n_pool={payload['n_pool']}, "
        f"n_test={payload['n_test']}, seeds={len(payload['seeds'])})"
    )
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--n-rounds", type=int, default=N_ROUNDS)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    cfg = DirichletConfig.from_yaml(args.config)
    settings = Settings()  # type: ignore[call-arg]
    os.environ["HF_TOKEN"] = settings.HF_TOKEN.get_secret_value()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[AL] checkpoint={args.checkpoint}", flush=True)

    train_paths, val_paths, test_paths = _resolve_shards(cfg.data.shards)
    _ = train_paths
    print(f"[AL] val_shards={len(val_paths)}  test_shards={len(test_paths)}", flush=True)

    mean, std = load_normalization(cfg.data.normalization)
    eval_tf = build_eval_transform(image_size=cfg.data.image_size, mean=mean, std=std)
    pool_ds = build_gz_desi_hf_dataset_for_dirichlet(
        val_paths, eval_tf, shuffle_buffer=1, shardshuffle=False,
        min_votes=cfg.data.min_votes,
    )
    test_ds = build_gz_desi_hf_dataset_for_dirichlet(
        test_paths, eval_tf, shuffle_buffer=1, shardshuffle=False,
        min_votes=cfg.data.min_votes,
    )
    pool_loader: DataLoader[Any] = DataLoader(
        pool_ds, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers,
        collate_fn=collate_dirichlet,
    )
    test_loader: DataLoader[Any] = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers,
        collate_fn=collate_dirichlet,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AL] device={device}", flush=True)
    model, encoder, _head = build_zoobot_dirichlet(
        num_answers=cfg.model.num_answers,
        alpha_floor=cfg.model.alpha_floor,
        encoder_id=cfg.model.encoder,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    encoder = encoder.to(device)
    feat_dim = getattr(encoder, "num_features", 640)

    print("[AL] caching pool features...", flush=True)
    pool_feats, pool_counts, pool_valid = _cache_features(encoder, pool_loader, device=device)
    print(f"[AL] pool: {pool_feats.shape[0]} galaxies (dim={pool_feats.shape[1]})", flush=True)

    print("[AL] caching test features...", flush=True)
    test_feats, test_counts, test_valid = _cache_features(encoder, test_loader, device=device)
    print(f"[AL] test: {test_feats.shape[0]} galaxies", flush=True)

    groups = question_index_groups()

    # ---- Full-data MAE baseline (head trained on entire pool) ----
    print("[AL] computing full-data MAE baseline...", flush=True)
    full_maes: list[float] = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        head = build_dirichlet_head(
            feat_dim, num_answers=cfg.model.num_answers, alpha_floor=cfg.model.alpha_floor
        )
        _train_head(head, pool_feats, pool_counts, pool_valid, groups)
        mae, _cov = _evaluate_head(
            head, test_feats, test_counts, test_valid, question_groups=groups
        )
        full_maes.append(mae)
        print(f"[AL]   seed={seed} full-data test MAE = {mae:.4f}", flush=True)
    full_data_mae = sum(full_maes) / len(full_maes)
    print(f"[AL] full-data MAE (mean across {len(args.seeds)} seeds) = {full_data_mae:.4f}", flush=True)

    # ---- AL loops ----
    per_seed_results: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in args.seeds:
        per_seed_results[str(seed)] = {}
        for method in METHODS:
            print(f"[AL] === seed={seed}  method={method} ===", flush=True)

            def head_factory() -> torch.nn.Module:
                return build_dirichlet_head(
                    feat_dim,
                    num_answers=cfg.model.num_answers,
                    alpha_floor=cfg.model.alpha_floor,
                )

            torch.manual_seed(seed)
            history = run_active_learning_loop(
                pool_feats, pool_counts, pool_valid,
                test_feats, test_counts, test_valid,
                question_groups=groups,
                head_factory=head_factory,
                train_head_fn=_train_head,
                method=method,
                n_rounds=args.n_rounds,
                init_fraction=INIT_FRACTION,
                seed=seed,
            )
            for r in history:
                print(
                    f"[AL]   n={r.n_labeled:>5d} ({r.fraction_labeled:5.2%})  "
                    f"MAE={r.test_mae_macro:.4f}  cov={r.test_coverage_macro:.4f}",
                    flush=True,
                )
            per_seed_results[str(seed)][method] = {
                "curve": [r.__dict__ for r in history],
            }

    payload = {
        "task": "T4.1-active-learning",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": str(device),
        "n_pool": int(pool_feats.shape[0]),
        "n_test": int(test_feats.shape[0]),
        "feature_dim": int(pool_feats.shape[1]),
        "init_fraction": INIT_FRACTION,
        "n_rounds": args.n_rounds,
        "head_train_steps": HEAD_TRAIN_STEPS,
        "head_lr": HEAD_LR,
        "head_weight_decay": HEAD_WEIGHT_DECAY,
        "seeds": per_seed_results,
        "full_data_mae": full_data_mae,
        "full_data_mae_per_seed": full_maes,
    }
    out_json = args.out_dir / "active_learning_metrics.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[AL] wrote {out_json}", flush=True)

    out_png = args.out_dir / "active_learning_curves.png"
    _save_curves_figure(payload, out_png)
    print(f"[AL] wrote {out_png}", flush=True)

    # Suppress unused-import warning on the ALRound symbol kept for clarity.
    _ = ALRound
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
