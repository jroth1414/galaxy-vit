"""Post-hoc Dirichlet-head temperature calibration (T3.6 phase 2).

Standard fix for the over-concentration failure mode in Dirichlet-head
training (Walmsley+23 calibration appendix; Guo+17 temperature scaling
generalized to Dirichlet). For a learned concentration vector
``alpha`` and a scalar temperature ``T >= 1``:

    alpha_calibrated = alpha / T

This preserves the posterior MEAN exactly (``alpha_i / sum(alpha) =
(alpha_i / T) / (sum(alpha) / T)``) so MAE on per-answer fractions is
unchanged, while shrinking the concentration sum widens each marginal
Beta posterior — directly raising coverage. T is selected per question
(per-question T_q) OR globally (single T), fitted on the val set to
target coverage = 0.95.

Two sweeps:

* ``single`` — one scalar T fitted on val by maximizing
  ``-(coverage - 0.95)**2`` (pull coverage toward the target).
* ``per_question`` — fit a separate T_q for each of the 10 questions
  by the same criterion.

Reports test-set MAE, coverage, and brier (using
:mod:`galaxy_vit.training.calibration` and
:mod:`galaxy_vit.inference.posterior`) under all three regimes (raw,
single-T, per-question T) so the eventual model-card writeup has the
full Pareto picture.

Outputs ``<run_dir>/calibrated_metrics.json``; the T3.6 acceptance
test reads this file.

Invocation::

    python -m scripts.calibrate_dirichlet \\
        --config configs/m3_dirichlet.yaml \\
        --checkpoint runs/m3_dirichlet/best.pt \\
        --out runs/m3_dirichlet/calibrated_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
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
from galaxy_vit.losses.dirichlet_mn import expected_fractions
from galaxy_vit.models.dirichlet_head import build_zoobot_dirichlet
from galaxy_vit.training.dirichlet_trainer import DirichletConfig
from galaxy_vit.training.multi_question_trainer import (
    MQConfig as _BaseConfig,  # only for shard helper
)
from galaxy_vit.training.multi_question_trainer import _resolve_shards

# Temperature grid: log-spaced 1.0 .. 60. Catches both small (1-3) and
# large (10-60) corrections to the over-concentration we observed in v1.
T_GRID: tuple[float, ...] = (
    1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
    12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0,
)
TARGET_COVERAGE = 0.95


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Iterate a loader, return concatenated (alpha, counts, valid) tensors on CPU.

    Memory: ~2.5k galaxies x 34 floats x 4 bytes ≈ 340 KB per split.
    """
    model.eval()
    alphas: list[torch.Tensor] = []
    counts_list: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []
    for x, counts, valid in loader:
        x = x.to(device, non_blocking=True)
        alpha = model(pixel_values=x).alpha.float().cpu()
        alphas.append(alpha)
        counts_list.append(counts.cpu())
        valids.append(valid.cpu())
    if not alphas:
        raise RuntimeError("no batches yielded by loader; check shard paths + min_votes")
    return torch.cat(alphas), torch.cat(counts_list), torch.cat(valids)


def _per_question_stats_at_T(
    alpha: torch.Tensor,
    counts: torch.Tensor,
    valid: torch.Tensor,
    *,
    T: torch.Tensor | float,
    question_groups: list[tuple[str, int, int]],
    ci: float = 0.95,
) -> dict[str, dict[str, float]]:
    """Compute per-question MAE + coverage at a given temperature.

    ``T`` is either a scalar or a per-question tensor of shape
    ``(num_questions,)``. For per-question T, each question's slice
    of ``alpha`` is scaled by its own T_q.
    """
    if isinstance(T, float | int):
        alpha_T = alpha / float(T)
    else:
        # Per-question: broadcast T_q across the answer slice for question q.
        scaled = torch.empty_like(alpha)
        for q_idx, (_q, start, end) in enumerate(question_groups):
            scaled[:, start:end] = alpha[:, start:end] / float(T[q_idx].item())
        alpha_T = scaled

    pred_fracs = expected_fractions(alpha_T, question_groups=question_groups)
    obs_fracs = expected_fractions(counts.float(), question_groups=question_groups)
    lower, upper = credible_interval(alpha_T, question_groups=question_groups, ci=ci)

    out: dict[str, dict[str, float]] = {}
    for q_idx, (q_name, start, end) in enumerate(question_groups):
        q_valid = valid[:, q_idx]
        n_v = int(q_valid.sum().item())
        if n_v == 0:
            out[q_name] = {"mae": 0.0, "coverage": 0.0, "n_valid": 0}
            continue
        slice_pred = pred_fracs[q_valid, start:end]
        slice_obs = obs_fracs[q_valid, start:end]
        slice_lower = lower[q_valid, start:end]
        slice_upper = upper[q_valid, start:end]
        mae = float((slice_pred - slice_obs).abs().mean().item())
        inside = ((slice_obs >= slice_lower) & (slice_obs <= slice_upper)).float()
        cov_q = float(inside.mean().item())
        out[q_name] = {"mae": mae, "coverage": cov_q, "n_valid": n_v}
    return out


def _macro(stats: dict[str, dict[str, float]], key: str) -> float:
    active = [s[key] for s in stats.values() if s["n_valid"] > 0]
    return float(sum(active) / len(active)) if active else 0.0


def _pick_single_T(
    alpha: torch.Tensor,
    counts: torch.Tensor,
    valid: torch.Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
    ci: float,
) -> tuple[float, list[dict[str, Any]]]:
    """Sweep single-T grid on val; pick T closest to target coverage 0.95."""
    sweep: list[dict[str, Any]] = []
    for T in T_GRID:
        stats = _per_question_stats_at_T(
            alpha, counts, valid, T=T, question_groups=question_groups, ci=ci,
        )
        macro_cov = _macro(stats, "coverage")
        sweep.append(
            {"T": T, "coverage_macro": macro_cov, "mae_macro": _macro(stats, "mae")}
        )
    # Minimize |coverage - target|.
    best = min(sweep, key=lambda r: abs(r["coverage_macro"] - TARGET_COVERAGE))
    return float(best["T"]), sweep


def _pick_per_question_T(
    alpha: torch.Tensor,
    counts: torch.Tensor,
    valid: torch.Tensor,
    *,
    question_groups: list[tuple[str, int, int]],
    ci: float,
) -> tuple[torch.Tensor, dict[str, list[dict[str, float]]]]:
    """Fit T_q per question independently; each picks T closest to target coverage."""
    sweep: dict[str, list[dict[str, float]]] = {}
    best_T = torch.ones(len(question_groups), dtype=torch.float32)
    for q_idx, (q_name, _, _) in enumerate(question_groups):
        rows: list[dict[str, float]] = []
        best_score = float("inf")
        best_T_q = 1.0
        for T in T_GRID:
            stats = _per_question_stats_at_T(
                alpha, counts, valid,
                T=torch.tensor([T if i == q_idx else 1.0 for i in range(len(question_groups))]),
                question_groups=question_groups, ci=ci,
            )
            s = stats[q_name]
            rows.append({"T": T, "coverage": s["coverage"], "mae": s["mae"]})
            score = abs(s["coverage"] - TARGET_COVERAGE)
            if score < best_score:
                best_score = score
                best_T_q = T
        sweep[q_name] = rows
        best_T[q_idx] = best_T_q
    return best_T, sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ci", type=float, default=0.95)
    args = parser.parse_args(argv)

    cfg = DirichletConfig.from_yaml(args.config)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    settings = Settings()  # type: ignore[call-arg]
    os.environ["HF_TOKEN"] = settings.HF_TOKEN.get_secret_value()
    print(f"[cal] checkpoint={args.checkpoint}  config={args.config}", flush=True)

    train_paths, val_paths, test_paths = _resolve_shards(cfg.data.shards)
    print(
        f"[cal] shards: val={len(val_paths)}  test={len(test_paths)}  "
        f"(train ignored)",
        flush=True,
    )
    _ = train_paths  # keep import path warm; train shards not used here

    mean, std = load_normalization(cfg.data.normalization)
    eval_tf = build_eval_transform(image_size=cfg.data.image_size, mean=mean, std=std)
    val_ds = build_gz_desi_hf_dataset_for_dirichlet(
        val_paths, eval_tf, shuffle_buffer=1, shardshuffle=False,
        min_votes=cfg.data.min_votes,
    )
    test_ds = build_gz_desi_hf_dataset_for_dirichlet(
        test_paths, eval_tf, shuffle_buffer=1, shardshuffle=False,
        min_votes=cfg.data.min_votes,
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers,
        collate_fn=collate_dirichlet,
    )
    test_loader: DataLoader[Any] = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers,
        collate_fn=collate_dirichlet,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, _ = build_zoobot_dirichlet(
        num_answers=cfg.model.num_answers,
        alpha_floor=cfg.model.alpha_floor,
        encoder_id=cfg.model.encoder,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    groups = question_index_groups()

    print("[cal] collecting val predictions...", flush=True)
    val_alpha, val_counts, val_valid = _collect_predictions(model, val_loader, device=device)
    print(f"[cal] val: {val_alpha.shape[0]} galaxies", flush=True)

    print("[cal] collecting test predictions...", flush=True)
    test_alpha, test_counts, test_valid = _collect_predictions(model, test_loader, device=device)
    print(f"[cal] test: {test_alpha.shape[0]} galaxies", flush=True)

    # --- raw (T = 1) ---
    raw_test_stats = _per_question_stats_at_T(
        test_alpha, test_counts, test_valid,
        T=1.0, question_groups=groups, ci=args.ci,
    )
    raw_mae = _macro(raw_test_stats, "mae")
    raw_cov = _macro(raw_test_stats, "coverage")
    print(f"[cal] RAW    test: MAE={raw_mae:.4f}  coverage={raw_cov:.4f}", flush=True)

    # --- single-T sweep on val, evaluated on test ---
    print("[cal] sweeping single-T on val...", flush=True)
    best_single_T, single_sweep = _pick_single_T(
        val_alpha, val_counts, val_valid, question_groups=groups, ci=args.ci,
    )
    single_test_stats = _per_question_stats_at_T(
        test_alpha, test_counts, test_valid,
        T=best_single_T, question_groups=groups, ci=args.ci,
    )
    single_mae = _macro(single_test_stats, "mae")
    single_cov = _macro(single_test_stats, "coverage")
    print(
        f"[cal] SINGLE T={best_single_T:.2f}: "
        f"test MAE={single_mae:.4f}  coverage={single_cov:.4f}",
        flush=True,
    )

    # --- per-question T sweep on val, evaluated on test ---
    print("[cal] sweeping per-question T on val...", flush=True)
    per_q_T, per_q_sweep = _pick_per_question_T(
        val_alpha, val_counts, val_valid, question_groups=groups, ci=args.ci,
    )
    per_q_test_stats = _per_question_stats_at_T(
        test_alpha, test_counts, test_valid,
        T=per_q_T, question_groups=groups, ci=args.ci,
    )
    per_q_mae = _macro(per_q_test_stats, "mae")
    per_q_cov = _macro(per_q_test_stats, "coverage")
    print(
        f"[cal] PER-Q  T_q chosen: "
        f"{ {q: float(per_q_T[i].item()) for i, (q, _, _) in enumerate(groups)} }",
        flush=True,
    )
    print(
        f"[cal] PER-Q  test: MAE={per_q_mae:.4f}  coverage={per_q_cov:.4f}",
        flush=True,
    )

    payload = {
        "task": "T3.6-temperature-calibration",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": str(device),
        "ci": args.ci,
        "n_val_galaxies": int(val_alpha.shape[0]),
        "n_test_galaxies": int(test_alpha.shape[0]),
        "raw": {
            "test_mae_macro": raw_mae,
            "test_coverage_macro": raw_cov,
            "per_question": raw_test_stats,
        },
        "single_T": {
            "T": best_single_T,
            "val_sweep": single_sweep,
            "test_mae_macro": single_mae,
            "test_coverage_macro": single_cov,
            "per_question": single_test_stats,
        },
        "per_question_T": {
            "T_per_question": {
                q: float(per_q_T[i].item()) for i, (q, _, _) in enumerate(groups)
            },
            "val_sweep_per_question": per_q_sweep,
            "test_mae_macro": per_q_mae,
            "test_coverage_macro": per_q_cov,
            "per_question": per_q_test_stats,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[cal] wrote {args.out}", flush=True)

    # Suppress mypy unused-import warning on the helper-only import.
    _ = _BaseConfig
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
