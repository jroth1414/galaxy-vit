"""T2.5 — Per-question calibration + reliability diagrams.

Loads the W+23 reproduction checkpoint (default
``runs/m2_w23_reproduction/best.pt``), iterates the GZ DESI DR8 test
shards, captures the per-question top-class softmax confidence and the
correctness indicator, and writes:

  * ``artifacts/calibration_<question>.png``    — per-question reliability
                                                   diagram + confidence histogram
  * ``artifacts/reliability_overview.png``       — 2x5 grid of all 10 questions
                                                   for the model card
  * ``artifacts/calibration_metrics.json``       — per-question ECE/MCE/Brier
                                                   + macro stats (committed)

Acceptance gate (T2.5): macro-ECE across the 10 questions must be at
most ``--macro-ece-max`` (default 0.10). T3.4 will compare its
Dirichlet-Multinomial head's calibration against this baseline.

Invocation::

    python -m scripts.extract_calibration \\
        --config configs/m2_w23_reproduction.yaml \\
        --checkpoint runs/m2_w23_reproduction/best.pt
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
    build_gz_desi_hf_dataset,
    collate_multi_question,
)
from galaxy_vit.data.gz_desi_labels import question_index_groups
from galaxy_vit.data.transforms import build_eval_transform, load_normalization
from galaxy_vit.models.zoobot_encoder import build_zoobot_finetune
from galaxy_vit.training.calibration import (
    binned_reliability,
    brier_score_topclass,
    expected_calibration_error,
    maximum_calibration_error,
)
from galaxy_vit.training.multi_question_trainer import MQConfig, _resolve_shards

DEFAULT_CONFIG = Path("configs/m2_w23_reproduction.yaml")
DEFAULT_CHECKPOINT = Path("runs/m2_w23_reproduction/best.pt")
DEFAULT_OUT_DIR = Path("artifacts")
DEFAULT_MACRO_ECE_MAX = 0.10
DEFAULT_N_BINS = 10
DEFAULT_MAX_BATCHES: int | None = None


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def _capture_confidence_correct(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run the test loader, return per-question (confidence, correct) tensors.

    For each question, ``confidence[q]`` is the top-class softmax
    probability and ``correct[q]`` is 1.0 when the argmax matches the
    plurality target. Only galaxies with the per-question valid mask
    set are kept.
    """
    model.eval()
    groups = question_index_groups()
    buffers: dict[str, dict[str, list[torch.Tensor]]] = {
        q: {"conf": [], "correct": []} for q, _, _ in groups
    }

    for batch_idx, (x, plurality, valid) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        plurality = plurality.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        logits = model(pixel_values=x).logits  # (B, num_answers)
        for q_idx, (q_name, start, end) in enumerate(groups):
            q_valid = valid[:, q_idx]
            if not q_valid.any():
                continue
            q_logits = logits[q_valid, start:end]
            q_target = plurality[q_valid, q_idx]
            probs = torch.softmax(q_logits.float(), dim=-1)
            top_conf, top_pred = probs.max(dim=-1)
            correct = (top_pred == q_target).float()
            buffers[q_name]["conf"].append(top_conf.cpu())
            buffers[q_name]["correct"].append(correct.cpu())

        if (batch_idx + 1) % 25 == 0:
            print(f"[cal] processed {batch_idx + 1} batches", flush=True)

    out: dict[str, dict[str, torch.Tensor]] = {}
    for q_name, lists in buffers.items():
        if not lists["conf"]:
            out[q_name] = {
                "conf": torch.empty(0, dtype=torch.float32),
                "correct": torch.empty(0, dtype=torch.float32),
            }
            continue
        out[q_name] = {
            "conf": torch.cat(lists["conf"], dim=0),
            "correct": torch.cat(lists["correct"], dim=0),
        }
    return out


def _per_question_metrics(
    captured: dict[str, dict[str, torch.Tensor]],
    *,
    n_bins: int,
) -> dict[str, dict[str, Any]]:
    """Compute reliability + ECE/MCE/Brier per question."""
    out: dict[str, dict[str, Any]] = {}
    for q_name, t in captured.items():
        conf = t["conf"]
        correct = t["correct"]
        n = int(conf.numel())
        if n == 0:
            out[q_name] = {
                "n_valid": 0,
                "ece": float("nan"),
                "mce": float("nan"),
                "brier": float("nan"),
                "mean_confidence": float("nan"),
                "accuracy": float("nan"),
                "reliability": None,
            }
            continue
        ece = expected_calibration_error(conf, correct, n_bins=n_bins)
        mce = maximum_calibration_error(conf, correct, n_bins=n_bins)
        brier = brier_score_topclass(conf, correct)
        rel = binned_reliability(conf, correct, n_bins=n_bins)
        out[q_name] = {
            "n_valid": n,
            "ece": ece,
            "mce": mce,
            "brier": brier,
            "mean_confidence": float(conf.mean().item()),
            "accuracy": float(correct.mean().item()),
            "reliability": rel,
        }
    return out


def _macro_ece(per_q: dict[str, dict[str, Any]]) -> float:
    active = [v["ece"] for v in per_q.values() if v["n_valid"] > 0]
    return float(sum(active) / len(active)) if active else float("nan")


def _plot_reliability_panel(
    ax: Any,
    rel: dict[str, list[float]],
    *,
    q_name: str,
    ece: float,
    n_valid: int,
) -> None:
    """Single reliability panel: confidence x-axis, accuracy y-axis, perfect-cal diagonal."""
    centers = [(lo + hi) / 2 for lo, hi in zip(rel["bin_lower"], rel["bin_upper"], strict=True)]
    counts = rel["count"]
    accuracy = rel["accuracy"]
    has_data = [c > 0 for c in counts]

    bar_x = [c for c, hd in zip(centers, has_data, strict=True) if hd]
    bar_y = [a for a, hd in zip(accuracy, has_data, strict=True) if hd]
    bar_widths = [
        (hi - lo) * 0.9
        for lo, hi, hd in zip(rel["bin_lower"], rel["bin_upper"], has_data, strict=True)
        if hd
    ]
    ax.bar(
        bar_x, bar_y, width=bar_widths,
        align="center", alpha=0.7, color="#0072B2",
        edgecolor="#003c5f", label="accuracy",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#444444", linestyle="--",
            linewidth=1.0, label="perfect cal")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{q_name}\nECE={ece:.3f}, n={n_valid}", fontsize=10)
    ax.grid(True, alpha=0.25)


def _save_per_question_figure(
    q_name: str,
    rel: dict[str, list[float]],
    *,
    ece: float,
    n_valid: int,
    out_path: Path,
) -> None:
    """Side-by-side reliability + confidence-histogram figure for one question."""
    import matplotlib.pyplot as plt

    fig, (ax_rel, ax_hist) = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=140)
    _plot_reliability_panel(ax_rel, rel, q_name=q_name, ece=ece, n_valid=n_valid)
    ax_rel.legend(loc="upper left", fontsize=8)

    centers = [(lo + hi) / 2 for lo, hi in zip(rel["bin_lower"], rel["bin_upper"], strict=True)]
    bar_widths = [(hi - lo) * 0.9 for lo, hi in zip(rel["bin_lower"], rel["bin_upper"], strict=True)]
    ax_hist.bar(
        centers, rel["count"], width=bar_widths,
        align="center", color="#E69F00", edgecolor="#7a5500",
    )
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_xlabel("confidence")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"{q_name} — confidence histogram", fontsize=10)
    ax_hist.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _save_overview_figure(
    per_q: dict[str, dict[str, Any]],
    *,
    out_path: Path,
    macro_ece: float,
) -> None:
    """2x5 grid of all 10 questions' reliability diagrams (one figure for the model card)."""
    import matplotlib.pyplot as plt

    items = list(per_q.items())
    fig, axes = plt.subplots(2, 5, figsize=(20.0, 8.5), dpi=140)
    for idx, (q_name, m) in enumerate(items):
        ax = axes[idx // 5, idx % 5]
        if m["n_valid"] == 0 or m["reliability"] is None:
            ax.text(0.5, 0.5, f"{q_name}\n(no data)", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        _plot_reliability_panel(
            ax, m["reliability"], q_name=q_name, ece=m["ece"], n_valid=m["n_valid"]
        )
    fig.suptitle(
        f"GZ DESI per-question reliability (macro-ECE={macro_ece:.3f})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--macro-ece-max", type=float, default=DEFAULT_MACRO_ECE_MAX)
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--max-batches", type=int, default=DEFAULT_MAX_BATCHES)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {args.checkpoint}; "
            "run the T2.3 trainer first (configs/m2_w23_reproduction.yaml)"
        )
    cfg = MQConfig.from_yaml(args.config)
    settings = Settings()  # type: ignore[call-arg]
    os.environ["HF_TOKEN"] = settings.HF_TOKEN.get_secret_value()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cal] config={args.config} checkpoint={args.checkpoint}", flush=True)
    print(f"[cal] out_dir={args.out_dir} n_bins={args.n_bins}", flush=True)

    _, _, test_paths = _resolve_shards(cfg.data.shards)
    print(f"[cal] test shards: {len(test_paths)}", flush=True)

    mean, std = load_normalization(cfg.data.normalization)
    eval_tf = build_eval_transform(image_size=cfg.data.image_size, mean=mean, std=std)
    test_ds = build_gz_desi_hf_dataset(
        test_paths,
        eval_tf,
        shuffle_buffer=1,
        shardshuffle=False,
        min_votes=cfg.data.min_votes,
    )
    test_loader: DataLoader[Any] = DataLoader(
        test_ds,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_multi_question,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cal] device={device}", flush=True)
    model, _, _ = build_zoobot_finetune(
        num_classes=cfg.model.num_classes, encoder_id=cfg.model.encoder
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    captured = _capture_confidence_correct(
        model, test_loader, device=device, max_batches=args.max_batches
    )
    per_q = _per_question_metrics(captured, n_bins=args.n_bins)
    macro_ece = _macro_ece(per_q)
    macro_brier = float(
        sum(v["brier"] for v in per_q.values() if v["n_valid"] > 0)
        / max(1, sum(1 for v in per_q.values() if v["n_valid"] > 0))
    )

    print(f"[cal] macro-ECE = {macro_ece:.4f}", flush=True)
    for q_name, m in per_q.items():
        print(
            f"  {q_name:25s}  ECE={m['ece']:.4f}  MCE={m['mce']:.4f}  "
            f"Brier={m['brier']:.4f}  conf={m['mean_confidence']:.4f}  "
            f"acc={m['accuracy']:.4f}  n={m['n_valid']}",
            flush=True,
        )

    # Per-question PNGs.
    for q_name, m in per_q.items():
        if m["n_valid"] == 0 or m["reliability"] is None:
            continue
        out_png = args.out_dir / f"calibration_{q_name}.png"
        _save_per_question_figure(
            q_name, m["reliability"],
            ece=m["ece"], n_valid=m["n_valid"], out_path=out_png,
        )
    overview_png = args.out_dir / "reliability_overview.png"
    _save_overview_figure(per_q, out_path=overview_png, macro_ece=macro_ece)
    print(f"[cal] wrote {overview_png}", flush=True)

    # Strip the heavy reliability dict from the JSON dump (kept only as
    # PNG); per-question summary stats are enough for the test gate +
    # model card.
    summary_per_q = {
        q: {
            "n_valid": m["n_valid"],
            "ece": m["ece"],
            "mce": m["mce"],
            "brier": m["brier"],
            "mean_confidence": m["mean_confidence"],
            "accuracy": m["accuracy"],
        }
        for q, m in per_q.items()
    }
    metrics = {
        "task": "T2.5-calibration",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": str(device),
        "n_bins": args.n_bins,
        "n_test_shards": len(test_paths),
        "macro_ece": macro_ece,
        "macro_brier": macro_brier,
        "macro_ece_max": args.macro_ece_max,
        "macro_ece_passes": macro_ece <= args.macro_ece_max,
        "per_question": summary_per_q,
    }
    out_json = args.out_dir / "calibration_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[cal] wrote {out_json}", flush=True)

    if macro_ece > args.macro_ece_max:
        print(
            f"[cal] FAIL: macro-ECE {macro_ece:.4f} > max {args.macro_ece_max}",
            flush=True,
        )
        return 2
    print(
        f"[cal] PASS: macro-ECE {macro_ece:.4f} <= max {args.macro_ece_max}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
