"""Trainer for the M1 ViT-B/16 baseline run (T1.4).

Owns the training loop end-to-end:

* Pydantic config schema parsed from a YAML file (DEVPLAN §5).
* Galaxy10Dataset bridging the T1.1 splits CSV + the HF dataset + the T1.3
  transform pipeline.
* AdamW with separate LR groups for encoder vs classifier head.
* Linear warmup -> cosine decay LR schedule.
* bf16 autocast on CUDA, gradient clipping, optional MixUp.
* Per-epoch eval producing top-1 + macro-F1 (galaxy_vit.training.metrics).
* Early stopping on the configured metric.
* TensorBoard scalar logging to ``<save_dir>/tb/`` for live local
  monitoring (``tensorboard --logdir runs/``).
* End-of-run matplotlib ``curves.png`` (loss + top-1 + macro-F1 vs epoch)
  written alongside ``metrics.json`` for permanent record / model card.
* Writes ``runs/<save_dir>/run_config.json`` + ``metrics.json`` per rule 8.

Invoke via::

    python -m galaxy_vit.training.trainer --config configs/m1_vit_baseline.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader, Dataset

from galaxy_vit.config import Settings
from galaxy_vit.data.galaxy10 import GALAXY10_NUM_CLASSES, load_galaxy10
from galaxy_vit.data.transforms import (
    build_eval_transform,
    build_train_transform,
    load_normalization,
    mixup_batch,
)
from galaxy_vit.models.vit_baseline import build_vit_baseline, split_param_groups
from galaxy_vit.training.metrics import macro_f1, per_class_counts, top1_accuracy

# --------------------------------------------------------------------- #
# Config schema (Pydantic v2)
# --------------------------------------------------------------------- #


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_StrictBase):
    source: str
    split_csv: Path
    image_size: int = 224
    normalization: Path
    num_workers: int = 4


class ModelConfig(_StrictBase):
    encoder: str
    num_classes: int = GALAXY10_NUM_CLASSES


class OptimConfig(_StrictBase):
    name: Literal["adamw"] = "adamw"
    encoder_lr: float
    head_lr: float
    weight_decay: float
    warmup_epochs: int
    schedule: Literal["cosine", "constant"] = "cosine"


class TrainConfig(_StrictBase):
    epochs: int
    batch_size: int
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    grad_clip: float = 1.0
    early_stop_metric: str = "val/macro_f1"
    early_stop_patience: int = 5
    mixup_alpha: float = 0.0


class LoggingConfig(_StrictBase):
    save_dir: Path
    tensorboard_subdir: str = "tb"
    tags: list[str] = Field(default_factory=list)


class TrainerConfig(_StrictBase):
    run_id: str
    seed: int = 42
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, path: Path) -> TrainerConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# --------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------- #


class Galaxy10Dataset(Dataset[tuple[Any, int]]):
    """Galaxy10 filtered to one split via T1.1's splits CSV.

    Maps global indices in the CSV (concatenation of HF splits in
    ``sorted(ds.keys())`` order — i.e. ``test`` then ``train``) back to
    ``(hf_split, local_idx)`` for HF lookup, then applies a transform per
    item. PIL Image rows are returned as the transform's input; the
    transform pipeline (T1.3) handles ToImage / ToDtype conversion.
    """

    def __init__(
        self,
        data_dir: Path,
        splits_csv: Path,
        split: Literal["train", "val", "test"],
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self.transform = transform
        self.hf_ds = load_galaxy10(data_dir)

        self._boundaries: list[tuple[str, int, int]] = []
        cumulative = 0
        for split_key in sorted(self.hf_ds.keys()):
            n = len(self.hf_ds[split_key])
            self._boundaries.append((split_key, cumulative, cumulative + n))
            cumulative += n

        self._items: list[tuple[int, int]] = []
        with splits_csv.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row["split"] == split:
                    self._items.append((int(row["index"]), int(row["label"])))
        self._items.sort(key=lambda x: x[0])

    def __len__(self) -> int:
        return len(self._items)

    def _map_global(self, global_idx: int) -> tuple[str, int]:
        for split_key, start, end in self._boundaries:
            if start <= global_idx < end:
                return split_key, global_idx - start
        raise IndexError(f"global_idx {global_idx} out of range")

    def __getitem__(self, i: int) -> tuple[Any, int]:
        global_idx, label = self._items[i]
        split_key, local_idx = self._map_global(global_idx)
        row = self.hf_ds[split_key][local_idx]
        img = row["image"]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# --------------------------------------------------------------------- #
# Reproducibility helpers
# --------------------------------------------------------------------- #


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _pip_freeze() -> list[str]:
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return [line.strip() for line in out.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


# --------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------- #


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_epochs: int,
    total_epochs: int,
    schedule: str,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup -> cosine (or constant) decay over epochs."""

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        if schedule == "constant":
            return 1.0
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------- #
# Train + eval loops
# --------------------------------------------------------------------- #

_PRECISION_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Any, int]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    precision: str,
    grad_clip: float,
    mixup_alpha: float,
    epoch: int,
    log_step: Callable[[dict[str, Any]], None],
) -> float:
    """One epoch of training. Returns mean loss across batches."""
    model.train()
    autocast_dtype = _PRECISION_DTYPE[precision]
    use_amp = precision != "fp32" and device.type == "cuda"

    total_loss = 0.0
    n_batches = 0
    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if mixup_alpha > 0:
            x, y_a, y_b, lam = mixup_batch(x, y, alpha=mixup_alpha)
        else:
            y_a, y_b, lam = y, y, 1.0

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=use_amp
        ):
            logits = model(pixel_values=x).logits
            if mixup_alpha > 0:
                loss_a = F.cross_entropy(logits, y_a)
                loss_b = F.cross_entropy(logits, y_b)
                loss = lam * loss_a + (1.0 - lam) * loss_b
            else:
                loss = F.cross_entropy(logits, y_a)

        loss.backward()  # type: ignore[no-untyped-call]
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

        if batch_idx % 50 == 0:
            log_step(
                {
                    "train/loss": float(loss.item()),
                    "train/step": batch_idx,
                    "train/epoch": epoch,
                }
            )

    return total_loss / max(1, n_batches)


@torch.no_grad()
def eval_loop(
    model: nn.Module,
    loader: DataLoader[tuple[Any, int]],
    *,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    """Run model over loader, return loss/top1/macro_f1/per_class/n."""
    model.eval()

    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(pixel_values=x).logits
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
        total_loss += float(loss.item())
        n_batches += 1

    preds_t = torch.cat(all_preds)
    labels_t = torch.cat(all_labels)
    return {
        "loss": total_loss / max(1, n_batches),
        "top1": top1_accuracy(preds_t, labels_t),
        "macro_f1": macro_f1(preds_t, labels_t, num_classes),
        "per_class": per_class_counts(preds_t, labels_t, num_classes),
        "n": int(preds_t.numel()),
    }


# --------------------------------------------------------------------- #
# Main entrypoint
# --------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--seed", type=int, default=None, help="Override config seed."
    )
    args = parser.parse_args(argv)

    cfg = TrainerConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})

    settings = Settings()  # type: ignore[call-arg]
    os.environ["HF_TOKEN"] = settings.HF_TOKEN.get_secret_value()

    print(
        f"[trainer] run_id={cfg.run_id} seed={cfg.seed} "
        f"HF_USER={settings.HF_USER} DATA_DIR={settings.DATA_DIR}",
        flush=True,
    )

    seed_everything(cfg.seed)

    save_dir = cfg.logging.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    mean, std = load_normalization(cfg.data.normalization)
    print(f"[trainer] normalization mean={mean} std={std}", flush=True)

    train_tf = build_train_transform(image_size=cfg.data.image_size, mean=mean, std=std)
    eval_tf = build_eval_transform(image_size=cfg.data.image_size, mean=mean, std=std)

    print(f"[trainer] loading datasets from {cfg.data.split_csv}", flush=True)
    train_ds = Galaxy10Dataset(settings.DATA_DIR, cfg.data.split_csv, "train", train_tf)
    val_ds = Galaxy10Dataset(settings.DATA_DIR, cfg.data.split_csv, "val", eval_tf)
    print(f"[trainer] train n={len(train_ds)} val n={len(val_ds)}", flush=True)

    train_loader: DataLoader[tuple[Any, int]] = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader: DataLoader[tuple[Any, int]] = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[trainer] device={device}", flush=True)

    model = build_vit_baseline(
        num_classes=cfg.model.num_classes, encoder_id=cfg.model.encoder
    )
    model = model.to(device)

    param_groups = split_param_groups(
        model,
        encoder_lr=cfg.optim.encoder_lr,
        head_lr=cfg.optim.head_lr,
        weight_decay=cfg.optim.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=cfg.optim.warmup_epochs,
        total_epochs=cfg.train.epochs,
        schedule=cfg.optim.schedule,
    )

    from torch.utils.tensorboard.writer import SummaryWriter

    tb_dir = save_dir / cfg.logging.tensorboard_subdir
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(
        f"[trainer] TensorBoard log dir: {tb_dir} "
        f"(view with `tensorboard --logdir {save_dir}`)",
        flush=True,
    )

    # Globally monotone step counter so train batches and val epochs share an
    # x-axis in TensorBoard.
    _global_step: dict[str, int] = {"i": 0}

    def log_metrics(metrics: dict[str, Any]) -> None:
        for key, value in metrics.items():
            if isinstance(value, int | float):
                writer.add_scalar(key, value, _global_step["i"])
        _global_step["i"] += 1

    cfg_payload = {
        "task": "T1.4-vit-baseline",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config": cfg.model_dump(mode="json"),
        "settings_redacted": settings.redacted(),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "tensorboard_logdir": str(tb_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "pip_freeze": _pip_freeze(),
    }
    (save_dir / "run_config.json").write_text(
        json.dumps(cfg_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    best_metric_value = -float("inf")
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(cfg.train.epochs):
        print(f"[trainer] === epoch {epoch + 1}/{cfg.train.epochs} ===", flush=True)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            precision=cfg.train.precision,
            grad_clip=cfg.train.grad_clip,
            mixup_alpha=cfg.train.mixup_alpha,
            epoch=epoch,
            log_step=log_metrics,
        )

        scheduler.step()

        eval_metrics = eval_loop(
            model, val_loader, device=device, num_classes=cfg.model.num_classes
        )

        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            "train/loss_avg": train_loss,
            "val/loss": eval_metrics["loss"],
            "val/top1": eval_metrics["top1"],
            "val/macro_f1": eval_metrics["macro_f1"],
            "lr/encoder": optimizer.param_groups[0]["lr"],
            "lr/head": optimizer.param_groups[1]["lr"],
        }
        log_metrics(epoch_metrics)
        history.append({**epoch_metrics, "val/per_class": eval_metrics["per_class"]})

        print(
            f"[trainer] epoch {epoch + 1}: train_loss={train_loss:.4f} "
            f"val_loss={eval_metrics['loss']:.4f} "
            f"val_top1={eval_metrics['top1']:.4f} "
            f"val_macro_f1={eval_metrics['macro_f1']:.4f}",
            flush=True,
        )

        # Early stopping.
        metric_name = cfg.train.early_stop_metric
        metric_val = float(epoch_metrics.get(metric_name, -1.0))
        if metric_val > best_metric_value:
            best_metric_value = metric_val
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = save_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metric": metric_val,
                    "metric_name": metric_name,
                    "config": cfg.model_dump(mode="json"),
                },
                ckpt_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.early_stop_patience:
                print(
                    f"[trainer] early stopping at epoch {epoch + 1} "
                    f"(no improvement on {metric_name} for {patience_counter} epochs)",
                    flush=True,
                )
                break

    final = {
        "run_id": cfg.run_id,
        "best_epoch": best_epoch,
        "best_metric_name": cfg.train.early_stop_metric,
        "best_metric_value": best_metric_value,
        "history": history,
        "best_val": {
            "top1": history[best_epoch]["val/top1"] if best_epoch >= 0 else None,
            "macro_f1": history[best_epoch]["val/macro_f1"]
            if best_epoch >= 0
            else None,
        },
        "n_epochs_run": len(history),
        "tensorboard_logdir": str(tb_dir),
    }
    (save_dir / "metrics.json").write_text(
        json.dumps(final, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[trainer] wrote {save_dir / 'metrics.json'}", flush=True)
    print(
        f"[trainer] best epoch: {best_epoch + 1}, "
        f"best {cfg.train.early_stop_metric}: {best_metric_value:.4f}",
        flush=True,
    )

    _render_curves(history, save_dir / "curves.png", run_id=cfg.run_id)
    print(f"[trainer] wrote {save_dir / 'curves.png'}", flush=True)

    writer.close()
    return 0


def _render_curves(
    history: list[dict[str, Any]],
    out_path: Path,
    *,
    run_id: str,
) -> None:
    """Render train/val loss + val top-1 / macro-F1 vs epoch as a single PNG."""
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend; no display server needed
    import matplotlib.pyplot as plt

    epochs = [int(h["epoch"]) + 1 for h in history]
    train_loss = [float(h["train/loss_avg"]) for h in history]
    val_loss = [float(h["val/loss"]) for h in history]
    val_top1 = [float(h["val/top1"]) for h in history]
    val_f1 = [float(h["val/macro_f1"]) for h in history]

    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, train_loss, label="train", marker="o", ms=4)
    ax_loss.plot(epochs, val_loss, label="val", marker="o", ms=4)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_metric.plot(epochs, val_top1, label="val top-1", marker="o", ms=4)
    ax_metric.plot(epochs, val_f1, label="val macro-F1", marker="o", ms=4)
    ax_metric.axhline(0.82, ls="--", c="gray", alpha=0.5, lw=1)
    ax_metric.axhline(0.78, ls="--", c="gray", alpha=0.5, lw=1)
    ax_metric.set_xlabel("epoch")
    ax_metric.set_ylabel("metric")
    ax_metric.set_title("Validation metrics (T1.4 thresholds: 0.82 / 0.78)")
    ax_metric.set_ylim(0, 1)
    ax_metric.grid(alpha=0.3)
    ax_metric.legend()

    fig.suptitle(run_id)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
