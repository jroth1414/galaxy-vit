"""Trainer for the multi-question Walmsley+23 reproduction (T2.3).

Separate module from ``trainer.py`` because the data path, loss, and
metric semantics differ enough that interleaving the two would hurt
readability. Reuses the helpers (``seed_everything``, ``_git_sha``,
``_pip_freeze``, ``build_scheduler``, ``_render_curves``) so the
schedule + reproducibility surface stays consistent.

Invocation::

    python -m galaxy_vit.training.multi_question_trainer \\
        --config configs/m2_w23_reproduction.yaml

Writes the same triplet of artefacts the M1 trainer writes, plus a
per-question top-1 breakdown:

    runs/<run_id>/metrics.json    # per-question + macro top-1 history
    runs/<run_id>/curves.png      # train loss + macro_top1 vs epoch
    runs/<run_id>/run_config.json # rule-8 reproducibility metadata
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader

from galaxy_vit.config import Settings
from galaxy_vit.data.gz_desi_hf_dataset import (
    build_gz_desi_hf_dataset,
    collate_multi_question,
)
from galaxy_vit.data.gz_desi_labels import (
    DEFAULT_MIN_VOTES,
    question_index_groups,
)
from galaxy_vit.data.transforms import (
    build_eval_transform,
    build_train_transform,
    load_normalization,
)
from galaxy_vit.models.zoobot_encoder import ZOOBOT_HF_ID, build_zoobot_finetune
from galaxy_vit.training.metrics import macro_f1, top1_accuracy  # noqa: F401  reuse
from galaxy_vit.training.multi_question import (
    MultiQuestionAccumulator,
    multi_question_loss,
)
from galaxy_vit.training.trainer import (
    _git_sha,
    _pip_freeze,
    _render_curves,
    build_scheduler,
    seed_everything,
)

NUM_DR8_ANSWERS = 34


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HFShardsConfig(_StrictBase):
    """Selector for HF gz_desi_wds tar shards, split into train/val/test."""

    shards_dir: Path
    train_shard_pattern: str = "gz_desi_train_*_512.tar"
    test_shard_pattern: str = "gz_desi_test_*_512.tar"
    # Carve val out of train: shards in [val_train_start, val_train_end) -> val.
    val_train_start: int = 80
    val_train_end: int = 100


class DataConfig(_StrictBase):
    source: Literal["gz_desi_wds"] = "gz_desi_wds"
    shards: HFShardsConfig
    image_size: int = 224
    normalization: Path
    num_workers: int = 0
    min_votes: int = DEFAULT_MIN_VOTES


class ModelConfig(_StrictBase):
    encoder: str = ZOOBOT_HF_ID
    num_classes: int = NUM_DR8_ANSWERS  # 34 for the multi-question head


class OptimConfig(_StrictBase):
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
    early_stop_metric: str = "val/macro_top1"
    early_stop_patience: int = 5
    head_only_epochs: int = 0
    steps_per_epoch: int | None = None  # cap for streaming datasets


class LoggingConfig(_StrictBase):
    save_dir: Path
    tensorboard_subdir: str = "tb"
    tags: list[str] = Field(default_factory=list)


class MQConfig(_StrictBase):
    run_id: str
    seed: int = 42
    data: DataConfig
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, path: Path) -> MQConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _resolve_shards(cfg: HFShardsConfig) -> tuple[list[Path], list[Path], list[Path]]:
    """Split the cached HF shards into (train, val, test) by filename pattern."""
    train_all = sorted(cfg.shards_dir.glob(cfg.train_shard_pattern))
    test_all = sorted(cfg.shards_dir.glob(cfg.test_shard_pattern))

    if not train_all:
        raise FileNotFoundError(
            f"no train shards matching {cfg.train_shard_pattern!r} in {cfg.shards_dir}"
        )
    if not test_all:
        raise FileNotFoundError(
            f"no test shards matching {cfg.test_shard_pattern!r} in {cfg.shards_dir}"
        )

    val_lo = max(0, cfg.val_train_start)
    val_hi = min(len(train_all), cfg.val_train_end)
    train_paths = train_all[:val_lo] + train_all[val_hi:]
    val_paths = train_all[val_lo:val_hi]
    test_paths = list(test_all)

    if not val_paths:
        raise ValueError(
            f"val carve-out ({val_lo}, {val_hi}) is empty given {len(train_all)} train shards"
        )
    return train_paths, val_paths, test_paths


_PRECISION_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    precision: str,
    grad_clip: float,
    epoch: int,
    log_step: Callable[[dict[str, Any]], None],
    steps_per_epoch: int | None,
) -> float:
    model.train()
    autocast_dtype = _PRECISION_DTYPE[precision]
    use_amp = precision != "fp32" and device.type == "cuda"
    question_groups = question_index_groups()

    total_loss = 0.0
    n_batches = 0
    for batch_idx, (x, plurality, valid) in enumerate(loader):
        if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
            break
        x = x.to(device, non_blocking=True)
        plurality = plurality.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=use_amp
        ):
            logits = model(pixel_values=x).logits
            loss = multi_question_loss(
                logits, plurality, valid, question_groups=question_groups
            )

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
def _eval_loop(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Run the full eval loop. Returns (per_question_dict, macro_top1, eval_loss)."""
    model.eval()
    accumulator = MultiQuestionAccumulator(question_index_groups())
    total_loss = 0.0
    n_batches = 0

    for x, plurality, valid in loader:
        x = x.to(device, non_blocking=True)
        plurality = plurality.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        logits = model(pixel_values=x).logits
        loss = multi_question_loss(
            logits, plurality, valid, question_groups=question_index_groups()
        )
        accumulator.update(logits, plurality, valid)
        total_loss += float(loss.item())
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    return accumulator.result(), accumulator.macro_top1, avg_loss


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = MQConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})

    settings = Settings()  # type: ignore[call-arg]
    os.environ["HF_TOKEN"] = settings.HF_TOKEN.get_secret_value()
    print(
        f"[mq_trainer] run_id={cfg.run_id} seed={cfg.seed} "
        f"shards_dir={cfg.data.shards.shards_dir}",
        flush=True,
    )

    seed_everything(cfg.seed)
    save_dir = cfg.logging.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    train_paths, val_paths, test_paths = _resolve_shards(cfg.data.shards)
    print(
        f"[mq_trainer] shards: train={len(train_paths)}, "
        f"val={len(val_paths)}, test={len(test_paths)}",
        flush=True,
    )

    mean, std = load_normalization(cfg.data.normalization)
    train_tf = build_train_transform(image_size=cfg.data.image_size, mean=mean, std=std)
    eval_tf = build_eval_transform(image_size=cfg.data.image_size, mean=mean, std=std)

    train_ds = build_gz_desi_hf_dataset(
        train_paths, train_tf, shuffle_buffer=512, shardshuffle=True,
        min_votes=cfg.data.min_votes, seed=cfg.seed,
    )
    val_ds = build_gz_desi_hf_dataset(
        val_paths, eval_tf, shuffle_buffer=1, shardshuffle=False,
        min_votes=cfg.data.min_votes,
    )
    test_ds = build_gz_desi_hf_dataset(
        test_paths, eval_tf, shuffle_buffer=1, shardshuffle=False,
        min_votes=cfg.data.min_votes,
    )

    train_loader: DataLoader[Any] = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers,
        collate_fn=collate_multi_question, drop_last=True,
    )
    val_loader: DataLoader[Any] = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers,
        collate_fn=collate_multi_question,
    )
    test_loader: DataLoader[Any] = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers,
        collate_fn=collate_multi_question,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mq_trainer] device={device}", flush=True)

    model, encoder_mod, head_mod = build_zoobot_finetune(
        num_classes=cfg.model.num_classes, encoder_id=cfg.model.encoder
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": list(encoder_mod.parameters()), "lr": cfg.optim.encoder_lr,
             "weight_decay": cfg.optim.weight_decay},
            {"params": list(head_mod.parameters()), "lr": cfg.optim.head_lr,
             "weight_decay": cfg.optim.weight_decay},
        ]
    )
    scheduler = build_scheduler(
        optimizer, warmup_epochs=cfg.optim.warmup_epochs,
        total_epochs=cfg.train.epochs, schedule=cfg.optim.schedule,
    )

    if cfg.train.head_only_epochs > 0:
        for p in encoder_mod.parameters():
            p.requires_grad = False
        print(
            f"[mq_trainer] head-only stage: encoder frozen for "
            f"first {cfg.train.head_only_epochs} epochs",
            flush=True,
        )

    from torch.utils.tensorboard.writer import SummaryWriter

    tb_dir = save_dir / cfg.logging.tensorboard_subdir
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[mq_trainer] TensorBoard log dir: {tb_dir}", flush=True)

    _global_step = {"i": 0}

    def log_metrics(metrics: dict[str, Any]) -> None:
        for k, v in metrics.items():
            if isinstance(v, int | float):
                writer.add_scalar(k, v, _global_step["i"])
        _global_step["i"] += 1

    cfg_payload = {
        "task": "T2.3-w23-reproduction",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config": cfg.model_dump(mode="json"),
        "settings_redacted": settings.redacted(),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "tensorboard_logdir": str(tb_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "n_train_shards": len(train_paths),
        "n_val_shards": len(val_paths),
        "n_test_shards": len(test_paths),
        "pip_freeze": _pip_freeze(),
    }
    (save_dir / "run_config.json").write_text(
        json.dumps(cfg_payload, indent=2, sort_keys=True), encoding="utf-8"
    )

    best_metric = -float("inf")
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, Any]] = []

    for epoch in range(cfg.train.epochs):
        if epoch == cfg.train.head_only_epochs and cfg.train.head_only_epochs > 0:
            for p in encoder_mod.parameters():
                p.requires_grad = True
            print(f"[mq_trainer] unfreezing encoder at epoch {epoch + 1}", flush=True)

        print(f"[mq_trainer] === epoch {epoch + 1}/{cfg.train.epochs} ===", flush=True)
        train_loss = _train_one_epoch(
            model, train_loader, optimizer,
            device=device, precision=cfg.train.precision,
            grad_clip=cfg.train.grad_clip, epoch=epoch, log_step=log_metrics,
            steps_per_epoch=cfg.train.steps_per_epoch,
        )
        scheduler.step()
        per_q, macro_top1, val_loss = _eval_loop(model, val_loader, device=device)

        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            "train/loss_avg": train_loss,
            "val/loss": val_loss,
            "val/macro_top1": macro_top1,
            "lr/encoder": optimizer.param_groups[0]["lr"],
            "lr/head": optimizer.param_groups[1]["lr"],
        }
        for q_name, q_data in per_q.items():
            epoch_metrics[f"val/{q_name}_top1"] = q_data["top1"]
            epoch_metrics[f"val/{q_name}_n_valid"] = q_data["n_valid"]
        log_metrics(epoch_metrics)
        history.append({**epoch_metrics, "val/per_question": per_q})

        print(
            f"[mq_trainer] epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"macro_top1={macro_top1:.4f}",
            flush=True,
        )
        for q_name, q_data in per_q.items():
            print(
                f"  {q_name:25s}  top1={q_data['top1']:.4f}  "
                f"n_valid={q_data['n_valid']}",
                flush=True,
            )

        if macro_top1 > best_metric:
            best_metric = macro_top1
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metric": macro_top1,
                    "metric_name": "val/macro_top1",
                    "config": cfg.model_dump(mode="json"),
                },
                save_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.early_stop_patience:
                print(
                    f"[mq_trainer] early stopping at epoch {epoch + 1} "
                    f"(no improvement on val/macro_top1 for {patience_counter} epochs)",
                    flush=True,
                )
                break

    print("[mq_trainer] running test eval with best epoch checkpoint...", flush=True)
    if best_epoch >= 0:
        ckpt = torch.load(save_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    test_per_q, test_macro_top1, test_loss = _eval_loop(model, test_loader, device=device)
    print(f"[mq_trainer] test/macro_top1 = {test_macro_top1:.4f}", flush=True)
    for q_name, q_data in test_per_q.items():
        print(
            f"  {q_name:25s}  top1={q_data['top1']:.4f}  "
            f"n_valid={q_data['n_valid']}",
            flush=True,
        )

    final = {
        "run_id": cfg.run_id,
        "best_epoch": best_epoch,
        "best_metric_name": "val/macro_top1",
        "best_metric_value": best_metric,
        "history": history,
        "test": {
            "macro_top1": test_macro_top1,
            "loss": test_loss,
            "per_question": test_per_q,
        },
        "n_epochs_run": len(history),
    }
    (save_dir / "metrics.json").write_text(
        json.dumps(final, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[mq_trainer] wrote {save_dir / 'metrics.json'}", flush=True)

    # Render curves: train_loss + val_macro_top1.
    # Re-package history into the shape _render_curves expects.
    curve_history = [
        {
            "epoch": h["epoch"],
            "train/loss_avg": h["train/loss_avg"],
            "val/loss": h["val/loss"],
            "val/top1": h["val/macro_top1"],  # repurpose the field for the curves plot
            "val/macro_f1": h["val/macro_top1"],  # ditto
        }
        for h in history
    ]
    _render_curves(curve_history, save_dir / "curves.png", run_id=cfg.run_id)
    print(f"[mq_trainer] wrote {save_dir / 'curves.png'}", flush=True)

    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
