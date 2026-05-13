"""Trainer for the Dirichlet-Multinomial multi-question head (T3.6).

Phase 3's full training run. Mirrors :mod:`galaxy_vit.training.multi_question_trainer`
(T2.3) in structure but swaps:

* Head:   ``build_zoobot_dirichlet`` (concentration output) instead of
          ``build_zoobot_finetune`` (logits output).
* Loss:   :func:`galaxy_vit.losses.dirichlet_mn.dirichlet_multinomial_nll`
          on raw vote counts instead of per-question masked
          cross-entropy on plurality labels.
* Eval:   per-question vote MAE + coverage @ 95% CI + finite-alpha
          sanity, instead of per-question top-1.
* Early-stop metric: ``val/vote_mae_macro`` (lower is better, so the
          comparison direction flips vs T2.3's ``val/macro_top1``).

Reuses ``seed_everything``, ``_git_sha``, ``_pip_freeze``,
``build_scheduler``, ``_render_curves`` from the T2.3 trainer so the
schedule + reproducibility surface stays consistent.

Invocation::

    python -m galaxy_vit.training.dirichlet_trainer \\
        --config configs/m3_dirichlet.yaml

Writes the same triplet of artefacts the M2 trainer writes, plus a
per-question MAE + coverage breakdown:

    runs/<run_id>/metrics.json    # per-question MAE + coverage history
    runs/<run_id>/curves.png      # train loss + val MAE vs epoch
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
    build_gz_desi_hf_dataset_for_dirichlet,
    collate_dirichlet,
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
from galaxy_vit.inference.posterior import credible_interval
from galaxy_vit.losses.dirichlet_mn import (
    dirichlet_multinomial_nll,
    expected_fractions,
)
from galaxy_vit.models.dirichlet_head import (
    DEFAULT_ALPHA_FLOOR,
    NUM_DR8_ANSWERS,
    build_zoobot_dirichlet,
)
from galaxy_vit.models.zoobot_encoder import ZOOBOT_HF_ID
from galaxy_vit.training.multi_question_trainer import (
    HFShardsConfig,
    _resolve_shards,
)
from galaxy_vit.training.trainer import (
    _git_sha,
    _pip_freeze,
    _render_curves,
    build_scheduler,
    seed_everything,
)


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_StrictBase):
    source: Literal["gz_desi_wds"] = "gz_desi_wds"
    shards: HFShardsConfig
    image_size: int = 224
    normalization: Path
    num_workers: int = 0
    min_votes: int = DEFAULT_MIN_VOTES


class ModelConfig(_StrictBase):
    encoder: str = ZOOBOT_HF_ID
    head: Literal["dirichlet_multinomial"] = "dirichlet_multinomial"
    num_answers: int = NUM_DR8_ANSWERS
    alpha_floor: float = DEFAULT_ALPHA_FLOOR


class LossConfig(_StrictBase):
    name: Literal["dirichlet_mn"] = "dirichlet_mn"
    min_votes: int = DEFAULT_MIN_VOTES


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
    early_stop_metric: str = "val/vote_mae_macro"
    early_stop_patience: int = 5
    head_only_epochs: int = 0
    steps_per_epoch: int | None = None  # cap for streaming datasets
    coverage_ci: float = 0.95


class LoggingConfig(_StrictBase):
    save_dir: Path
    tensorboard_subdir: str = "tb"
    tags: list[str] = Field(default_factory=list)
    # C-15: optional per-epoch demo-galaxy feature dump. When set, the
    # trainer encodes a fixed set of demo galaxy thumbnails after each
    # epoch and appends the resulting (24, 640) feature matrix to a
    # parquet at this path. Drives the TrainingMovie tab's animation.
    per_epoch_features_path: Path | None = None
    # Directory holding the demo thumbnails referenced by the per-epoch
    # hook. Defaults to artifacts/demo_galaxies (the T4.3 demo bundle).
    demo_galaxies_dir: Path = Path("artifacts/demo_galaxies")


class DirichletConfig(_StrictBase):
    run_id: str
    seed: int = 42
    data: DataConfig
    model: ModelConfig
    loss: LossConfig
    optim: OptimConfig
    train: TrainConfig
    logging: LoggingConfig

    @classmethod
    def from_yaml(cls, path: Path) -> DirichletConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


_PRECISION_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _load_demo_galaxy_thumbs(
    demo_galaxies_dir: Path,
    *,
    image_size: int,
    mean: list[float],
    std: list[float],
) -> tuple[list[str], torch.Tensor] | None:
    """C-15 helper: load demo-galaxy thumbnails through the eval transform.

    Returns ``(galaxy_ids, pixel_values)`` where pixel_values has shape
    ``(n, 3, image_size, image_size)`` ready for ``model.encoder(...)``.
    Returns None when the demo bundle is missing so the per-epoch
    hook can silently no-op for runs that don't ship the artifact.
    """
    manifest_path = demo_galaxies_dir / "manifest.json"
    if not manifest_path.is_file():
        print(
            f"[d_trainer] per-epoch demo features requested but "
            f"{manifest_path} is missing; skipping",
            flush=True,
        )
        return None
    manifest: list[dict[str, Any]] = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    eval_tf = build_eval_transform(image_size=image_size, mean=mean, std=std)
    from PIL import Image as PILImage_

    pixel_values: list[torch.Tensor] = []
    galaxy_ids: list[str] = []
    for entry in manifest:
        gid = str(entry["id"])
        thumb_path = demo_galaxies_dir / "thumbs" / f"{gid}.jpg"
        if not thumb_path.is_file():
            continue
        image = PILImage_.open(thumb_path).convert("RGB")
        pixel_values.append(eval_tf(image))
        galaxy_ids.append(gid)
    if not pixel_values:
        return None
    batch = torch.stack(pixel_values, dim=0)
    return galaxy_ids, batch


@torch.no_grad()
def _extract_demo_features(
    model: torch.nn.Module,
    demo_batch: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """C-15: encoder forward pass on a fixed demo batch.

    Returns ``(n, num_features)`` float32 features on CPU. The model
    is put back into ``train()`` mode by the caller -- we don't touch
    it here so the per-epoch hook stays minimally side-effecting.
    """
    was_training = model.training
    model.eval()
    try:
        x = demo_batch.to(device, non_blocking=True)
        feats = model.encoder(x).float().cpu()  # type: ignore[operator]
    finally:
        if was_training:
            model.train()
    assert isinstance(feats, torch.Tensor)
    return feats


def _write_per_epoch_features(
    out_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Persist accumulated per-epoch demo features as a parquet."""
    import pandas as pd

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)


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
    groups = question_index_groups()

    total_loss = 0.0
    n_batches = 0
    for batch_idx, (x, counts, valid) in enumerate(loader):
        if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
            break
        x = x.to(device, non_blocking=True)
        counts = counts.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=autocast_dtype, enabled=use_amp
        ):
            alpha = model(pixel_values=x).alpha
        # Loss in fp32 (gammaln in bf16 is unreliable -- see DEVPLAN T3.4 spec).
        alpha_fp32 = alpha.float()
        loss = dirichlet_multinomial_nll(
            alpha_fp32, counts, valid, question_groups=groups
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
    coverage_ci: float = 0.95,
) -> tuple[dict[str, dict[str, float]], float, float, float, bool]:
    """Run the full eval loop.

    Returns
    -------
    ``(per_question, mae_macro, coverage_macro, eval_loss, all_finite)``:

    * ``per_question[name] = {"mae": float, "coverage": float, "n_valid": int}``
    * ``mae_macro``: mean MAE across questions with n_valid > 0
    * ``coverage_macro``: mean coverage across questions with n_valid > 0
    * ``eval_loss``: average DM NLL on the valid subset
    * ``all_finite``: False if any alpha was NaN / Inf at any point
    """
    model.eval()
    groups = question_index_groups()
    per_question_acc: dict[str, dict[str, float]] = {
        q: {"mae_sum": 0.0, "coverage_in": 0.0, "n_valid": 0} for q, _, _ in groups
    }
    total_loss = 0.0
    n_batches = 0
    all_finite = True

    for x, counts, valid in loader:
        x = x.to(device, non_blocking=True)
        counts = counts.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        alpha = model(pixel_values=x).alpha.float()
        if not bool(torch.isfinite(alpha).all().item()):
            all_finite = False
        loss = dirichlet_multinomial_nll(
            alpha, counts, valid, question_groups=groups
        )
        total_loss += float(loss.item())
        n_batches += 1

        # Per-question MAE on expected fractions vs observed (count / N).
        # expected_fractions normalizes per-question slice -> counts.float()
        # gives empirical c_i / sum(c_q) using the same per-slice safe-denom.
        pred_fracs = expected_fractions(alpha, question_groups=groups)
        obs_fracs = expected_fractions(counts.float(), question_groups=groups)
        # Per-question coverage at the configured CI.
        lower, upper = credible_interval(
            alpha, question_groups=groups, ci=coverage_ci
        )
        for q_idx, (q_name, start, end) in enumerate(groups):
            q_valid = valid[:, q_idx]
            n_v = int(q_valid.sum().item())
            if n_v == 0:
                continue
            slice_pred = pred_fracs[q_valid, start:end]
            slice_obs = obs_fracs[q_valid, start:end]
            slice_lower = lower[q_valid, start:end]
            slice_upper = upper[q_valid, start:end]
            mae = float((slice_pred - slice_obs).abs().mean().item())
            inside = ((slice_obs >= slice_lower) & (slice_obs <= slice_upper)).float()
            cov_q = float(inside.mean().item())
            # Weighted accumulation by n_v (so multi-batch averages are correct).
            per_question_acc[q_name]["mae_sum"] += mae * n_v
            per_question_acc[q_name]["coverage_in"] += cov_q * n_v
            per_question_acc[q_name]["n_valid"] += n_v

    per_question: dict[str, dict[str, float]] = {}
    mae_values: list[float] = []
    cov_values: list[float] = []
    for q_name, acc in per_question_acc.items():
        n_v = int(acc["n_valid"])
        if n_v == 0:
            per_question[q_name] = {"mae": 0.0, "coverage": 0.0, "n_valid": 0}
            continue
        mae = acc["mae_sum"] / n_v
        cov = acc["coverage_in"] / n_v
        per_question[q_name] = {"mae": mae, "coverage": cov, "n_valid": n_v}
        mae_values.append(mae)
        cov_values.append(cov)

    mae_macro = sum(mae_values) / len(mae_values) if mae_values else 0.0
    cov_macro = sum(cov_values) / len(cov_values) if cov_values else 0.0
    avg_loss = total_loss / max(1, n_batches)
    return per_question, mae_macro, cov_macro, avg_loss, all_finite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    cfg = DirichletConfig.from_yaml(args.config)
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})

    settings = Settings()  # type: ignore[call-arg]
    os.environ["HF_TOKEN"] = settings.HF_TOKEN.get_secret_value()
    print(
        f"[d_trainer] run_id={cfg.run_id} seed={cfg.seed} "
        f"shards_dir={cfg.data.shards.shards_dir}",
        flush=True,
    )

    seed_everything(cfg.seed)
    save_dir = cfg.logging.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    train_paths, val_paths, test_paths = _resolve_shards(cfg.data.shards)
    print(
        f"[d_trainer] shards: train={len(train_paths)}, "
        f"val={len(val_paths)}, test={len(test_paths)}",
        flush=True,
    )

    mean, std = load_normalization(cfg.data.normalization)
    train_tf = build_train_transform(image_size=cfg.data.image_size, mean=mean, std=std)
    eval_tf = build_eval_transform(image_size=cfg.data.image_size, mean=mean, std=std)

    train_ds = build_gz_desi_hf_dataset_for_dirichlet(
        train_paths, train_tf, shuffle_buffer=512, shardshuffle=True,
        min_votes=cfg.data.min_votes, seed=cfg.seed,
    )
    val_ds = build_gz_desi_hf_dataset_for_dirichlet(
        val_paths, eval_tf, shuffle_buffer=1, shardshuffle=False,
        min_votes=cfg.data.min_votes,
    )
    test_ds = build_gz_desi_hf_dataset_for_dirichlet(
        test_paths, eval_tf, shuffle_buffer=1, shardshuffle=False,
        min_votes=cfg.data.min_votes,
    )

    train_loader: DataLoader[Any] = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers,
        collate_fn=collate_dirichlet, drop_last=True,
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
    print(f"[d_trainer] device={device}", flush=True)

    model, encoder_mod, head_mod = build_zoobot_dirichlet(
        num_answers=cfg.model.num_answers,
        alpha_floor=cfg.model.alpha_floor,
        encoder_id=cfg.model.encoder,
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
            f"[d_trainer] head-only stage: encoder frozen for "
            f"first {cfg.train.head_only_epochs} epochs",
            flush=True,
        )

    from torch.utils.tensorboard.writer import SummaryWriter

    tb_dir = save_dir / cfg.logging.tensorboard_subdir
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"[d_trainer] TensorBoard log dir: {tb_dir}", flush=True)

    _global_step = {"i": 0}

    def log_metrics(metrics: dict[str, Any]) -> None:
        for k, v in metrics.items():
            if isinstance(v, int | float):
                writer.add_scalar(k, v, _global_step["i"])
        _global_step["i"] += 1

    cfg_payload = {
        "task": "T3.6-dirichlet-training",
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

    best_metric = float("inf")  # vote_mae_macro is lower-is-better
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, Any]] = []

    # C-15: per-epoch demo-feature accumulator. Lazily loaded only when
    # the config opts in; we capture epoch=-1 (pre-training, pretrained
    # encoder) so the movie's first frame shows the starting layout.
    per_epoch_rows: list[dict[str, Any]] = []
    demo_batch: torch.Tensor | None = None
    demo_galaxy_ids: list[str] | None = None
    per_epoch_features_path = cfg.logging.per_epoch_features_path
    if per_epoch_features_path is not None:
        loaded = _load_demo_galaxy_thumbs(
            cfg.logging.demo_galaxies_dir,
            image_size=cfg.data.image_size,
            mean=mean,
            std=std,
        )
        if loaded is not None:
            demo_galaxy_ids, demo_batch = loaded
            pre_feats = _extract_demo_features(
                model, demo_batch, device=device
            )
            for gid, f in zip(
                demo_galaxy_ids, pre_feats.tolist(), strict=True
            ):
                per_epoch_rows.append(
                    {"epoch": -1, "galaxy_id": gid, "features": f}
                )
            print(
                f"[d_trainer] per-epoch features ON; "
                f"{len(demo_galaxy_ids)} demo galaxies, "
                f"pretrained snapshot captured at epoch=-1",
                flush=True,
            )

    for epoch in range(cfg.train.epochs):
        if epoch == cfg.train.head_only_epochs and cfg.train.head_only_epochs > 0:
            for p in encoder_mod.parameters():
                p.requires_grad = True
            print(f"[d_trainer] unfreezing encoder at epoch {epoch + 1}", flush=True)

        print(f"[d_trainer] === epoch {epoch + 1}/{cfg.train.epochs} ===", flush=True)
        train_loss = _train_one_epoch(
            model, train_loader, optimizer,
            device=device, precision=cfg.train.precision,
            grad_clip=cfg.train.grad_clip, epoch=epoch, log_step=log_metrics,
            steps_per_epoch=cfg.train.steps_per_epoch,
        )
        scheduler.step()
        per_q, mae_macro, cov_macro, val_loss, all_finite = _eval_loop(
            model, val_loader, device=device, coverage_ci=cfg.train.coverage_ci,
        )

        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            "train/loss_avg": train_loss,
            "val/loss": val_loss,
            "val/vote_mae_macro": mae_macro,
            "val/coverage_macro": cov_macro,
            "val/all_finite": float(all_finite),
            "lr/encoder": optimizer.param_groups[0]["lr"],
            "lr/head": optimizer.param_groups[1]["lr"],
        }
        for q_name, q_data in per_q.items():
            epoch_metrics[f"val/{q_name}_mae"] = q_data["mae"]
            epoch_metrics[f"val/{q_name}_coverage"] = q_data["coverage"]
            epoch_metrics[f"val/{q_name}_n_valid"] = q_data["n_valid"]
        log_metrics(epoch_metrics)
        history.append({**epoch_metrics, "val/per_question": per_q})

        print(
            f"[d_trainer] epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"vote_mae_macro={mae_macro:.4f} coverage_macro={cov_macro:.4f} "
            f"all_finite={all_finite}",
            flush=True,
        )
        for q_name, q_data in per_q.items():
            print(
                f"  {q_name:25s}  mae={q_data['mae']:.4f}  "
                f"coverage={q_data['coverage']:.4f}  n_valid={q_data['n_valid']}",
                flush=True,
            )

        # C-15: dump per-epoch demo features (post-eval, before
        # checkpoint decision so a saved best.pt sits next to a
        # consistent snapshot).
        if (
            per_epoch_features_path is not None
            and demo_batch is not None
            and demo_galaxy_ids is not None
        ):
            feats = _extract_demo_features(model, demo_batch, device=device)
            for gid, f in zip(
                demo_galaxy_ids, feats.tolist(), strict=True
            ):
                per_epoch_rows.append(
                    {"epoch": epoch, "galaxy_id": gid, "features": f}
                )

        if mae_macro < best_metric:
            best_metric = mae_macro
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metric": mae_macro,
                    "metric_name": "val/vote_mae_macro",
                    "config": cfg.model_dump(mode="json"),
                },
                save_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.early_stop_patience:
                print(
                    f"[d_trainer] early stopping at epoch {epoch + 1} "
                    f"(no improvement on val/vote_mae_macro for {patience_counter} epochs)",
                    flush=True,
                )
                break

    # C-15: flush per-epoch demo features once training is complete
    # (whether by epoch budget or early-stop). Written before the
    # final test eval so a crash there doesn't lose the snapshots.
    if per_epoch_features_path is not None and per_epoch_rows:
        _write_per_epoch_features(per_epoch_features_path, per_epoch_rows)
        print(
            f"[d_trainer] wrote {per_epoch_features_path} "
            f"({len(per_epoch_rows)} rows across "
            f"{len({r['epoch'] for r in per_epoch_rows})} epochs)",
            flush=True,
        )

    print("[d_trainer] running test eval with best epoch checkpoint...", flush=True)
    if best_epoch >= 0:
        ckpt = torch.load(save_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    test_per_q, test_mae_macro, test_cov_macro, test_loss, test_finite = _eval_loop(
        model, test_loader, device=device, coverage_ci=cfg.train.coverage_ci,
    )
    print(
        f"[d_trainer] test/vote_mae_macro = {test_mae_macro:.4f}  "
        f"coverage_macro = {test_cov_macro:.4f}  all_finite = {test_finite}",
        flush=True,
    )
    for q_name, q_data in test_per_q.items():
        print(
            f"  {q_name:25s}  mae={q_data['mae']:.4f}  "
            f"coverage={q_data['coverage']:.4f}  n_valid={q_data['n_valid']}",
            flush=True,
        )

    final = {
        "run_id": cfg.run_id,
        "best_epoch": best_epoch,
        "best_metric_name": "val/vote_mae_macro",
        "best_metric_value": best_metric,
        "history": history,
        "test": {
            "vote_mae_macro": test_mae_macro,
            "coverage_macro": test_cov_macro,
            "loss": test_loss,
            "per_question": test_per_q,
            "all_finite": test_finite,
        },
        "n_epochs_run": len(history),
    }
    (save_dir / "metrics.json").write_text(
        json.dumps(final, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[d_trainer] wrote {save_dir / 'metrics.json'}", flush=True)

    # Render curves: train_loss + val_vote_mae_macro (re-purpose _render_curves'
    # top1/macro_f1 fields for the MAE plot since the curve renderer is
    # generic-axis).
    curve_history = [
        {
            "epoch": h["epoch"],
            "train/loss_avg": h["train/loss_avg"],
            "val/loss": h["val/loss"],
            "val/top1": -h["val/vote_mae_macro"],  # neg so higher = better, like top1
            "val/macro_f1": h["val/coverage_macro"],
        }
        for h in history
    ]
    _render_curves(curve_history, save_dir / "curves.png", run_id=cfg.run_id)
    print(f"[d_trainer] wrote {save_dir / 'curves.png'}", flush=True)

    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
