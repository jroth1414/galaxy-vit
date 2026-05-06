"""CLI: render 30 stratified GradCAM (and optional attention-rollout) overlays.

Loads the best checkpoint from a finished training run (default:
``runs/m1_zoobot_finetune/best.pt``), draws 3 stratified test-split
samples per class via a seeded RNG (30 total for Galaxy10's 10 classes),
runs each through the model, and writes a PNG per sample under
``<run-dir>/interpretability/``.

Filename convention::

    <class_id>_<test_sample_index>_pred-<pred_class>_true-<true_class>.png

Per DEVPLAN T1.6 acceptance, both interpretability techniques are
exercised:

* ``--mode gradcam``  (default) — works for any CNN encoder. For Zoobot
  ConvNeXt-nano the target layer is ``encoder.stages[-1]`` (final
  conv stage; 7x7 spatial output upsampled to image resolution).
* ``--mode rollout``  — Abnar & Zuidema 2020 attention rollout; only
  applies to ViT-baseline checkpoints. Falls back to GradCAM with a
  warning on non-transformer models.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from galaxy_vit.config import Settings
from galaxy_vit.data.transforms import build_eval_transform, load_normalization
from galaxy_vit.inference.attention import (
    attention_rollout,
    gradcam,
    overlay_heatmap_on_image,
)
from galaxy_vit.inference.predict import resolve_target_layer
from galaxy_vit.training.trainer import (
    Galaxy10Dataset,
    TrainerConfig,
    build_model_and_split,
)

DEFAULT_RUN_DIR = Path("runs/m1_zoobot_finetune")
DEFAULT_K_PER_CLASS = 3
NUM_CLASSES = 10


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _select_stratified(
    labels: list[int],
    *,
    k_per_class: int,
    num_classes: int,
    seed: int,
) -> list[int]:
    """Return ``k_per_class * num_classes`` indices into ``labels``, balanced."""
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_class[lab].append(i)
    rng = random.Random(seed)
    chosen: list[int] = []
    for c in range(num_classes):
        candidates = by_class.get(c, [])
        if not candidates:
            continue
        k = min(k_per_class, len(candidates))
        chosen.extend(rng.sample(candidates, k))
    return chosen


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Directory containing best.pt + run_config.json.",
    )
    parser.add_argument(
        "--mode",
        choices=("gradcam", "rollout"),
        default="gradcam",
        help="Saliency technique. rollout requires a transformer model.",
    )
    parser.add_argument(
        "--k-per-class",
        type=int,
        default=DEFAULT_K_PER_CLASS,
        help="Stratified samples per class (default 3 -> 30 total).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for stratified sample selection.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir: Path = args.run_dir
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found at {ckpt_path}; run trainer first"
        )

    settings = Settings()  # type: ignore[call-arg]
    os.environ["HF_TOKEN"] = settings.HF_TOKEN.get_secret_value()

    print(f"[interpret] loading checkpoint from {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = TrainerConfig.model_validate(ckpt["config"])
    print(f"[interpret] run_id={cfg.run_id} model.kind={cfg.model.kind}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, encoder, _head = build_model_and_split(cfg.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print(f"[interpret] model loaded on {device}", flush=True)

    # Build eval transform (matches the trainer's val pipeline).
    mean, std = load_normalization(cfg.data.normalization)
    eval_tf = build_eval_transform(image_size=cfg.data.image_size, mean=mean, std=std)
    test_ds = Galaxy10Dataset(
        settings.DATA_DIR, cfg.data.split_csv, "test", eval_tf
    )
    raw_ds = Galaxy10Dataset(
        settings.DATA_DIR, cfg.data.split_csv, "test", transform=None
    )
    print(f"[interpret] test split: n={len(test_ds)}", flush=True)

    # Stratified selection.
    test_labels = test_ds.labels
    chosen = _select_stratified(
        test_labels,
        k_per_class=args.k_per_class,
        num_classes=NUM_CLASSES,
        seed=args.seed,
    )
    print(
        f"[interpret] selected {len(chosen)} samples "
        f"({args.k_per_class} per class * {NUM_CLASSES} classes)",
        flush=True,
    )

    out_dir = run_dir / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_layer = (
        resolve_target_layer(encoder, cfg.model.kind)
        if args.mode == "gradcam"
        else None
    )

    written_files: list[str] = []
    for ord_i, ds_i in enumerate(chosen):
        x_norm, true_label = test_ds[ds_i]
        raw_img, _ = raw_ds[ds_i]  # PIL image, original DECaLS thumbnail
        x = x_norm.unsqueeze(0).to(device)

        if args.mode == "rollout":
            heatmap = attention_rollout(model, x)[0]
        else:
            assert target_layer is not None
            heatmap = gradcam(model, x, target_layer=target_layer)[0]

        with torch.no_grad():
            logits = model(pixel_values=x).logits
        pred_label = int(logits.argmax(dim=-1)[0].item())

        overlay = overlay_heatmap_on_image(raw_img, heatmap)
        fname = (
            f"{true_label}_{ds_i}_pred-{pred_label}_true-{true_label}.png"
        )
        overlay.save(out_dir / fname)
        written_files.append(fname)
        if (ord_i + 1) % 5 == 0:
            print(
                f"[interpret] {ord_i + 1}/{len(chosen)} written...", flush=True
            )

    print(f"[interpret] wrote {len(written_files)} overlays to {out_dir}", flush=True)

    cfg_payload: dict[str, Any] = {
        "task": "T1.6-interpretability",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "mode": args.mode,
        "seed": args.seed,
        "k_per_class": args.k_per_class,
        "num_classes": NUM_CLASSES,
        "run_dir": str(run_dir),
        "ckpt": str(ckpt_path),
        "ckpt_epoch": ckpt.get("epoch"),
        "ckpt_metric": ckpt.get("metric"),
        "ckpt_metric_name": ckpt.get("metric_name"),
        "model_kind": cfg.model.kind,
        "model_encoder": cfg.model.encoder,
        "n_overlays": len(written_files),
        "out_dir": str(out_dir),
        "git_sha": _git_sha(),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    cfg_path = out_dir / "run_config.json"
    cfg_path.write_text(
        json.dumps(cfg_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[interpret] wrote {cfg_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
