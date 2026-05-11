"""T5.1 - Inference pass + dry-run cost projection.

Loads the T3.6 Dirichlet checkpoint, batches through ALL galaxies in a
shard set, and writes a per-galaxy alpha parquet. Designed to scale
from a 10k-galaxy local dry-run to the full 8.67M GZ DESI catalog on
a rented A100/H100.

Two modes via ``--mode``:

* ``dry-run``   (default): iterate up to ``--max-rows`` galaxies on the
  local HF gz_desi_wds train+val+test shards, measure throughput, write
  ``runs/m3_dirichlet/inference_dry_run.json`` with the projection for
  V100 / A100 hardware speedups (literature-based throughput ratios)
  applied to the 8.67M-galaxy target. Useful before HITL #4 spends
  cloud GPU money.

* ``full``      : iterate every shard, no row cap, write the parquet to
  ``releases/gz_desi_dirichlet_v1.parquet`` plus an SHA-256 checksum
  file. Runs on whatever GPU is wired up. T5.1's actual run.

Output parquet schema (full mode):
  * ``key``           : str  (per-galaxy stable identifier from the shard JSON)
  * ``dr8_id``        : str  (DR8 identifier; may be missing for non-DR8 rows)
  * ``alpha_{i}``     : float32 (34 columns, one per answer in canonical order)

Both modes record run_config + throughput + checksum in JSON so the
T5.1 acceptance test can read those without re-deserializing the parquet.

Invocation::

    # Dry-run on the local 5070 Ti, 10k galaxies, 2-minute projection
    python -m scripts.run_inference_pass --mode dry-run --max-rows 10000

    # Full pass on rented A100 (write 8.67M-row parquet)
    python -m scripts.run_inference_pass --mode full \\
        --shards-dir <hf_cache>/datasets--mwalmsley--galaxy-images-decals \\
        --shard-pattern "*.tar"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from galaxy_vit.config import Settings
from galaxy_vit.data.gz_desi_hf import has_any_dr8_votes
from galaxy_vit.data.gz_desi_streaming import _iter_samples_from_shard
from galaxy_vit.data.transforms import build_eval_transform, load_normalization
from galaxy_vit.models.dirichlet_head import build_zoobot_dirichlet
from galaxy_vit.training.dirichlet_trainer import DirichletConfig
from galaxy_vit.training.multi_question_trainer import _resolve_shards

DEFAULT_CONFIG = Path("configs/m3_dirichlet.yaml")
DEFAULT_CHECKPOINT = Path("runs/m3_dirichlet/best.pt")
DEFAULT_DRY_RUN_OUT = Path("runs/m3_dirichlet/inference_dry_run.json")
DEFAULT_FULL_OUT = Path("releases/gz_desi_dirichlet_v1.parquet")
GZ_DESI_FULL_CATALOG_SIZE = 8_670_000  # Walmsley+23 8.67M number

# Hardware throughput ratios for the projection. Calibrated against the
# 5070 Ti baseline = 1.0x. Source: rough rules-of-thumb for ConvNeXt-nano
# inference at fp16 / bf16. Real numbers vary +/-30%; the projection
# uses these as a defensible mid-estimate so the user can sanity-check
# the order of magnitude before committing cloud GPU spend.
HW_SPEEDUP_VS_5070TI: dict[str, float] = {
    "5070_ti_bf16": 1.0,
    "v100_fp16":    1.5,
    "a100_bf16":    3.5,
    "h100_bf16":    6.0,
}

# Cloud GPU $/hour as of 2026-Q2 (Lambda Labs / Vast.ai mid-tier; user
# should re-check before commit). Used only to make the projection
# concrete; the user can override or ignore.
CLOUD_PRICE_USD_PER_HOUR: dict[str, float] = {
    "v100_fp16":  0.55,
    "a100_bf16":  1.30,
    "h100_bf16":  2.50,
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def _run_inference(
    model: torch.nn.Module,
    transform: Any,
    shards: list[Path],
    *,
    device: torch.device,
    batch_size: int,
    max_rows: int | None,
    dr8_only: bool,
) -> tuple[list[dict[str, Any]], float, int]:
    """Iterate ``shards``, batch through the model, return (rows, elapsed_s, n_examined).

    Rows are accumulated in-memory. For full-pass on 8.67M this is
    ~8.67M * 34 floats * 4 bytes = 1.2 GB plus metadata; the script's
    caller writes the parquet on disk after the loop completes. For the
    dry-run we cap at ``max_rows`` so memory stays trivial.
    """
    model.eval()
    rows: list[dict[str, Any]] = []
    batch_images: list[torch.Tensor] = []
    batch_keys: list[str] = []
    batch_dr8: list[str] = []
    n_examined = 0
    t0 = time.perf_counter()

    def _flush() -> None:
        if not batch_images:
            return
        x = torch.stack(batch_images, dim=0).to(device, non_blocking=True)
        alpha = model(pixel_values=x).alpha.float().cpu()
        for i, (k, d) in enumerate(zip(batch_keys, batch_dr8, strict=True)):
            row: dict[str, Any] = {"key": k, "dr8_id": d}
            for j in range(alpha.shape[1]):
                row[f"alpha_{j}"] = float(alpha[i, j].item())
            rows.append(row)
        batch_images.clear()
        batch_keys.clear()
        batch_dr8.clear()

    for shard in shards:
        for img, hf_labels in _iter_samples_from_shard(shard):
            n_examined += 1
            if dr8_only and not has_any_dr8_votes(hf_labels):
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
            tensor = transform(img)
            batch_images.append(tensor)
            batch_keys.append(str(hf_labels.get("key", "") or f"row_{len(rows)}"))
            batch_dr8.append(str(hf_labels.get("dr8_id", "") or ""))
            if len(batch_images) >= batch_size:
                _flush()
                if max_rows is not None and len(rows) >= max_rows:
                    break
        if max_rows is not None and len(rows) >= max_rows:
            break
    _flush()
    elapsed = time.perf_counter() - t0
    return rows, elapsed, n_examined


def _project_full_pass(
    throughput_galaxies_per_s_local: float,
    *,
    target_rows: int,
) -> dict[str, Any]:
    """Project hours + cost for each candidate hardware at ``target_rows``."""
    projections: dict[str, Any] = {}
    for hw, speedup in HW_SPEEDUP_VS_5070TI.items():
        throughput = throughput_galaxies_per_s_local * speedup
        hours = target_rows / throughput / 3600.0
        price = CLOUD_PRICE_USD_PER_HOUR.get(hw)
        projections[hw] = {
            "throughput_galaxies_per_s": throughput,
            "hours_for_target": hours,
            "cost_usd_estimate": (hours * price) if price else None,
            "price_usd_per_hour": price,
            "speedup_vs_5070ti": speedup,
        }
    return projections


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--mode", choices=("dry-run", "full"), default="dry-run",
    )
    parser.add_argument(
        "--max-rows", type=int, default=10_000,
        help="Dry-run cap; ignored when --mode=full.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--dr8-only", action="store_true",
        help="Filter to DR8-only galaxies (matches the T3.6 training filter).",
    )
    parser.add_argument(
        "--target-rows", type=int, default=GZ_DESI_FULL_CATALOG_SIZE,
        help="Project hours + cost for this many rows (default 8.67M).",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    cfg = DirichletConfig.from_yaml(args.config)
    settings = Settings()  # type: ignore[call-arg]
    os.environ["HF_TOKEN"] = settings.HF_TOKEN.get_secret_value()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    train_paths, val_paths, test_paths = _resolve_shards(cfg.data.shards)
    all_shards = list(train_paths) + list(val_paths) + list(test_paths)
    print(
        f"[infer] mode={args.mode}  shards={len(all_shards)}  "
        f"max_rows={args.max_rows if args.mode == 'dry-run' else 'no cap'}  "
        f"batch_size={args.batch_size}  dr8_only={args.dr8_only}",
        flush=True,
    )

    mean, std = load_normalization(cfg.data.normalization)
    eval_tf = build_eval_transform(image_size=cfg.data.image_size, mean=mean, std=std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] device={device}", flush=True)

    model, _enc, _head = build_zoobot_dirichlet(
        num_answers=cfg.model.num_answers,
        alpha_floor=cfg.model.alpha_floor,
        encoder_id=cfg.model.encoder,
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    max_rows = args.max_rows if args.mode == "dry-run" else None
    rows, elapsed, n_examined = _run_inference(
        model, eval_tf, all_shards,
        device=device, batch_size=args.batch_size,
        max_rows=max_rows, dr8_only=args.dr8_only,
    )
    throughput = len(rows) / max(elapsed, 1e-9)
    print(
        f"[infer] processed {len(rows)} rows in {elapsed:.1f}s  "
        f"throughput={throughput:.1f} galaxies/s  examined={n_examined}",
        flush=True,
    )

    common: dict[str, Any] = {
        "task": "T5.1-inference-pass",
        "mode": args.mode,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "git_sha": _git_sha(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "device": str(device),
        "rows_written": len(rows),
        "examined": n_examined,
        "elapsed_s": elapsed,
        "throughput_galaxies_per_s": throughput,
        "batch_size": args.batch_size,
        "dr8_only": args.dr8_only,
    }

    if args.mode == "dry-run":
        projection = _project_full_pass(throughput, target_rows=args.target_rows)
        common["projection"] = {
            "target_rows": args.target_rows,
            "hardware": projection,
        }
        out_path = args.out or DEFAULT_DRY_RUN_OUT
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(common, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[infer] wrote {out_path}", flush=True)
        print("[infer] PROJECTION:", flush=True)
        for hw, pj in projection.items():
            cost = pj["cost_usd_estimate"]
            cost_s = f"${cost:.2f}" if cost is not None else "(no price)"
            print(
                f"  {hw:>13s}  hours={pj['hours_for_target']:7.2f}  cost~={cost_s}",
                flush=True,
            )
        return 0

    # full mode
    out_path = args.out or DEFAULT_FULL_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    checksum = _sha256(out_path)
    print(f"[infer] wrote {out_path} ({len(rows)} rows)  sha256={checksum}", flush=True)

    common["output_parquet"] = str(out_path)
    common["sha256"] = checksum
    common["n_columns"] = len(df.columns)
    sidecar = out_path.with_suffix(".meta.json")
    sidecar.write_text(json.dumps(common, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[infer] wrote {sidecar}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
