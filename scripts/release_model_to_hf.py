"""T6.2 - Push the Dirichlet model checkpoint + model card to the HF Hub.

Counterpart to scripts/release_to_hf.py (which publishes the predictions
parquet). Same two-phase pattern:

1. ``--dry-run`` (default): validate the local artefacts -- best.pt,
   run_config.json, calibrated_metrics.json, model_card.md exist and
   are consistent. Verify the checkpoint loads cleanly into a fresh
   model instance (sanity-check the published weights aren't
   torch-version-locked). No network.

2. ``--publish``: with HF_TOKEN, create the repo
   ``<HF_USER>/galaxy-vit-zoobot-dirichlet`` (DEVPLAN canonical name),
   upload best.pt + run_config.json + calibrated_metrics.json, and
   render docs/model_card.md -> repo README.md.

The user runs --publish manually after reviewing the dry-run output.
Same discipline as T5.3: prep is automated; the irreversible push is
human-executed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from galaxy_vit.config import Settings

DEFAULT_CHECKPOINT = Path("runs/m3_dirichlet/best.pt")
DEFAULT_RUN_CONFIG = Path("runs/m3_dirichlet/run_config.json")
DEFAULT_CALIBRATION = Path("runs/m3_dirichlet/calibrated_metrics.json")
DEFAULT_CARD = Path("docs/model_card.md")
DEFAULT_REPO_SUFFIX = "galaxy-vit-zoobot-dirichlet"  # DEVPLAN canonical


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _smoke_load(checkpoint: Path) -> tuple[bool, str]:
    """Try to load the checkpoint into a fresh model instance.

    Returns (ok, message). ok=False on any exception, with the
    truncated traceback in the message. This catches version-pinned
    state_dicts that would silently break on the first user who tries
    to load them from the Hub.
    """
    try:
        import torch

        from galaxy_vit.models.dirichlet_head import build_zoobot_dirichlet

        model, _, _ = build_zoobot_dirichlet(num_answers=34, alpha_floor=1.0)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "model_state_dict" not in ckpt:
            return False, "ckpt has no 'model_state_dict' key"
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            return False, f"missing keys: {missing[:3]}..."
        if unexpected:
            return False, f"unexpected keys: {unexpected[:3]}..."
        return True, "load OK"
    except Exception as exc:  # pragma: no cover -- diagnostic path
        return False, f"{type(exc).__name__}: {exc}"


def _dry_run(
    checkpoint: Path,
    run_config: Path,
    calibration: Path,
    card: Path,
    *,
    repo_id: str,
) -> int:
    """Validate local artefacts + print next-steps checklist. No network."""
    print(f"[release] DRY-RUN  target repo_id = {repo_id}", flush=True)
    for p, label in [
        (checkpoint, "checkpoint"),
        (run_config, "run_config"),
        (calibration, "calibrated_metrics"),
        (card, "model card"),
    ]:
        if not p.is_file():
            print(f"[release] FAIL: missing {label}: {p}", flush=True)
            return 2

    ckpt_sha = _sha256(checkpoint)
    print(
        f"[release] OK: {checkpoint} "
        f"({checkpoint.stat().st_size / (1 << 20):.1f} MB, "
        f"sha256={ckpt_sha[:16]}...)",
        flush=True,
    )
    print(f"[release] OK: {run_config} ({run_config.stat().st_size} B)", flush=True)
    print(
        f"[release] OK: {calibration} ({calibration.stat().st_size} B)", flush=True
    )
    print(f"[release] OK: {card} ({card.stat().st_size} B)", flush=True)

    print("[release] smoke-loading checkpoint into a fresh model...", flush=True)
    ok, msg = _smoke_load(checkpoint)
    if not ok:
        print(f"[release] FAIL: smoke load failed -- {msg}", flush=True)
        return 2
    print(f"[release] OK: {msg}", flush=True)

    print(flush=True)
    print("[release] NEXT STEPS:", flush=True)
    print("  1. Re-run with --publish (HF_TOKEN must have write scope):", flush=True)
    print(f"     python -m scripts.release_model_to_hf --publish --repo-id {repo_id}", flush=True)
    print(
        "  2. Optional: also publish to Zenodo for a citable model DOI",
        flush=True,
    )
    print(
        "     (paste docs/zenodo_metadata.json with upload_type=software).",
        flush=True,
    )
    return 0


def _publish(
    checkpoint: Path,
    run_config: Path,
    calibration: Path,
    card: Path,
    *,
    repo_id: str,
    hf_user: str,
) -> int:
    """Push checkpoint + sidecars + rendered README to the HF Hub model repo."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[release] FAIL: huggingface_hub not installed", flush=True)
        return 2

    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            settings = Settings()  # type: ignore[call-arg]
            token = settings.HF_TOKEN.get_secret_value()
        except Exception as exc:
            print(
                f"[release] FAIL: couldn't load HF_TOKEN (env or .env): {exc}",
                flush=True,
            )
            return 2
    if not token:
        print("[release] FAIL: HF_TOKEN missing in env AND .env", flush=True)
        return 2

    # Render card placeholders (only <HF_USER> needs substitution here;
    # the model card doesn't reference a Zenodo DOI for the weights).
    rendered_card = card.read_text(encoding="utf-8").replace("<HF_USER>", hf_user)
    rendered_card_path = card.parent / "_model_card_rendered.md"
    rendered_card_path.write_text(rendered_card, encoding="utf-8")
    print(f"[release] rendered card -> {rendered_card_path}", flush=True)

    api = HfApi(token=token)
    print(f"[release] ensuring model repo {repo_id} exists...", flush=True)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    print(f"[release] uploading {checkpoint}...", flush=True)
    api.upload_file(
        path_or_fileobj=str(checkpoint),
        path_in_repo=checkpoint.name,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[release] uploading {run_config}...", flush=True)
    api.upload_file(
        path_or_fileobj=str(run_config),
        path_in_repo=run_config.name,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[release] uploading {calibration}...", flush=True)
    api.upload_file(
        path_or_fileobj=str(calibration),
        path_in_repo=calibration.name,
        repo_id=repo_id,
        repo_type="model",
    )
    print("[release] uploading rendered README.md...", flush=True)
    api.upload_file(
        path_or_fileobj=str(rendered_card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[release] PUBLISHED: https://huggingface.co/{repo_id}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-config", type=Path, default=DEFAULT_RUN_CONFIG)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument(
        "--repo-id", type=str, default=None,
        help=(
            "HF model repo id (default: <HF_USER>/"
            f"{DEFAULT_REPO_SUFFIX}, reading HF_USER from env or .env)."
        ),
    )
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)

    hf_user = os.environ.get("HF_USER", "")
    if not hf_user:
        try:
            settings = Settings()  # type: ignore[call-arg]
            hf_user = settings.HF_USER
        except Exception:
            hf_user = "<HF_USER>"
    repo_id = args.repo_id or f"{hf_user}/{DEFAULT_REPO_SUFFIX}"

    if not args.publish:
        return _dry_run(
            args.checkpoint, args.run_config, args.calibration, args.card,
            repo_id=repo_id,
        )

    if hf_user == "<HF_USER>":
        print("[release] FAIL: HF_USER not set; required for --publish", flush=True)
        return 2
    return _publish(
        args.checkpoint, args.run_config, args.calibration, args.card,
        repo_id=repo_id, hf_user=hf_user,
    )


if __name__ == "__main__":
    raise SystemExit(main())
