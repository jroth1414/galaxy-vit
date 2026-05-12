"""T5.3 - Push the release parquet + dataset card to the HF Hub.

Idempotent: re-running uploads only the files that changed. Requires
``HF_TOKEN`` with write scope in the environment (the same secret
T1.x scripts use). Default repo ID is
``<HF_USER>/galaxy-vit-gz-desi-dirichlet-predictions``;
override with ``--repo-id`` if you prefer a different name on first
publish (this is the canonical DEVPLAN-undocumented dataset name --
DEVPLAN names the MODEL repo but not the DATASET repo).

Two-phase release:

1. ``--dry-run`` (default): no network. Validates that the local
   parquet + sidecar exist, the SHA-256 in the sidecar matches a
   fresh recompute, the dataset card renders, and prints the exact
   HF Hub URL + Zenodo workflow steps. Safe to run repeatedly.

2. ``--publish``: with HF_TOKEN set, creates the repo if needed,
   uploads ``releases/gz_desi_dirichlet_v1.parquet``,
   ``releases/gz_desi_dirichlet_v1.meta.json``, and renders
   ``docs/dataset_card.md`` -> repo ``README.md`` (with placeholder
   substitution). Prints the final HF Hub URL.

The user runs ``--publish`` manually after reviewing the dry-run
output. Same discipline as ``git push`` / Slack send: the script
prepares everything; the human executes the irreversible step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from galaxy_vit.config import Settings

DEFAULT_PARQUET = Path("releases/gz_desi_dirichlet_v1.parquet")
DEFAULT_META = Path("releases/gz_desi_dirichlet_v1.meta.json")
DEFAULT_CARD = Path("docs/dataset_card.md")
DEFAULT_ZENODO = Path("docs/zenodo_metadata.json")
DEFAULT_REPO_SUFFIX = "galaxy-vit-gz-desi-dirichlet-predictions"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _render_card(template_path: Path, *, hf_user: str, zenodo_doi: str) -> str:
    """Substitute <HF_USER> and <ZENODO_DOI> placeholders in the dataset card."""
    src = template_path.read_text(encoding="utf-8")
    return src.replace("<HF_USER>", hf_user).replace("<ZENODO_DOI>", zenodo_doi)


def _dry_run(
    parquet: Path,
    meta: Path,
    card: Path,
    zenodo: Path,
    *,
    repo_id: str,
) -> int:
    """Validate local artefacts + print the next-steps checklist. No network."""
    print(f"[release] DRY-RUN  target repo_id = {repo_id}", flush=True)
    if not parquet.is_file():
        print(f"[release] FAIL: missing {parquet}", flush=True)
        return 2
    if not meta.is_file():
        print(f"[release] FAIL: missing sidecar {meta}", flush=True)
        return 2
    if not card.is_file():
        print(f"[release] FAIL: missing dataset card {card}", flush=True)
        return 2
    if not zenodo.is_file():
        print(f"[release] FAIL: missing Zenodo template {zenodo}", flush=True)
        return 2

    payload = json.loads(meta.read_text(encoding="utf-8"))
    recorded_sha = str(payload["sha256"])
    actual_sha = _sha256(parquet)
    if recorded_sha != actual_sha:
        print(
            f"[release] FAIL: sha256 mismatch\n  meta:   {recorded_sha}\n"
            f"  actual: {actual_sha}",
            flush=True,
        )
        return 2

    print(f"[release] OK: {parquet} ({parquet.stat().st_size / (1 << 20):.1f} MB)", flush=True)
    print(f"[release] OK: sidecar sha256 matches  {actual_sha[:16]}...", flush=True)
    print(f"[release] OK: card {card} ({card.stat().st_size} B)", flush=True)
    print(f"[release] OK: Zenodo metadata {zenodo} ({zenodo.stat().st_size} B)", flush=True)
    print(flush=True)
    print("[release] NEXT STEPS:", flush=True)
    print("  1. Reserve a Zenodo DOI:", flush=True)
    print("     https://zenodo.org/deposit/new -> 'Reserve DOI' at top", flush=True)
    print(f"     Paste metadata from {zenodo} into the form (open in your editor).", flush=True)
    print("  2. Back-fill the reserved DOI into the dataset card placeholder", flush=True)
    print(f"     <ZENODO_DOI> in {card}.", flush=True)
    print("  3. Re-run this script with --publish (HF_TOKEN must have write scope):", flush=True)
    print(
        f"     python -m scripts.release_to_hf --publish --repo-id {repo_id} "
        f"--zenodo-doi <RESERVED_DOI>",
        flush=True,
    )
    print("  4. After HF publish: upload the parquet + sidecar to the Zenodo deposit", flush=True)
    print("     (the parquet stays in two places; HF Hub for lookup/streaming,", flush=True)
    print("     Zenodo for the citable DOI + long-term preservation).", flush=True)
    print("  5. Click 'Publish' on the Zenodo deposit (irreversible).", flush=True)
    return 0


def _publish(
    parquet: Path,
    meta: Path,
    card: Path,
    *,
    repo_id: str,
    hf_user: str,
    zenodo_doi: str,
) -> int:
    """Push parquet + sidecar + rendered README to the HF Hub dataset repo."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[release] FAIL: huggingface_hub not installed", flush=True)
        return 2

    # Load HF_TOKEN from .env first (same Settings flow the rest of the
    # project uses), then fall back to the raw process environment.
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

    rendered_card = _render_card(card, hf_user=hf_user, zenodo_doi=zenodo_doi)
    rendered_card_path = card.parent / "_dataset_card_rendered.md"
    rendered_card_path.write_text(rendered_card, encoding="utf-8")
    print(f"[release] rendered card -> {rendered_card_path}", flush=True)

    api = HfApi(token=token)
    print(f"[release] ensuring dataset repo {repo_id} exists...", flush=True)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    print(f"[release] uploading {parquet}...", flush=True)
    api.upload_file(
        path_or_fileobj=str(parquet),
        path_in_repo=parquet.name,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"[release] uploading {meta}...", flush=True)
    api.upload_file(
        path_or_fileobj=str(meta),
        path_in_repo=meta.name,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("[release] uploading rendered README.md...", flush=True)
    api.upload_file(
        path_or_fileobj=str(rendered_card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"[release] PUBLISHED: https://huggingface.co/datasets/{repo_id}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--card", type=Path, default=DEFAULT_CARD)
    parser.add_argument("--zenodo", type=Path, default=DEFAULT_ZENODO)
    parser.add_argument(
        "--repo-id", type=str, default=None,
        help=(
            "HF dataset repo id (default: <HF_USER>/"
            f"{DEFAULT_REPO_SUFFIX} reading HF_USER from the env)."
        ),
    )
    parser.add_argument(
        "--publish", action="store_true",
        help="Push to HF Hub (default: dry-run only).",
    )
    parser.add_argument(
        "--zenodo-doi", type=str, default="<ZENODO_DOI>",
        help="Reserved Zenodo DOI to bake into the rendered card (publish mode).",
    )
    args = parser.parse_args(argv)

    # Resolve HF_USER from env first, then from .env via Settings.
    hf_user = os.environ.get("HF_USER", "")
    if not hf_user:
        try:
            settings = Settings()  # type: ignore[call-arg]
            hf_user = settings.HF_USER
        except Exception:
            hf_user = "<HF_USER>"
    repo_id = args.repo_id or f"{hf_user}/{DEFAULT_REPO_SUFFIX}"

    if not args.publish:
        return _dry_run(args.parquet, args.meta, args.card, args.zenodo, repo_id=repo_id)

    if hf_user == "<HF_USER>":
        print(
            "[release] FAIL: HF_USER env var not set; required for --publish",
            flush=True,
        )
        return 2
    return _publish(
        args.parquet, args.meta, args.card,
        repo_id=repo_id, hf_user=hf_user, zenodo_doi=args.zenodo_doi,
    )


if __name__ == "__main__":
    raise SystemExit(main())
