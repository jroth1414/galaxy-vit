"""T5.3 - HF dataset release acceptance.

DEVPLAN T5.3 acceptance: ``dataset card renders; Zenodo DOI reserved``.

The Zenodo DOI is reserved manually (web form, human action). That
piece can't be machine-verified in a unit test. The pieces that CAN
be verified in CI:

* The dataset card file exists, parses as Markdown + YAML frontmatter,
  and uses no broken placeholders the upload script wouldn't substitute.
* The Zenodo metadata template exists and is valid JSON with the
  required Zenodo fields (title, upload_type, description, creators,
  access_right, license).
* The release parquet + sidecar exist locally and the sidecar SHA-256
  matches the parquet's content (so the upload won't silently push a
  stale checksum).
* The release script dry-run mode exits 0 (validates everything
  without touching the network).

The actual ``--publish`` step is gated behind ``HF_TOKEN`` + a manual
flag; not exercised here.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

CARD = Path("docs/dataset_card.md")
ZENODO = Path("docs/zenodo_metadata.json")
PARQUET = Path("releases/gz_desi_dirichlet_v1.parquet")
META = Path("releases/gz_desi_dirichlet_v1.meta.json")
RELEASE_SCRIPT = Path("scripts/release_to_hf.py")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_T5_3_dataset_card_exists_and_renders() -> None:
    """The card file exists and has a well-formed YAML frontmatter."""
    assert CARD.is_file()
    src = CARD.read_text(encoding="utf-8")
    # YAML frontmatter delimited by --- ... ---  at the top.
    m = re.match(r"^---\n(.*?)\n---\n", src, re.DOTALL)
    assert m is not None, "no YAML frontmatter at the top of the dataset card"
    front = m.group(1)
    # Required keys per HF dataset-card spec.
    for key in ("license:", "task_categories:", "tags:"):
        assert key in front, f"frontmatter missing {key!r}"
    # Body has the table of canonical alpha indices (sanity).
    assert "alpha_0" in src and "alpha_33" in src
    assert "smooth-or-featured" in src
    # Placeholders are present (intentionally; substituted at upload time).
    assert "<HF_USER>" in src
    assert "<ZENODO_DOI>" in src


def test_T5_3_zenodo_metadata_valid_json() -> None:
    """The Zenodo metadata template is valid JSON with required fields."""
    assert ZENODO.is_file()
    payload = json.loads(ZENODO.read_text(encoding="utf-8"))
    md = payload["metadata"]
    for field in ("title", "upload_type", "description", "creators", "access_right", "license"):
        assert field in md, f"Zenodo metadata missing required field {field!r}"
    assert md["upload_type"] == "dataset"
    assert md["access_right"] == "open"
    # Workflow steps documented for the human.
    assert isinstance(payload["_publish_workflow"], list)
    assert any("zenodo.org" in s.lower() for s in payload["_publish_workflow"])


@pytest.mark.skipif(
    not (PARQUET.is_file() and META.is_file()),
    reason=(
        "run T5.1 first: python -m scripts.run_inference_pass --mode full"
    ),
)
def test_T5_3_release_artifacts_present_and_checksum_matches() -> None:
    """Parquet + sidecar exist; SHA-256 in sidecar matches a fresh recompute.

    Catches stale checksums in the sidecar that would cause the HF
    upload to publish unverifiable provenance.
    """
    meta = json.loads(META.read_text(encoding="utf-8"))
    recorded = str(meta["sha256"])
    actual = _sha256(PARQUET)
    assert actual == recorded, (
        f"sha256 drift:\n  recorded={recorded}\n  actual  ={actual}"
    )


@pytest.mark.skipif(
    not (PARQUET.is_file() and META.is_file()),
    reason="run T5.1 first",
)
def test_T5_3_release_script_dry_run_exits_zero() -> None:
    """The release script's dry-run mode validates everything and exits 0.

    Functional regression check: if the script grows new validation
    logic that fails on the current artefact set, this test catches
    it before the user runs the publish step.
    """
    result = subprocess.run(
        [sys.executable, "-m", "scripts.release_to_hf"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"release_to_hf dry-run failed with exit {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # The dry-run prints a checklist for the human; sanity-check the
    # next-steps section is present so we don't regress to silent.
    assert "NEXT STEPS" in result.stdout
    assert "Zenodo" in result.stdout
