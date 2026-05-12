"""T6.2 - HF Hub model release acceptance.

DEVPLAN T6.2 acceptance: ``huggingface_hub loads weights end-to-end in
fresh venv``. We verify the equivalent locally:

* Model card exists, has well-formed YAML frontmatter with HF model-card
  required keys (license, pipeline_tag).
* The dry-run smoke-load (load best.pt into a fresh
  build_zoobot_dirichlet model instance) succeeds.
* run_config.json + calibrated_metrics.json are present (the sidecars
  shipped alongside the checkpoint).
* Release script's dry-run mode exits 0 + prints the publish checklist.

The actual ``--publish`` step is human-executed and not exercised here.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

CARD = Path("docs/model_card.md")
CHECKPOINT = Path("runs/m3_dirichlet/best.pt")
RUN_CONFIG = Path("runs/m3_dirichlet/run_config.json")
CALIBRATION = Path("runs/m3_dirichlet/calibrated_metrics.json")


def test_T6_2_model_card_exists_and_renders() -> None:
    """Card has YAML frontmatter with HF model-card required keys + body
    references the canonical architecture details."""
    assert CARD.is_file()
    src = CARD.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", src, re.DOTALL)
    assert m is not None, "model card lacks YAML frontmatter"
    front = m.group(1)
    for key in ("license:", "pipeline_tag:", "base_model:"):
        assert key in front, f"frontmatter missing {key!r}"
    # Body references the architecture + metrics we ship.
    assert "ConvNeXt-nano" in src
    assert "Dirichlet" in src
    assert "0.0883" in src or "0.088" in src  # macro MAE
    assert "0.93" in src                       # calibrated coverage
    # Loadable example block.
    assert "load_state_dict" in src
    assert "hf_hub_download" in src


@pytest.mark.skipif(
    not (CHECKPOINT.is_file() and RUN_CONFIG.is_file() and CALIBRATION.is_file()),
    reason=(
        "run T3.6 trainer + scripts/calibrate_dirichlet.py first to produce "
        "runs/m3_dirichlet/{best.pt,run_config.json,calibrated_metrics.json}"
    ),
)
def test_T6_2_run_config_records_canonical_architecture() -> None:
    """The shipped run_config.json identifies the model as the canonical
    Dirichlet architecture (T6.2 acceptance: fresh consumers know what
    they're loading)."""
    payload = json.loads(RUN_CONFIG.read_text(encoding="utf-8"))
    cfg = payload["config"]
    assert cfg["model"]["head"] == "dirichlet_multinomial"
    assert cfg["model"]["num_answers"] == 34
    assert cfg["model"]["encoder"] == "mwalmsley/zoobot-encoder-convnext_nano"


@pytest.mark.skipif(
    not (CHECKPOINT.is_file() and RUN_CONFIG.is_file() and CALIBRATION.is_file()),
    reason="run T3.6 trainer + calibration first",
)
def test_T6_2_release_script_dry_run_exits_zero() -> None:
    """The release script's dry-run mode (including the smoke-load) exits 0.

    Catches regressions where the checkpoint state_dict becomes
    incompatible with build_zoobot_dirichlet (e.g., schema drift,
    timm encoder API change) BEFORE a user hits the same issue
    downloading from the Hub.
    """
    # The subprocess imports the project's model module which needs torch
    # and timm. CI's [dev,m1,m1-serve] install includes torch but not
    # timm; skip rather than fail when running outside the [m1-train]
    # extra.
    pytest.importorskip("timm")
    result = subprocess.run(
        [sys.executable, "-m", "scripts.release_model_to_hf"],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"release_model_to_hf dry-run failed (exit {result.returncode})\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "smoke-loading" in result.stdout
    assert "OK: load OK" in result.stdout
    assert "NEXT STEPS" in result.stdout
