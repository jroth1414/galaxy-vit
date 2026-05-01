"""T0.2 — galaxy_vit.config.Settings env-validation acceptance test."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from galaxy_vit.config import Settings

# Mirrors the keys enumerated in .env.example at the repo root.
REQUIRED_VARS: tuple[str, ...] = (
    "HF_USER",
    "HF_TOKEN",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "DATA_DIR",
)


def test_missing_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings() raises ValidationError when required env vars are absent.

    Also passes ``_env_file=None`` to suppress .env loading, so the test is
    hermetic regardless of whether the developer has populated a local .env.
    The message must mention every missing field name to help diagnose
    misconfiguration.
    """
    for var in REQUIRED_VARS:
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)  # type: ignore[call-arg]

    error_text = str(excinfo.value)
    for var in REQUIRED_VARS:
        assert var in error_text, (
            f"validation error should mention missing field {var}; got:\n{error_text}"
        )
