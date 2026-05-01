"""Galaxy-ViT runtime configuration loaded from env vars / .env file.

All required variables are enumerated in `.env.example` at the repo root.
Construction of `Settings()` raises `pydantic.ValidationError` if any are
missing in both the process environment and the loaded `.env` file.

CLI usage (per README quickstart §2):

    python -m galaxy_vit.config

Prints each setting on its own `KEY=VALUE` line with secret values masked,
and exits nonzero if validation fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed runtime settings.

    Reads from the process environment first, then `.env` in the working
    directory. Tests construct `Settings(_env_file=None)` to suppress the
    file lookup so they remain hermetic regardless of developer state.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    HF_USER: str = Field(min_length=1, description="Hugging Face username for repo IDs.")
    HF_TOKEN: SecretStr = Field(description="Hugging Face write token (write scope).")
    WANDB_API_KEY: SecretStr = Field(description="Weights & Biases API key.")
    WANDB_ENTITY: str = Field(min_length=1, description="W&B entity (user or team).")
    DATA_DIR: Path = Field(description="Local data cache directory (must be writable).")

    def redacted(self) -> dict[str, str]:
        """Return a dict suitable for logging; secret values are masked with ``***``."""
        return {
            "HF_USER": self.HF_USER,
            "HF_TOKEN": "***" if self.HF_TOKEN.get_secret_value() else "",
            "WANDB_API_KEY": "***" if self.WANDB_API_KEY.get_secret_value() else "",
            "WANDB_ENTITY": self.WANDB_ENTITY,
            "DATA_DIR": str(self.DATA_DIR),
        }


def main() -> int:
    """CLI entrypoint: validate Settings and print redacted; exit nonzero on failure."""
    try:
        settings = Settings()  # type: ignore[call-arg]
    except Exception as exc:
        print(f"galaxy_vit.config: settings invalid - {exc}", file=sys.stderr)
        return 1
    for key, value in settings.redacted().items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
