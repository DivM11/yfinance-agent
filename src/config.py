"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf


def load_config(base_path: Path | None = None) -> Dict[str, Any]:
    """Load config.yml and merge in a local secrets file if present."""
    root = base_path or Path(__file__).resolve().parents[1]
    config_path = root / "config.yml"
    config = OmegaConf.load(config_path)

    secrets_path = root / config.secrets.file
    if secrets_path.exists():
        secrets = OmegaConf.load(secrets_path)
        config = OmegaConf.merge(config, secrets)

    OmegaConf.resolve(config)
    return OmegaConf.to_container(config, resolve=True)
