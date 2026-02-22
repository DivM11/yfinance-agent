"""Configuration loading utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from omegaconf import OmegaConf


def load_config(base_path: Path | None = None) -> Dict[str, Any]:
    """Load config.yml and merge in environment-based secrets if present."""
    root = base_path or Path(__file__).resolve().parents[1]
    dotenv_path = root / ".env"
    load_dotenv(dotenv_path)
    config_path = root / "config.yml"
    config = OmegaConf.load(config_path)

    key_env_var = "OPENROUTER_API_KEY"
    try:
        key_env_var = config.openrouter.api.key_env_var
    except (AttributeError, KeyError, TypeError):
        pass

    api_key = os.getenv(key_env_var)
    if api_key:
        config.openrouter.api.api_key = api_key

    OmegaConf.resolve(config)
    return OmegaConf.to_container(config, resolve=True)
