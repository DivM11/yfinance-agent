"""Unit tests for config loading."""

from pathlib import Path

from src.config import load_config


def _write_config(root: Path) -> None:
    config_text = (
        "app:\n"
        "  layout: \"wide\"\n"
        "secrets:\n"
        "  file: \".secrets.yml\"\n"
        "openrouter:\n"
        "  model: \"openrouter/auto\"\n"
    )
    (root / "config.yml").write_text(config_text, encoding="utf-8")


def test_load_config_without_secrets(tmp_path: Path) -> None:
    _write_config(tmp_path)

    config = load_config(base_path=tmp_path)

    assert config["app"]["layout"] == "wide"
    assert "api_key" not in config.get("openrouter", {})


def test_load_config_with_secrets(tmp_path: Path) -> None:
    _write_config(tmp_path)
    secrets_text = "openrouter:\n  api_key: \"test-key\"\n"
    (tmp_path / ".secrets.yml").write_text(secrets_text, encoding="utf-8")

    config = load_config(base_path=tmp_path)

    assert config["openrouter"]["api_key"] == "test-key"
