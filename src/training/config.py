"""Load and display training YAML configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_training_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = resolve_repo_path(config_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Training config does not exist: {resolved}")

    with resolved.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a mapping: {resolved}")

    return config, resolved


def format_config(config: dict[str, Any]) -> str:
    """Return a stable, readable config snapshot for logs."""
    return json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True)
