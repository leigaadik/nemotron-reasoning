"""Recursive YAML configuration loading with repository-relative paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def resolve_path(path: str | Path, *, relative_to: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    base = relative_to or REPO_ROOT
    return (base / path).resolve()


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    resolved = resolve_path(path)
    with resolved.open(encoding="utf-8") as stream:
        current = yaml.safe_load(stream) or {}
    if not isinstance(current, dict):
        raise ValueError(f"Config must be a mapping: {resolved}")
    parent_value = current.pop("extends", None)
    if parent_value:
        parent_path = resolve_path(parent_value, relative_to=resolved.parent)
        parent, _ = load_config(parent_path)
        current = deep_merge(parent, current)
    return _expand(current), resolved


def format_config(config: dict[str, Any]) -> str:
    return json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True)
