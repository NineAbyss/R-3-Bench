"""Secret-free configuration loading for later evaluator phases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from r3bench.resource_paths import resolve_path


class ConfigError(ValueError):
    """Raised when a public evaluator configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML mapping without resolving credentials."""

    file_path = resolve_path(path)
    suffix = file_path.suffix.lower()
    try:
        text = file_path.read_text(encoding="utf-8")
        if suffix == ".json":
            value = json.loads(text)
        elif suffix in {".yaml", ".yml"}:
            import yaml

            value = yaml.safe_load(text)
        else:
            raise ConfigError(f"unsupported configuration extension: {suffix}")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot load configuration {file_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("configuration root must be a mapping")
    return value
