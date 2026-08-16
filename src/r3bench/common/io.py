"""Small strict JSON helpers used by the public evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from r3bench.resource_paths import resolve_path


class InputFormatError(ValueError):
    """Raised when a public data file is not strict UTF-8 JSON/JSONL."""


def _reject_nonfinite(value: str) -> None:
    raise InputFormatError(f"non-finite JSON value is not allowed: {value}")


def read_json(path: str | Path) -> Any:
    file_path = resolve_path(path)
    try:
        return json.loads(
            file_path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputFormatError(f"cannot read JSON from {file_path}: {exc}") from exc


def parse_jsonl_bytes(data: bytes, *, source: str | Path) -> list[dict[str, Any]]:
    """Parse one immutable UTF-8 JSONL snapshot."""

    file_path = Path(source)
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise InputFormatError(
            f"cannot read UTF-8 JSONL from {file_path}: {exc}"
        ) from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line, parse_constant=_reject_nonfinite)
        except json.JSONDecodeError as exc:
            raise InputFormatError(
                f"invalid JSON object at {file_path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise InputFormatError(
                f"expected a JSON object at {file_path}:{line_number}"
            )
        rows.append(value)
    if not rows:
        raise InputFormatError(f"JSONL file contains no records: {file_path}")
    return rows


def read_jsonl_snapshot(path: str | Path) -> tuple[bytes, list[dict[str, Any]]]:
    """Read JSONL once and return the exact bytes plus parsed rows."""

    file_path = resolve_path(path)
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise InputFormatError(
            f"cannot read UTF-8 JSONL from {file_path}: {exc}"
        ) from exc
    return data, parse_jsonl_bytes(data, source=file_path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl_snapshot(path)[1]
