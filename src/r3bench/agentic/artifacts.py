"""Safe final-artifact mapping for public Agentic episodes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class ArtifactContractError(ValueError):
    """Raised when an exported final-artifact contract is unsafe."""


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    container_path: str
    sandbox_relative_path: str
    problem_label: str | None
    required: bool


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactContractError(f"{path.name} must contain an object")
    return value


def _expected_relative(container_path: str) -> str:
    if container_path.startswith("/app/solution_") and container_path.endswith(".cpp"):
        return container_path.removeprefix("/")
    if container_path == "/logs/artifacts/answer.txt":
        return "logs/artifacts/answer.txt"
    raise ArtifactContractError("unsupported public final-artifact path")


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactContractError("sandbox_relative_path must be non-empty")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ArtifactContractError("sandbox_relative_path escapes the episode sandbox")
    return path.as_posix()


class FinalArtifactManager:
    """Write only task-designated artifacts into one episode-local sandbox."""

    def __init__(self, task_dir: Path, output_dir: Path) -> None:
        config = _object(task_dir / "expected_artifacts.json")
        rows = config.get("artifacts")
        if not isinstance(rows, list) or not rows:
            raise ArtifactContractError("expected_artifacts.json requires artifacts")
        specs: list[ArtifactSpec] = []
        for row in rows:
            if not isinstance(row, dict):
                raise ArtifactContractError("artifact entries must be objects")
            container = row.get("container_path", row.get("path"))
            if not isinstance(container, str):
                raise ArtifactContractError("artifact container_path must be a string")
            relative = _safe_relative(
                row.get("sandbox_relative_path", _expected_relative(container))
            )
            if relative != _expected_relative(container):
                raise ArtifactContractError("artifact mapping differs from the public contract")
            label = row.get("problem_label")
            if label is not None and label not in tuple("ABCDEF"):
                raise ArtifactContractError("artifact problem_label is invalid")
            specs.append(
                ArtifactSpec(
                    container_path=container,
                    sandbox_relative_path=relative,
                    problem_label=label,
                    required=bool(row.get("required", False)),
                )
            )
        if len({row.container_path for row in specs}) != len(specs):
            raise ArtifactContractError("artifact container paths must be unique")
        self.sandbox_root = output_dir / "artifacts"
        self.specs = tuple(specs)
        self._by_container = {row.container_path: row for row in specs}

    def resolve(self, container_path: str) -> Path:
        try:
            spec = self._by_container[container_path]
        except KeyError as exc:
            raise ArtifactContractError("path is not a designated final artifact") from exc
        target = (self.sandbox_root / spec.sandbox_relative_path).resolve()
        try:
            target.relative_to(self.sandbox_root.resolve())
        except ValueError as exc:
            raise ArtifactContractError("artifact path escapes the episode sandbox") from exc
        return target

    def write(self, container_path: str, content: str) -> None:
        if not isinstance(content, str):
            raise ArtifactContractError("artifact content must be a string")
        target = self.resolve(container_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def manifest(self) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for spec in self.specs:
            target = self.resolve(spec.container_path)
            exists = target.is_file()
            rows.append(
                {
                    "container_path": spec.container_path,
                    "problem_label": spec.problem_label,
                    "required": spec.required,
                    "exists": exists,
                    "size_bytes": target.stat().st_size if exists else 0,
                    "sha256": (
                        hashlib.sha256(target.read_bytes()).hexdigest()
                        if exists
                        else None
                    ),
                    "artifact_relative_path": (
                        f"artifacts/{spec.sandbox_relative_path}" if exists else None
                    ),
                }
            )
        return {
            "schema_version": "1.0",
            "grade_after_episode": True,
            "correctness_feedback_exposed": False,
            "artifacts": rows,
        }


__all__ = ["ArtifactContractError", "ArtifactSpec", "FinalArtifactManager"]
