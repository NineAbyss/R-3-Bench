"""Non-secret-reporting safety scan for release source and generated outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


ScanKind = Literal["source", "outputs"]
_CREDENTIAL = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{16,}\b|\bhf_[A-Za-z0-9]{16,}\b|"
    r"\bolp_[A-Za-z0-9]{16,}\b|Bearer\s+[A-Za-z0-9._~+/-]{16,})"
)
_MACHINE_PATH = re.compile(
    r"(?<![A-Za-z0-9])/(?:home|mnt|data|Users|root)/[^\s`\"']+"
)
_PRIVATE_ENDPOINT = re.compile(
    r"(?i)https?://(?:localhost|127(?:\.\d+){3}|10(?:\.\d+){3}|"
    r"192\.168(?:\.\d+){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d+){2}|"
    r"[A-Za-z0-9.-]+\.(?:local|internal))(?::\d+)?(?:/|$)"
)
_MODEL_FORK = re.compile(
    r"(?i)(?:glm|hunyuan|gpt|claude).*(?:runner|adapter)|"
    r"(?:runner|adapter).*(?:glm|hunyuan|gpt|claude)"
)
_FORBIDDEN_OUTPUT_KEYS = frozenset(
    {
        "provider_headers",
        "provider_request_id",
        "request_headers",
        "authorization",
        "raw_trajectory",
        "raw_trajectories",
        "raw_log",
        "raw_logs",
        "hidden_tests",
        "hidden_testcases",
        "asset_root",
        "assets_root",
        "service_url",
    }
)
_FORBIDDEN_ARTIFACT_NAMES = re.compile(
    r"(?i)(?:raw[_-]trajector|raw[_-]log|provider[_-]headers?|"
    r"provider[_-]request[_-]id|hidden[_-]tests?)"
)
_TEXT_SUFFIXES = {
    ".cfg",
    ".cff",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
_SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "venv",
}


@dataclass(frozen=True, slots=True)
class SafetyFinding:
    category: str
    path: str
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "path": self.path,
            "line": self.line,
        }


@dataclass(frozen=True, slots=True)
class SafetyScanResult:
    kind: ScanKind
    scanned_file_count: int
    findings: tuple[SafetyFinding, ...]
    guard_regex_files: tuple[str, ...]
    synthetic_fixture_findings: tuple[SafetyFinding, ...]

    @property
    def passed(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "kind": self.kind,
            "passed": self.passed,
            "scanned_file_count": self.scanned_file_count,
            "findings": [finding.to_dict() for finding in self.findings],
            "guard_regex_files": list(self.guard_regex_files),
            "synthetic_fixture_findings": [
                finding.to_dict() for finding in self.synthetic_fixture_findings
            ],
        }


def _is_synthetic(path: Path) -> bool:
    lowered = {part.lower() for part in path.parts}
    return "examples" in lowered or "fixtures" in lowered


def _lines(text: str, pattern: re.Pattern[str], category: str, path: str) -> list[SafetyFinding]:
    return [
        SafetyFinding(category=category, path=path, line=line_number)
        for line_number, line in enumerate(text.splitlines(), start=1)
        if pattern.search(line)
    ]


def _walk_keys(value: Any, path: str, line: int | None = None) -> list[SafetyFinding]:
    findings: list[SafetyFinding] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_OUTPUT_KEYS:
                findings.append(
                    SafetyFinding("forbidden_output_field", path, line)
                )
            findings.extend(_walk_keys(item, path, line))
    elif isinstance(value, list):
        for item in value:
            findings.extend(_walk_keys(item, path, line))
    return findings


def _json_key_findings(path: Path, relative: str) -> list[SafetyFinding]:
    if path.suffix == ".json":
        try:
            return _walk_keys(json.loads(path.read_text(encoding="utf-8")), relative)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return []
    if path.suffix == ".jsonl":
        findings: list[SafetyFinding] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            return []
        for number, line in enumerate(lines, start=1):
            try:
                findings.extend(_walk_keys(json.loads(line), relative, number))
            except json.JSONDecodeError:
                continue
        return findings
    return []


def scan_tree(root: str | Path, *, kind: ScanKind = "source") -> SafetyScanResult:
    """Scan without retaining or returning matched values."""

    base = Path(root).resolve()
    findings: list[SafetyFinding] = []
    synthetic: list[SafetyFinding] = []
    guards: list[str] = []
    scanned = 0
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(base)
        if any(part in _SKIP_PARTS for part in relative_path.parts):
            continue
        relative = relative_path.as_posix()
        if path.name != ".env.example" and (
            path.name == ".env" or path.name.startswith(".env.")
        ):
            findings.append(SafetyFinding("environment_file", relative))
            continue
        if _FORBIDDEN_ARTIFACT_NAMES.search(path.name):
            findings.append(SafetyFinding("forbidden_artifact_name", relative))
        if kind == "source" and _MODEL_FORK.search(path.name):
            findings.append(SafetyFinding("model_specific_runner_fork", relative))
        if path.suffix not in _TEXT_SUFFIXES and path.name not in {
            ".env.example",
            ".gitignore",
        }:
            continue
        scanned += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(
            token in text
            for token in ("_CREDENTIAL", "_SECRET", "SECRET_VALUE", "FORBIDDEN_")
        ):
            guards.append(relative)
        local = (
            _lines(text, _CREDENTIAL, "credential_value", relative)
            + _lines(text, _MACHINE_PATH, "machine_path", relative)
            + _lines(text, _PRIVATE_ENDPOINT, "private_endpoint", relative)
        )
        if kind == "outputs":
            local.extend(_json_key_findings(path, relative))
        if _is_synthetic(relative_path):
            synthetic.extend(local)
        else:
            findings.extend(local)
    return SafetyScanResult(
        kind=kind,
        scanned_file_count=scanned,
        findings=tuple(findings),
        guard_regex_files=tuple(sorted(set(guards))),
        synthetic_fixture_findings=tuple(synthetic),
    )


__all__ = ["SafetyFinding", "SafetyScanResult", "scan_tree"]
