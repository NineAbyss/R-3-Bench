"""Convert sanitized Agentic final artifacts into ordinary saved outputs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


_PROBLEM_SECTION = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*|=+\s*)?\[?Problem\s+([A-F])\]?"
    r"\s*:?\s*(?:=+)?\s*$"
)
_PLAIN_ANSWER = re.compile(
    r"<answer\s*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL
)
_BOXED = re.compile(r"\\boxed\s*\{")


class AgenticScoringHandoffError(ValueError):
    """Raised when an Agentic artifact manifest is incomplete or unsafe."""


def _section_bodies(text: str) -> dict[str, str]:
    matches = tuple(_PROBLEM_SECTION.finditer(text))
    sections: dict[str, str] = {}
    duplicates: set[str] = set()
    for index, match in enumerate(matches):
        label = match.group(1).upper()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if label in sections:
            duplicates.add(label)
        else:
            sections[label] = body
    for label in duplicates:
        sections.pop(label, None)
    return sections


def _boxed_answers(text: str) -> tuple[str, ...]:
    answers: list[str] = []
    for match in _BOXED.finditer(text):
        start = match.end()
        depth = 1
        index = start
        while index < len(text) and depth:
            character = text[index]
            if character == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif character == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
            index += 1
        if depth == 0:
            answer = text[start : index - 1].strip()
            if answer:
                answers.append(answer)
    return tuple(answers)


def extract_agentic_answer_sections(
    domain: str,
    text: str,
    labels: tuple[str, ...],
) -> dict[str, str]:
    """Extract only answers satisfying the paper's domain-specific contract."""

    if domain not in {"math", "abstract_reasoning"}:
        raise AgenticScoringHandoffError(
            "answer sections apply only to Math and Abstract Reasoning"
        )
    if not isinstance(text, str):
        raise AgenticScoringHandoffError("answer artifact must be UTF-8 text")
    if not labels or any(label not in tuple("ABCDEF") for label in labels):
        raise AgenticScoringHandoffError("answer labels are invalid")
    sections = _section_bodies(text)
    if not sections and labels == ("A",):
        sections = {"A": text.strip()}
    answers: dict[str, str] = {}
    for label in labels:
        body = sections.get(label)
        if body is None:
            continue
        if domain == "math":
            candidates = _boxed_answers(body)
        else:
            candidates = tuple(
                match.group(1).strip()
                for match in _PLAIN_ANSWER.finditer(body)
                if match.group(1).strip()
            )
        if len(candidates) == 1:
            answers[label] = candidates[0]
    return answers


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AgenticScoringHandoffError(f"{path.name} must contain an object")
    return value


def _safe_artifact(output_dir: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AgenticScoringHandoffError("present artifact has no relative path")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("artifacts",):
        raise AgenticScoringHandoffError("artifact path is outside the public output")
    target = (output_dir / Path(*path.parts)).resolve()
    try:
        target.relative_to(output_dir.resolve())
    except ValueError as exc:
        raise AgenticScoringHandoffError("artifact path escapes output_dir") from exc
    return target


def collect_agentic_saved_outputs(output_dir: str | Path) -> tuple[dict[str, Any], ...]:
    root = Path(output_dir)
    summary = _object(root / "backend_summary.json")
    manifest = _object(root / "final_artifacts_manifest.json")
    binding = _object(root / "task_binding" / "public_problem_manifest.json")
    domain = summary.get("domain")
    if domain not in {"coding", "math", "abstract_reasoning"}:
        raise AgenticScoringHandoffError("backend summary has invalid domain")
    problems = binding.get("problems")
    artifacts = manifest.get("artifacts")
    if not isinstance(problems, list) or not isinstance(artifacts, list):
        raise AgenticScoringHandoffError("artifact or problem manifest is malformed")
    by_label: dict[str, Mapping[str, Any]] = {}
    for row in artifacts:
        if not isinstance(row, Mapping):
            raise AgenticScoringHandoffError("artifact rows must be objects")
        label = row.get("problem_label")
        if label is None and row.get("container_path") == "/logs/artifacts/answer.txt":
            label = "ALL"
        if not isinstance(label, str) or label in by_label:
            raise AgenticScoringHandoffError("artifact labels are invalid or duplicated")
        by_label[label] = row

    answer_sections: dict[str, str] = {}
    if domain != "coding":
        row = by_label.get("ALL")
        if row is not None and row.get("exists") is True:
            path = _safe_artifact(root, row.get("artifact_relative_path"))
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row.get("sha256"):
                raise AgenticScoringHandoffError("answer artifact digest mismatch")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = None
            if text is not None:
                expected_labels = tuple(
                    str(problem.get("problem_label"))
                    for problem in problems
                    if isinstance(problem, Mapping)
                )
                answer_sections = extract_agentic_answer_sections(
                    str(domain), text, expected_labels
                )

    rows: list[dict[str, Any]] = []
    provenance = {
        field: summary[field]
        for field in ("execution_id", "task_id", "model_key")
        if isinstance(summary.get(field), str) and summary[field]
    }
    for problem in problems:
        if not isinstance(problem, Mapping):
            raise AgenticScoringHandoffError("problem rows must be objects")
        problem_id = problem.get("problem_id")
        label = problem.get("problem_label")
        if not isinstance(problem_id, str) or label not in tuple("ABCDEF"):
            raise AgenticScoringHandoffError("public problem binding is invalid")
        parsed: str | None
        if domain == "coding":
            artifact = by_label.get(str(label))
            parsed = None
            if artifact is not None and artifact.get("exists") is True:
                path = _safe_artifact(root, artifact.get("artifact_relative_path"))
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != artifact.get("sha256"):
                    raise AgenticScoringHandoffError("Coding artifact digest mismatch")
                try:
                    parsed = raw.decode("utf-8")
                except UnicodeDecodeError:
                    parsed = None
        else:
            parsed = answer_sections.get(str(label))
        rows.append(
            {
                "domain": domain,
                "problem_id": problem_id,
                "parsed_answer": parsed,
                "source_setting": "agentic",
                **provenance,
            }
        )
    return tuple(rows)


def write_agentic_saved_outputs(output_dir: str | Path, target: str | Path) -> int:
    rows = collect_agentic_saved_outputs(output_dir)
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return len(rows)


__all__ = [
    "AgenticScoringHandoffError",
    "collect_agentic_saved_outputs",
    "extract_agentic_answer_sections",
    "write_agentic_saved_outputs",
]
