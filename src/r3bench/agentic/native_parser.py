"""Pure parser for the public Agentic native-tool protocol."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class AgenticActionKind(str, Enum):
    FOCUS = "focus"
    SHELVE = "shelve"
    BASH = "bash"
    WRITE_FINAL = "write_final"
    STATUS = "status"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class AgenticToolAction:
    kind: AgenticActionKind
    command: str | None = None
    problem_id: str | None = None
    artifact_path: str | None = None
    artifact_content: str | None = None

    def accounting_command(self) -> str:
        if self.kind == AgenticActionKind.FOCUS:
            return f"focus_problem {self.problem_id}"
        if self.kind == AgenticActionKind.SHELVE:
            return "shelve_problem"
        if self.kind == AgenticActionKind.BASH:
            return self.command or ""
        if self.kind == AgenticActionKind.WRITE_FINAL:
            return f"write_final_artifact {self.artifact_path}"
        if self.kind == AgenticActionKind.STATUS:
            return "contest_status"
        return "mark_task_complete"


@dataclass(frozen=True, slots=True)
class ParsedAgenticTurn:
    actions: tuple[AgenticToolAction, ...]
    protocol_errors: tuple[str, ...]


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _arguments(value: object) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("tool arguments must be an object")
    return value


def _parse_action(value: Mapping[str, Any]) -> AgenticToolAction:
    name = _text(value.get("name"), "tool name")
    arguments = _arguments(value.get("arguments"))
    if name in {"focus_problem", "focus"}:
        return AgenticToolAction(
            AgenticActionKind.FOCUS,
            problem_id=_text(
                arguments.get("problem_id", arguments.get("id")), "problem_id"
            ),
        )
    if name in {"shelve_problem", "shelve"}:
        return AgenticToolAction(AgenticActionKind.SHELVE)
    if name in {"bash", "bash_command", "run_command"}:
        return AgenticToolAction(
            AgenticActionKind.BASH,
            command=_text(
                arguments.get("command", arguments.get("keystrokes")), "command"
            ),
        )
    if name == "run_for_problem":
        return AgenticToolAction(
            AgenticActionKind.BASH,
            command=_text(
                arguments.get("command", arguments.get("keystrokes")), "command"
            ),
            problem_id=_text(
                arguments.get(
                    "problem_label",
                    arguments.get("problem_id", arguments.get("problem")),
                ),
                "problem_id",
            ),
        )
    if name in {"write_final", "write_final_artifact"}:
        content = arguments.get("content", "")
        if not isinstance(content, str):
            raise ValueError("artifact content must be a string")
        return AgenticToolAction(
            AgenticActionKind.WRITE_FINAL,
            artifact_path=_text(arguments.get("path"), "artifact path"),
            artifact_content=content,
        )
    if name in {"contest_status", "remaining_budget", "status"}:
        return AgenticToolAction(AgenticActionKind.STATUS)
    if name in {"mark_task_complete", "task_complete"}:
        return AgenticToolAction(AgenticActionKind.COMPLETE)
    if name in {
        "submit",
        "submit_answer",
        "submit_solution",
        "live_judge",
        "official_judge",
        "hidden_tests",
    }:
        return AgenticToolAction(AgenticActionKind.BASH, command=name)
    raise ValueError(f"unsupported native tool: {name}")


def parse_assistant_output(value: Mapping[str, Any]) -> ParsedAgenticTurn:
    """Parse only structured, visible tool calls; reasoning is ignored."""

    if not isinstance(value, Mapping):
        raise TypeError("Agentic provider output must be an object")
    raw_calls = value.get("tool_calls", ())
    if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, (str, bytes)):
        return ParsedAgenticTurn((), ("tool_calls_must_be_array",))
    actions: list[AgenticToolAction] = []
    errors: list[str] = []
    for index, raw in enumerate(raw_calls):
        if not isinstance(raw, Mapping):
            errors.append(f"tool_call_{index}_must_be_object")
            continue
        try:
            actions.append(_parse_action(raw))
        except ValueError as exc:
            errors.append(f"tool_call_{index}:{exc}")
    if not actions:
        errors.append("no_valid_tool_call")
    return ParsedAgenticTurn(tuple(actions), tuple(errors))


__all__ = [
    "AgenticActionKind",
    "AgenticToolAction",
    "ParsedAgenticTurn",
    "parse_assistant_output",
]
