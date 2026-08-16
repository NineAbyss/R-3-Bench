"""Network-free offline Agentic mock/replay loop and provider contracts."""

from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from r3bench.agentic.action_accounting import (
    ActionClass,
    ActionDecision,
    apply_budget_decision,
    policy_from_name,
)
from r3bench.agentic.artifacts import ArtifactContractError, FinalArtifactManager
from r3bench.agentic.budget import ActionBudget
from r3bench.agentic.native_parser import (
    AgenticActionKind,
    AgenticToolAction,
    parse_assistant_output,
)
from r3bench.agentic.scope import AgenticScopeState


class AgenticRuntimeError(RuntimeError):
    """Raised when a public Agentic episode violates its task contract."""


class AgenticProviderExhausted(AgenticRuntimeError):
    """Raised when a finite mock/replay provider has no next turn."""


@dataclass(frozen=True, slots=True)
class AgenticEpisodeConfig:
    task_dir: Path
    output_dir: Path
    model_key: str
    max_turns: int = 10
    protocol_failure_limit: int = 3

    def __post_init__(self) -> None:
        if not self.model_key.strip():
            raise ValueError("model_key must be non-empty")
        if self.max_turns <= 0 or self.protocol_failure_limit <= 0:
            raise ValueError("turn and protocol-failure limits must be positive")


@dataclass(frozen=True, slots=True)
class AgenticEpisodeResult:
    status: str
    stop_reason: str
    task_id: str
    suite_id: str
    domain: str
    mode: str
    model_key: str
    turns: int
    protocol_failures: int
    budget_limit: int | None
    budget_used: int
    budget_remaining: int | None
    blocked_attempts: int
    completed: bool


class AgenticProvider(ABC):
    provider_mode: str

    @abstractmethod
    def next_response(self, *, turn: int) -> Mapping[str, Any]:
        """Return one visible native-tool response without provider metadata."""


class ReplayAgenticProvider(AgenticProvider):
    provider_mode = "replay"

    def __init__(self, path: Path) -> None:
        rows: list[Mapping[str, Any]] = []
        forbidden = {
            "provider_headers",
            "provider_request_id",
            "request_headers",
            "api_key",
            "reasoning_content",
        }
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AgenticRuntimeError(
                    f"replay row {line_number} must be an object"
                )
            if forbidden.intersection(value):
                raise AgenticRuntimeError(
                    f"replay row {line_number} contains forbidden provider metadata"
                )
            rows.append(value)
        if not rows:
            raise AgenticRuntimeError("replay file contains no responses")
        self._rows = rows
        self._index = 0

    def next_response(self, *, turn: int) -> Mapping[str, Any]:
        del turn
        if self._index >= len(self._rows):
            raise AgenticProviderExhausted("replay responses exhausted")
        row = self._rows[self._index]
        self._index += 1
        return row


class MockAgenticProvider(AgenticProvider):
    provider_mode = "mock"

    def __init__(self, responses: Sequence[Mapping[str, Any]]) -> None:
        if not responses:
            raise ValueError("mock responses cannot be empty")
        self._rows = tuple(dict(row) for row in responses)
        self._index = 0

    def next_response(self, *, turn: int) -> Mapping[str, Any]:
        del turn
        if self._index >= len(self._rows):
            raise AgenticProviderExhausted("mock responses exhausted")
        row = self._rows[self._index]
        self._index += 1
        return row


def default_mock_responses(domain: str) -> tuple[Mapping[str, Any], ...]:
    artifact = (
        "/app/solution_A.cpp"
        if domain == "coding"
        else "/logs/artifacts/answer.txt"
    )
    if domain == "coding":
        content = "int main(){return 0;}\n"
    elif domain == "math":
        content = "## Problem A\n\\boxed{synthetic answer}\n"
    else:
        content = "## Problem A\n<answer>synthetic answer</answer>\n"
    return (
        {
            "tool_calls": [
                {"name": "focus_problem", "arguments": {"problem_id": "A"}},
                {"name": "contest_status", "arguments": {}},
            ]
        },
        {
            "tool_calls": [
                {
                    "name": "write_final",
                    "arguments": {"path": artifact, "content": content},
                },
                {"name": "mark_task_complete", "arguments": {}},
            ]
        },
    )


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AgenticRuntimeError(f"{path.name} must contain an object")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, allow_nan=False, sort_keys=True)
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _blocked_scope_decision(
    command: str,
    classified_as: ActionClass,
    budget: ActionBudget,
    reason: str,
) -> ActionDecision:
    before = budget.remaining
    budget.record_blocked()
    return ActionDecision(
        command=command,
        action_class=ActionClass.BLOCKED,
        classified_as=classified_as,
        allowed=False,
        counted=classified_as == ActionClass.COUNTED,
        executed=False,
        budget_consumed=0,
        budget_before=before,
        budget_after=budget.remaining,
        reason=reason,
    )


def _action_record(
    *,
    turn: int,
    index: int,
    action: AgenticToolAction,
    decision: ActionDecision,
    active_problem_id: str | None,
) -> dict[str, Any]:
    return {
        "turn": turn,
        "action_index": index,
        "action_kind": action.kind.value,
        "allowed": decision.allowed,
        "counted": decision.counted,
        "budget_consumed": decision.budget_consumed,
        "budget_before": decision.budget_before,
        "budget_after": decision.budget_after,
        "reason": decision.reason,
        "active_problem_id": active_problem_id,
        "correctness_feedback": False,
    }


def run_agentic_episode(
    config: AgenticEpisodeConfig,
    provider: AgenticProvider,
) -> AgenticEpisodeResult:
    """Run the explicitly non-paper-equivalent offline test state machine."""

    required = (
        "instruction.md",
        "task_config.json",
        "budget_config.json",
        "expected_artifacts.json",
        "public_problem_manifest.json",
    )
    missing = [name for name in required if not (config.task_dir / name).is_file()]
    if missing:
        raise AgenticRuntimeError(f"task directory is missing: {', '.join(missing)}")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise AgenticRuntimeError("episode output directory must be empty")
    task = _object(config.task_dir / "task_config.json")
    budget_config = _object(config.task_dir / "budget_config.json")
    labels = task.get("problem_labels")
    if not isinstance(labels, dict) or tuple(labels) not in {("A",), tuple("ABCDEF")}:
        raise AgenticRuntimeError("task problem_labels must be A or A through F")
    if any(not isinstance(value, str) or not value for value in labels.values()):
        raise AgenticRuntimeError("task problem IDs must be non-empty strings")
    budget_limit = budget_config.get("counted_action_budget")
    if budget_limit is not None and (
        isinstance(budget_limit, bool)
        or not isinstance(budget_limit, int)
        or budget_limit < 0
    ):
        raise AgenticRuntimeError("counted_action_budget is invalid")
    policy_name = str(budget_config.get("policy", ""))
    policy = policy_from_name(policy_name)
    budget = ActionBudget(budget_limit)
    scope = AgenticScopeState(
        valid_problem_ids=frozenset(str(value) for value in labels.values()),
        problem_labels={str(key): str(value) for key, value in labels.items()},
    )
    artifacts = FinalArtifactManager(config.task_dir, config.output_dir)
    action_rows: list[dict[str, Any]] = []
    scope_rows: list[dict[str, Any]] = []
    protocol_failures = 0
    completed = False
    stop_reason = "max_turns"
    turns = 0

    for turn in range(1, config.max_turns + 1):
        turns = turn
        try:
            raw = provider.next_response(turn=turn)
        except AgenticProviderExhausted:
            stop_reason = "provider_exhausted"
            break
        parsed = parse_assistant_output(raw)
        if parsed.protocol_errors:
            protocol_failures += 1
        if protocol_failures >= config.protocol_failure_limit:
            stop_reason = "protocol_failure_limit"
            break
        for index, action in enumerate(parsed.actions, start=1):
            if action.kind == AgenticActionKind.BASH and action.problem_id is not None:
                try:
                    scope.focus_problem(action.problem_id)
                    scope_rows.append(
                        {
                            "turn": turn,
                            "action_index": index,
                            "event": "transport_focus",
                            "active_problem_id": scope.active_problem_id,
                        }
                    )
                except ValueError:
                    pass
            command = action.accounting_command()
            classified = policy.classify_action(command)
            scope_decision = scope.authorize_action(classified, command)
            if not scope_decision.allowed:
                decision = _blocked_scope_decision(
                    command, classified, budget, scope_decision.reason
                )
            elif action.kind == AgenticActionKind.WRITE_FINAL:
                try:
                    artifacts.resolve(action.artifact_path or "")
                except ArtifactContractError:
                    decision = _blocked_scope_decision(
                        command,
                        classified,
                        budget,
                        "non_designated_artifact_write_blocked",
                    )
                else:
                    decision = apply_budget_decision(command, budget, policy)
            else:
                decision = apply_budget_decision(command, budget, policy)

            if decision.allowed:
                if action.kind == AgenticActionKind.FOCUS:
                    scope.focus_problem(action.problem_id or "")
                    scope_rows.append(
                        {
                            "turn": turn,
                            "action_index": index,
                            "event": "focus",
                            "active_problem_id": scope.active_problem_id,
                        }
                    )
                elif action.kind == AgenticActionKind.SHELVE:
                    previous = scope.active_problem_id
                    scope.shelve_problem()
                    scope_rows.append(
                        {
                            "turn": turn,
                            "action_index": index,
                            "event": "shelve",
                            "previous_problem_id": previous,
                            "active_problem_id": None,
                        }
                    )
                elif action.kind == AgenticActionKind.WRITE_FINAL:
                    artifacts.write(
                        action.artifact_path or "", action.artifact_content or ""
                    )
                elif action.kind == AgenticActionKind.COMPLETE:
                    completed = True
            action_rows.append(
                _action_record(
                    turn=turn,
                    index=index,
                    action=action,
                    decision=decision,
                    active_problem_id=scope.active_problem_id,
                )
            )
            if completed:
                stop_reason = "task_complete"
                break
        if completed:
            break

    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(config.output_dir / "action_records.jsonl", action_rows)
    _write_jsonl(config.output_dir / "scope_records.jsonl", scope_rows)
    _write_json(
        config.output_dir / "public_action_log.json",
        {
            "schema_version": "1.0",
            "budget": budget.limit,
            "used": budget.used,
            "remaining": budget.remaining,
            "blocked_attempts": budget.blocked_attempts,
            "policy": policy_name,
            "action_attempts": len(action_rows),
            "correctness_feedback_exposed": False,
        },
    )
    _write_json(config.output_dir / "final_artifacts_manifest.json", artifacts.manifest())
    binding = config.output_dir / "task_binding"
    binding.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        config.task_dir / "public_problem_manifest.json",
        binding / "public_problem_manifest.json",
    )
    result = AgenticEpisodeResult(
        status="completed" if completed else "stopped",
        stop_reason=stop_reason,
        task_id=str(task.get("task_id", "")),
        suite_id=str(task.get("suite_id", "")),
        domain=str(task.get("domain", "")),
        mode=str(task.get("mode", "")),
        model_key=config.model_key,
        turns=turns,
        protocol_failures=protocol_failures,
        budget_limit=budget.limit,
        budget_used=budget.used,
        budget_remaining=budget.remaining,
        blocked_attempts=budget.blocked_attempts,
        completed=completed,
    )
    summary = {
        "schema_version": "1.0",
        **asdict(result),
        "backend_profile": "offline_mock_replay_v1",
        "paper_equivalent_runtime": False,
        "provider_mode": provider.provider_mode,
        "executor_mode": "no_os_execution",
        "model_api_called": False,
        "os_commands_executed": False,
        "network_called": False,
        "container_runtime_called": False,
        "external_verifier_called": False,
        "scorer_called": False,
        "correctness_feedback_exposed": False,
        "raw_trajectory_saved": False,
    }
    _write_json(config.output_dir / "backend_summary.json", summary)
    return result


__all__ = [
    "AgenticEpisodeConfig",
    "AgenticEpisodeResult",
    "AgenticProvider",
    "AgenticProviderExhausted",
    "AgenticRuntimeError",
    "MockAgenticProvider",
    "ReplayAgenticProvider",
    "default_mock_responses",
    "run_agentic_episode",
]
