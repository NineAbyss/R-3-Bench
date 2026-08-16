#!/usr/bin/env python3
"""Simulate Agentic accounting and scope transitions without executing commands."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3bench.agentic.action_accounting import (
    ActionClass,
    ActionDecision,
    apply_budget_decision,
    policy_from_name,
)
from r3bench.agentic.budget import ActionBudget
from r3bench.agentic.scope import AgenticScopeState


class AgenticDryRunError(ValueError):
    """Raised when an exported dry-run task is missing or malformed."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AgenticDryRunError(f"{path.name} must contain a JSON object")
    return value


def _resolve_task_dir(path: Path) -> Path:
    if (path / "task_config.json").is_file():
        return path
    children = sorted(
        child for child in path.iterdir() if (child / "task_config.json").is_file()
    )
    if len(children) != 1:
        raise AgenticDryRunError(
            "task-dir must be one exported task or contain exactly one exported task"
        )
    return children[0]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _scope_blocked_decision(
    command: str, budget: ActionBudget, reason: str
) -> ActionDecision:
    before = budget.remaining
    budget.record_blocked()
    return ActionDecision(
        command=command,
        action_class=ActionClass.BLOCKED,
        classified_as=ActionClass.COUNTED,
        allowed=False,
        counted=True,
        executed=False,
        budget_consumed=0,
        budget_before=before,
        budget_after=budget.remaining,
        reason=reason,
    )


def _simulation_commands(
    problem_ids: list[str], first_artifact: str
) -> list[str]:
    return [
        "contest_status",
        "python3 -c 'print(1)'",
        f"focus_problem {problem_ids[0]}",
        "python3 -c 'print(1)'",
        "printf 'scratch' > scratch.txt",
        "g++ scratch.cpp -o scratch",
        f"printf '// candidate' > {first_artifact}",
        "./scratch",
        "shelve_problem",
        f"focus_problem {problem_ids[1]}",
        "grep value scratch.txt",
        f"submit_solution {first_artifact}",
        "mark_task_complete",
    ]


def run_dryrun(task_root: Path, output_dir: Path) -> dict[str, Any]:
    """Run a fixed fake action sequence. No command is passed to the OS."""

    task_dir = _resolve_task_dir(task_root)
    task_config = _read_json(task_dir / "task_config.json")
    budget_config = _read_json(task_dir / "budget_config.json")
    artifact_config = _read_json(task_dir / "expected_artifacts.json")
    labels = task_config.get("problem_labels")
    if not isinstance(labels, dict) or tuple(labels) != tuple("ABCDEF"):
        raise AgenticDryRunError("task problem_labels must contain A through F in order")
    problem_ids = list(labels.values())
    if any(not isinstance(value, str) or not value for value in problem_ids):
        raise AgenticDryRunError("task problem IDs must be non-empty strings")
    budget_limit = budget_config.get("counted_action_budget")
    if isinstance(budget_limit, bool) or not isinstance(budget_limit, int):
        raise AgenticDryRunError("task counted_action_budget must be an integer")
    artifacts = artifact_config.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AgenticDryRunError("task expected_artifacts must be non-empty")
    first_artifact = artifacts[0].get("container_path", artifacts[0].get("path"))
    if not isinstance(first_artifact, str) or not first_artifact:
        raise AgenticDryRunError("task artifact path must be a non-empty string")

    budget = ActionBudget(limit=budget_limit)
    policy = policy_from_name(str(budget_config.get("policy", "")))
    scope = AgenticScopeState(
        valid_problem_ids=frozenset(problem_ids),
        problem_labels={str(label): str(problem_id) for label, problem_id in labels.items()},
    )
    budget_log: list[dict[str, Any]] = []
    scope_log: list[dict[str, Any]] = []

    for step, command in enumerate(
        _simulation_commands(problem_ids, first_artifact), start=1
    ):
        classification = policy.classify_action(command)
        scope_decision = scope.authorize_action(classification, command)
        if not scope_decision.allowed:
            decision = _scope_blocked_decision(
                command, budget, scope_decision.reason
            )
            scope_log.append(
                {
                    "step": step,
                    "event": "blocked",
                    "active_problem_id": scope.active_problem_id,
                    "reason": scope_decision.reason,
                }
            )
        else:
            decision = apply_budget_decision(command, budget, policy)
            tokens = tuple(shlex.split(command))
            if decision.allowed and classification == ActionClass.FREE_BOOKKEEPING:
                if tokens[0] == "focus_problem":
                    scope.focus_problem(tokens[1])
                    scope_log.append(
                        {
                            "step": step,
                            "event": "focus",
                            "active_problem_id": scope.active_problem_id,
                        }
                    )
                else:
                    previous = scope.active_problem_id
                    scope.shelve_problem()
                    scope_log.append(
                        {
                            "step": step,
                            "event": "shelve",
                            "previous_problem_id": previous,
                            "active_problem_id": None,
                        }
                    )
            elif classification == ActionClass.COUNTED:
                scope_log.append(
                    {
                        "step": step,
                        "event": "attribution" if decision.allowed else "blocked",
                        "active_problem_id": scope.active_problem_id,
                        "attributed_problem_id": (
                            scope.active_problem_id if decision.allowed else None
                        ),
                        "reason": decision.reason,
                    }
                )
        budget_log.append(
            {
                "step": step,
                **asdict(decision),
                "action_class": decision.action_class.value,
                "classified_as": decision.classified_as.value,
                "active_problem_id": scope.active_problem_id,
            }
        )

    summary = {
        "schema_version": "1.0",
        "status": "dry_run_complete",
        "task_id": task_config.get("task_id"),
        "domain": task_config.get("domain"),
        "suite_id": task_config.get("suite_id"),
        "policy": budget_config.get("policy"),
        "budget_limit": budget.limit,
        "budget_used": budget.used,
        "budget_remaining": budget.remaining,
        "blocked_attempts": budget.blocked_attempts,
        "simulated_action_count": len(budget_log),
        "counted_actions_accepted": sum(
            row["budget_consumed"] == 1 for row in budget_log
        ),
        "free_actions_accepted": sum(
            row["allowed"] and not row["counted"] for row in budget_log
        ),
        "commands_executed_by_os": False,
        "model_called": False,
        "network_called": False,
        "docker_called": False,
        "harbor_called": False,
        "terminus_called": False,
        "correctness_feedback_exposed": False,
    }
    _write_json(output_dir / "budget_log.json", budget_log)
    _write_json(output_dir / "scope_log.json", scope_log)
    _write_json(output_dir / "dryrun_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_dryrun(Path(args.task_dir), Path(args.output_dir))
    except (AgenticDryRunError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"agentic dry-run failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
