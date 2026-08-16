"""Sanitized public Agentic task export built from the canonical public loader."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from r3bench.common.loader import load_contest_suites, load_single_problems
from r3bench.common.schema import ContestSuite, Domain, ProblemRecord
from r3bench.agentic.protocol_contract import paper_sandbox_limits
from r3bench.resource_paths import resource_path


_TEMPLATE = resource_path("prompts", "agentic", "contest_native_tool.txt")
_CODING_TEMPLATE = resource_path("prompts", "agentic", "coding_contest_task.txt")
_SINGLE_TEMPLATE = resource_path(
    "prompts", "agentic", "single_problem_response_curve.txt"
)
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")
_PRIVATE_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:home|mnt)/|/tmp/rbench(?:/|\b)")
_SECRET = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{12,}\b|\bhf_[A-Za-z0-9]{12,}\b|"
    r"\bolp_[A-Za-z0-9]{12,}\b|Bearer\s+[A-Za-z0-9._~+/-]{12,})"
)


class AgenticTaskExportError(ValueError):
    """Raised when a public Agentic task cannot be exported safely."""


ActionPolicyName = Literal["compute_tools"]
BudgetMode = Literal["counted_cap", "unbounded_calibration"]
_PAPER_RUNTIME_PROFILE = "harbor_terminus2_paper_v1"
_OFFLINE_RUNTIME_PROFILE = "offline_mock_replay_v1"


@dataclass(frozen=True, slots=True)
class ExportedAgenticTask:
    task_id: str
    suite_id: str
    domain: Domain
    task_dir: Path
    problem_count: int
    counted_action_budget: int | None
    budget_mode: BudgetMode
    execution_scope: str
    task_fingerprint_sha256: str
    budget_level: int | None = None
    repeat_id: int = 1


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )


def compute_task_fingerprint(task_dir: Path) -> str:
    """Hash the public task contract without creating a recursive hash field."""

    hasher = hashlib.sha256()
    for name in (
        "instruction.md",
        "task_config.json",
        "budget_config.json",
        "expected_artifacts.json",
        "public_problem_manifest.json",
    ):
        path = task_dir / name
        if name == "task_config.json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise AgenticTaskExportError("task_config.json must contain an object")
            value.pop("task_fingerprint_sha256", None)
            payload = json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        else:
            payload = path.read_bytes()
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(payload)
        hasher.update(b"\0")
    return hasher.hexdigest()


def _safe_id(value: str) -> str:
    cleaned = _SAFE_ID.sub("-", value).strip("-")
    if not cleaned:
        raise AgenticTaskExportError("suite_id cannot be converted into a safe task ID")
    return cleaned


def _validate_budget(
    value: int | None,
    budget_mode: BudgetMode,
) -> int | None:
    if budget_mode not in {"counted_cap", "unbounded_calibration"}:
        raise AgenticTaskExportError(f"unsupported budget mode: {budget_mode!r}")
    if budget_mode == "unbounded_calibration":
        if value is not None:
            raise AgenticTaskExportError(
                "unbounded calibration requires counted action budget=null"
            )
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AgenticTaskExportError(
            "counted action budget must be a non-negative integer"
        )
    return value


def _budget_display(budget: int | None, budget_mode: BudgetMode) -> str:
    return "unbounded calibration baseline" if budget is None else str(budget)


def _artifact_contract(
    domain: Domain, labels: str = "ABCDEF"
) -> tuple[list[dict[str, Any]], str, str]:
    if domain == "coding":
        artifacts = [
            {
                "problem_label": label,
                "container_path": f"/app/solution_{label}.cpp",
                "sandbox_relative_path": f"app/solution_{label}.cpp",
                "format": "C++17 source",
                "required": False,
            }
            for label in labels
        ]
        paths = ", ".join(item["container_path"] for item in artifacts)
        return (
            artifacts,
            paths,
            "Write one complete C++17 source file per attempted problem. "
            "Missing files are allowed and receive no credit.",
        )
    label_range = "Problem A" if labels == "A" else "Problem A through Problem F"
    if domain == "math":
        if labels == "A":
            answer_format = "Exactly one \\boxed{...} final answer."
            submission_format = (
                "Write exactly one final answer in `\\boxed{...}`. A missing, "
                "duplicate, or malformed box receives no credit."
            )
        else:
            answer_format = (
                "One `Problem X` section per attempted problem, containing "
                "exactly one \\boxed{...} answer."
            )
            submission_format = (
                f"Create one clearly labeled `Problem X` section for each "
                f"attempted item in {label_range}. Put exactly one final answer "
                "in `\\boxed{...}` in that section. A missing, duplicate, or "
                "malformed box receives no credit; omitted sections are allowed."
            )
        answer_contract = (
            "boxed_v1" if labels == "A" else "problem_sections_boxed_v1"
        )
    else:
        if labels == "A":
            answer_format = "Exactly one <answer>...</answer> final answer."
            submission_format = (
                "Write exactly one final answer in the ordinary "
                "`<answer>...</answer>` tag. Labeled tags such as `<answer A>` "
                "are invalid."
            )
        else:
            answer_format = (
                "One `Problem X` section per attempted problem, containing "
                "exactly one <answer>...</answer> answer."
            )
            submission_format = (
                f"Create one clearly labeled `Problem X` section for each "
                f"attempted item in {label_range}. Put exactly one final answer "
                "in the ordinary `<answer>...</answer>` tag in that section. "
                "Labeled tags such as `<answer A>` are invalid; omitted sections "
                "are allowed."
            )
        answer_contract = (
            "answer_tag_v1"
            if labels == "A"
            else "problem_sections_answer_tag_v1"
        )
    return (
        [
            {
                "container_path": "/logs/artifacts/answer.txt",
                "sandbox_relative_path": "logs/artifacts/answer.txt",
                "format": answer_format,
                "answer_contract": answer_contract,
                "required": False,
            }
        ],
        "/logs/artifacts/answer.txt",
        submission_format,
    )


def _problem_text(suite: ContestSuite) -> str:
    blocks = []
    for problem in suite.problems:
        if suite.domain == "coding":
            payload = problem.domain_payload
            title = str(payload.get("title") or problem.problem_id)
            source_url = payload.get("source_url")
            time_limit = payload.get("time_limit_ms")
            memory_limit = payload.get("memory_limit_mb")
            metadata = [
                f"### Problem {problem.problem_label}: {title}",
                (
                    f"Problem link: {source_url}"
                    if isinstance(source_url, str) and source_url
                    else f"Problem ID: {problem.problem_id}"
                ),
            ]
            if isinstance(time_limit, int):
                metadata.append(f"Time limit: {time_limit} ms")
            if isinstance(memory_limit, int):
                metadata.append(f"Memory limit: {memory_limit} MB")
            metadata.extend(("#### Statement", problem.problem_statement))
            blocks.append("\n\n".join(metadata))
        else:
            blocks.append(
                f"## Problem {problem.problem_label}\n"
                f"Problem ID: {problem.problem_id}\n\n"
                f"{problem.problem_statement}"
            )
    return "\n\n".join(blocks)


def _execution_scope(suite: ContestSuite) -> str:
    """Use public provenance, not directory naming, to identify synthetic data."""

    def is_synthetic(problem: Any) -> bool:
        upstream = problem.domain_payload.get("upstream_dataset")
        if upstream is None:
            upstream = problem.metadata_public.get("upstream_dataset")
        return (
            upstream == "synthetic/r3bench-examples"
            or problem.source == "synthetic"
        )

    if all(is_synthetic(problem) for problem in suite.problems):
        return "synthetic_toy"
    return "public_benchmark"


def _problem_execution_scope(problem: ProblemRecord) -> str:
    upstream = problem.domain_payload.get("upstream_dataset")
    if upstream is None:
        upstream = problem.metadata_public.get("upstream_dataset")
    if upstream == "synthetic/r3bench-examples" or problem.source == "synthetic":
        return "synthetic_toy"
    return "public_benchmark"


def _single_problem_text(problem: ProblemRecord) -> str:
    if problem.domain == "coding":
        payload = problem.domain_payload
        title = str(payload.get("title") or problem.problem_id)
        source_url = payload.get("source_url")
        metadata = [
            f"# Problem A: {title}",
            (
                f"Problem link: {source_url}"
                if isinstance(source_url, str) and source_url
                else f"Problem ID: {problem.problem_id}"
            ),
        ]
        time_limit = payload.get("time_limit_ms")
        memory_limit = payload.get("memory_limit_mb")
        if isinstance(time_limit, int):
            metadata.append(f"Time limit: {time_limit} ms")
        if isinstance(memory_limit, int):
            metadata.append(f"Memory limit: {memory_limit} MB")
        metadata.extend(("## Statement", problem.problem_statement))
        return "\n\n".join(metadata)
    return (
        f"# Problem A\nProblem ID: {problem.problem_id}\n\n"
        f"{problem.problem_statement}"
    )


def _render_single_instruction(
    problem: ProblemRecord,
    budget: int,
    artifact_paths: str,
    submission_format: str,
    action_policy: ActionPolicyName,
) -> str:
    template = _SINGLE_TEMPLATE.read_text(encoding="utf-8")
    accounting_rule = (
        "Only explicitly recognized bookkeeping, status, environment, task "
        "completion, pure file-write, and direct final-artifact-write actions "
        "are free. Every other terminal command consumes a counted action, "
        "including unrecognized commands."
    )
    values = {
        "{{DOMAIN}}": problem.domain,
        "{{PROBLEM_ID}}": problem.problem_id,
        "{{COUNTED_ACTION_BUDGET}}": str(budget),
        "{{ACTION_POLICY}}": action_policy,
        "{{FINAL_ARTIFACT_PATHS}}": artifact_paths,
        "{{ACTION_ACCOUNTING_RULE}}": accounting_rule,
        "{{SUBMISSION_FORMAT}}": submission_format,
        "{{PROBLEM}}": _single_problem_text(problem),
    }
    rendered = template
    for placeholder, value in values.items():
        rendered = rendered.replace(placeholder, value)
    unresolved = re.findall(r"\{\{[A-Z0-9_]+\}\}", rendered)
    if unresolved:
        raise AgenticTaskExportError(
            f"unresolved single-problem placeholders: {unresolved}"
        )
    if _PRIVATE_PATH.search(rendered) or _SECRET.search(rendered):
        raise AgenticTaskExportError(
            "rendered single-problem instruction contains unsafe content"
        )
    return rendered


def _render_instruction(
    suite: ContestSuite,
    budget: int | None,
    budget_mode: BudgetMode,
    artifact_paths: str,
    submission_format: str,
    action_policy: ActionPolicyName,
) -> str:
    template_path = _CODING_TEMPLATE if suite.domain == "coding" else _TEMPLATE
    template = template_path.read_text(encoding="utf-8")
    accounting_rule = (
        "Only explicitly recognized bookkeeping, status, environment, task "
        "completion, pure file-write, and direct final-artifact-write actions "
        "are free. Every other terminal command consumes a counted action, "
        "including unrecognized commands."
    )
    values = {
        "{{DOMAIN}}": suite.domain,
        "{{SUITE_ID}}": suite.suite_id,
        "{{COUNTED_ACTION_BUDGET}}": _budget_display(budget, budget_mode),
        "{{ACTION_POLICY}}": action_policy,
        "{{FINAL_ARTIFACT_PATHS}}": artifact_paths,
        "{{ACTION_ACCOUNTING_RULE}}": accounting_rule,
        "{{FINALIZATION_RULE}}": (
            "Pure writes to designated final artifacts are free and remain "
            "available after paid-compute exhaustion."
        ),
        "{{SUBMISSION_FORMAT}}": submission_format,
        "{{PROBLEMS}}": _problem_text(suite),
    }
    rendered = template
    for placeholder, value in values.items():
        rendered = rendered.replace(placeholder, value)
    unresolved = re.findall(r"\{\{[A-Z0-9_]+\}\}", rendered)
    if unresolved:
        raise AgenticTaskExportError(f"unresolved template placeholders: {unresolved}")
    if _PRIVATE_PATH.search(rendered) or _SECRET.search(rendered):
        raise AgenticTaskExportError("rendered instruction contains unsafe content")
    return rendered


def _export_suite(
    suite: ContestSuite,
    *,
    output_dir: Path,
    budget: int | None,
    budget_mode: BudgetMode,
    action_policy: ActionPolicyName,
) -> ExportedAgenticTask:
    task_id = f"r3bench-agentic-{suite.domain}-{_safe_id(suite.suite_id)}"
    task_dir = output_dir / _safe_id(suite.suite_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    artifacts, artifact_paths, submission_format = _artifact_contract(suite.domain)
    instruction = _render_instruction(
        suite,
        budget,
        budget_mode,
        artifact_paths,
        submission_format,
        action_policy,
    )
    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")

    labels = {
        problem.problem_label: problem.problem_id for problem in suite.problems
    }
    execution_scope = _execution_scope(suite)
    task_config = {
        "schema_version": "2.2",
        "task_id": task_id,
        "domain": suite.domain,
        "split": suite.split,
        "suite_id": suite.suite_id,
        "mode": "contest",
        "visibility": "hidden",
        "runtime": _PAPER_RUNTIME_PROFILE,
        "offline_test_runtime": _OFFLINE_RUNTIME_PROFILE,
        "paper_equivalent_runtime_required": True,
        "sandbox_limits": paper_sandbox_limits(suite.domain),
        "action_policy": action_policy,
        "budget_mode": budget_mode,
        "problem_labels": labels,
        "correctness_feedback": False,
        "external_runtime_required_for_real_execution": True,
        "execution_scope": execution_scope,
        "backend_smoke_only": execution_scope == "synthetic_toy",
        "external_backend_handoff_required": True,
    }
    _write_json(
        task_dir / "task_config.json",
        task_config,
    )
    _write_json(
        task_dir / "budget_config.json",
        {
            "schema_version": "2.0",
            "resource_type": "counted_actions",
            "policy": action_policy,
            "budget_mode": budget_mode,
            "counted_action_budget": budget,
            "counted_action_unit_cost": 1,
            "blocked_attempts_consume_executed_budget": False,
            "free_categories": [
                "bookkeeping",
                "status",
                "environment",
                "pure_file_write",
                "final_artifact_write",
                "task_completion",
            ],
            "final_artifact_write_counted": False,
        },
    )
    _write_json(
        task_dir / "expected_artifacts.json",
        {
            "schema_version": "2.0",
            "domain": suite.domain,
            "artifacts": artifacts,
            "grade_after_episode": True,
            "live_correctness_feedback": False,
        },
    )
    _write_json(
        task_dir / "public_problem_manifest.json",
        {
            "schema_version": "2.0",
            "domain": suite.domain,
            "suite_id": suite.suite_id,
            "problems": [
                {
                    "problem_id": problem.problem_id,
                    "problem_label": problem.problem_label,
                    "position": problem.problem_index,
                    "problem_statement": problem.problem_statement,
                }
                for problem in suite.problems
            ],
            "difficulty_labels_exposed": False,
        },
    )
    task_config["task_fingerprint_sha256"] = compute_task_fingerprint(task_dir)
    _write_json(task_dir / "task_config.json", task_config)
    return ExportedAgenticTask(
        task_id=task_id,
        suite_id=suite.suite_id,
        domain=suite.domain,
        task_dir=task_dir,
        problem_count=len(suite.problems),
        counted_action_budget=budget,
        budget_mode=budget_mode,
        execution_scope=execution_scope,
        task_fingerprint_sha256=task_config["task_fingerprint_sha256"],
    )


def _export_single_problem(
    problem: ProblemRecord,
    *,
    output_dir: Path,
    budget: int,
    budget_level: int,
    repeat_id: int,
    action_policy: ActionPolicyName,
) -> ExportedAgenticTask:
    safe_problem = _safe_id(problem.problem_id)
    suite_id = (
        f"response_curve_{safe_problem}_level_{budget_level}_budget_{budget}"
        f"_repeat_{repeat_id}"
    )
    task_id = f"r3bench-agentic-{problem.domain}-{suite_id}"
    task_dir = output_dir / suite_id
    task_dir.mkdir(parents=True, exist_ok=True)
    artifacts, artifact_paths, submission_format = _artifact_contract(
        problem.domain, "A"
    )
    instruction = _render_single_instruction(
        problem, budget, artifact_paths, submission_format, action_policy
    )
    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")
    execution_scope = _problem_execution_scope(problem)
    task_config = {
        "schema_version": "2.2",
        "task_id": task_id,
        "domain": problem.domain,
        "split": problem.split,
        "suite_id": suite_id,
        "mode": "single_problem_response_curve",
        "visibility": "hidden",
        "runtime": _PAPER_RUNTIME_PROFILE,
        "offline_test_runtime": _OFFLINE_RUNTIME_PROFILE,
        "paper_equivalent_runtime_required": True,
        "sandbox_limits": paper_sandbox_limits(problem.domain),
        "action_policy": action_policy,
        "budget_level": budget_level,
        "repeat_id": repeat_id,
        "problem_labels": {"A": problem.problem_id},
        "correctness_feedback": False,
        "external_runtime_required_for_real_execution": True,
        "execution_scope": execution_scope,
        "backend_smoke_only": execution_scope == "synthetic_toy",
        "external_backend_handoff_required": True,
    }
    _write_json(task_dir / "task_config.json", task_config)
    _write_json(
        task_dir / "budget_config.json",
        {
            "schema_version": "2.0",
            "resource_type": "counted_actions",
            "policy": action_policy,
            "budget_level": budget_level,
            "repeat_id": repeat_id,
            "counted_action_budget": budget,
            "counted_action_unit_cost": 1,
            "blocked_attempts_consume_executed_budget": False,
            "free_categories": [
                "bookkeeping",
                "status",
                "environment",
                "pure_file_write",
                "final_artifact_write",
                "task_completion",
            ],
            "final_artifact_write_counted": False,
        },
    )
    _write_json(
        task_dir / "expected_artifacts.json",
        {
            "schema_version": "2.0",
            "domain": problem.domain,
            "artifacts": artifacts,
            "grade_after_episode": True,
            "live_correctness_feedback": False,
        },
    )
    _write_json(
        task_dir / "public_problem_manifest.json",
        {
            "schema_version": "2.0",
            "domain": problem.domain,
            "suite_id": suite_id,
            "problems": [
                {
                    "problem_id": problem.problem_id,
                    "problem_label": "A",
                    "position": 1,
                    "problem_statement": problem.problem_statement,
                }
            ],
            "difficulty_labels_exposed": False,
        },
    )
    task_config["task_fingerprint_sha256"] = compute_task_fingerprint(task_dir)
    _write_json(task_dir / "task_config.json", task_config)
    return ExportedAgenticTask(
        task_id=task_id,
        suite_id=suite_id,
        domain=problem.domain,
        task_dir=task_dir,
        problem_count=1,
        counted_action_budget=budget,
        budget_mode="counted_cap",
        execution_scope=execution_scope,
        task_fingerprint_sha256=task_config["task_fingerprint_sha256"],
        budget_level=budget_level,
        repeat_id=repeat_id,
    )


def export_agentic_tasks(
    *,
    domain: Domain,
    data_source: str | Path,
    output_dir: str | Path,
    budget: int | None,
    budget_mode: BudgetMode = "counted_cap",
    split: str = "test",
    limit_suites: int | None = 1,
    suite_offset: int = 0,
    confirm_full_export: bool = False,
    strict_data: bool = False,
    action_policy: ActionPolicyName | None = None,
) -> tuple[ExportedAgenticTask, ...]:
    """Export deterministic task directories without invoking an agent runtime."""

    checked_budget = _validate_budget(budget, budget_mode)
    selected_policy: ActionPolicyName = action_policy or "compute_tools"
    if selected_policy != "compute_tools":
        raise AgenticTaskExportError(
            "formal Agentic exports require action_policy=compute_tools for every domain"
        )
    if limit_suites is not None and limit_suites <= 0:
        raise AgenticTaskExportError("limit_suites must be positive or omitted")
    if isinstance(suite_offset, bool) or suite_offset < 0:
        raise AgenticTaskExportError("suite_offset must be a non-negative integer")
    if (limit_suites is None or limit_suites > 1) and not confirm_full_export:
        raise AgenticTaskExportError(
            "multi-suite/full export requires --confirm-full-export"
        )
    suites = load_contest_suites(
        domain, split, data_source, strict=strict_data
    )
    if suite_offset >= len(suites):
        raise AgenticTaskExportError("suite_offset is beyond the available suites")
    selected: Iterable[ContestSuite] = (
        suites[suite_offset:]
        if limit_suites is None
        else suites[suite_offset : suite_offset + limit_suites]
    )
    target = Path(output_dir)
    exported = tuple(
        _export_suite(
            suite,
            output_dir=target,
            budget=checked_budget,
            budget_mode=budget_mode,
            action_policy=selected_policy,
        )
        for suite in selected
    )
    if not exported:
        raise AgenticTaskExportError("no contest suites were selected")
    return exported


def export_agentic_response_curve_tasks(
    *,
    domain: Domain,
    data_source: str | Path,
    output_dir: str | Path,
    budgets: Iterable[int],
    repeat_ids: Iterable[int] = range(1, 6),
    split: str = "test",
    problem_ids: Iterable[str] | None = None,
    limit_problems: int | None = None,
    confirm_full_curve: bool = False,
    strict_data: bool = False,
    action_policy: ActionPolicyName | None = None,
) -> tuple[ExportedAgenticTask, ...]:
    """Export ordered budget levels, with five repeats by default."""

    selected_policy: ActionPolicyName = action_policy or "compute_tools"
    if selected_policy != "compute_tools":
        raise AgenticTaskExportError(
            "formal Agentic response curves require action_policy=compute_tools "
            "for every domain"
        )
    checked_budgets = tuple(
        _validate_budget(value, "counted_cap") for value in budgets
    )
    if not checked_budgets:
        raise AgenticTaskExportError("response curve requires at least one budget")
    if len(checked_budgets) > 6:
        raise AgenticTaskExportError(
            "response curve supports at most six ordered budget levels"
        )
    checked_repeat_ids = tuple(repeat_ids)
    if not checked_repeat_ids:
        raise AgenticTaskExportError("response curve requires at least one repeat ID")
    if len(set(checked_repeat_ids)) != len(checked_repeat_ids) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in checked_repeat_ids
    ):
        raise AgenticTaskExportError(
            "response-curve repeat IDs must be unique positive integers"
        )
    problems = load_single_problems(
        domain, split, data_source, strict=strict_data
    )
    requested_ids = tuple(problem_ids or ())
    if requested_ids:
        by_id = {problem.problem_id: problem for problem in problems}
        missing = [problem_id for problem_id in requested_ids if problem_id not in by_id]
        if missing:
            raise AgenticTaskExportError(
                "response-curve problem IDs are absent from canonical data"
            )
        selected_problems = tuple(by_id[problem_id] for problem_id in requested_ids)
    else:
        selected_problems = problems
    if limit_problems is not None:
        if limit_problems <= 0:
            raise AgenticTaskExportError("limit_problems must be positive")
        selected_problems = selected_problems[:limit_problems]
    episode_count = (
        len(selected_problems) * len(checked_budgets) * len(checked_repeat_ids)
    )
    if episode_count > 6 and not confirm_full_curve:
        raise AgenticTaskExportError(
            "more than six response-curve episodes require "
            "--confirm-full-curve"
        )
    target = Path(output_dir)
    return tuple(
        _export_single_problem(
            problem,
            output_dir=target,
            budget=budget,
            budget_level=budget_level,
            repeat_id=repeat_id,
            action_policy=selected_policy,
        )
        for problem in selected_problems
        for budget_level, budget in enumerate(checked_budgets, start=1)
        for repeat_id in checked_repeat_ids
    )


__all__ = [
    "AgenticTaskExportError",
    "ActionPolicyName",
    "BudgetMode",
    "ExportedAgenticTask",
    "compute_task_fingerprint",
    "export_agentic_response_curve_tasks",
    "export_agentic_tasks",
]
