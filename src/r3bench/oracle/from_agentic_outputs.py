"""Convert scored Agentic outputs into response-curve and Oracle inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from r3bench.common.io import read_json, read_jsonl
from r3bench.common.schema import ProblemRecord
from r3bench.oracle.response_curve_schema import (
    ContestProblemResult,
    OracleSchemaError,
    ResponseCurvePoint,
)


def _object(path: Path, kind: str) -> Mapping[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise OracleSchemaError(f"{kind} must contain an object")
    return value


def _score_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in read_jsonl(path):
        problem_id = row.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id:
            raise OracleSchemaError("Agentic scoring row has no problem_id")
        if problem_id in result:
            raise OracleSchemaError(f"duplicate Agentic scoring row for {problem_id!r}")
        result[problem_id] = row
    return result


def _statuses(row: Mapping[str, Any], *, allow_unjudged: bool) -> tuple[str, str]:
    parse_status = row.get("parse_status")
    judge_status = row.get("judge_status")
    if parse_status not in {"parsed", "missing", "parse_error"}:
        raise OracleSchemaError("unsupported Agentic scoring parse_status")
    if judge_status not in {"judged", "not_judged", "judge_error"}:
        raise OracleSchemaError("unsupported Agentic scoring judge_status")
    if (
        judge_status == "not_judged"
        and parse_status == "parsed"
        and not allow_unjudged
    ):
        raise OracleSchemaError(
            "Agentic postprocessing requires judged parsed outputs"
        )
    return str(parse_status), str(judge_status)


def _reward(row: Mapping[str, Any], *, allow_unjudged: bool) -> int:
    parse_status = row.get("parse_status")
    judge_status = row.get("judge_status")
    if judge_status == "judge_error":
        return 0
    if judge_status != "judged":
        if parse_status in {"missing", "parse_error"} or allow_unjudged:
            return 0
        raise OracleSchemaError(
            "Agentic postprocessing requires judged parsed outputs"
        )
    return int(row.get("correct") is True or float(row.get("score") or 0.0) >= 1.0)


def _run_identity(
    run_dir: Path, *, repeat_id: int | None, budget_level: int | None
) -> str:
    summary = _object(run_dir / "backend_summary.json", "backend summary")
    execution_id = summary.get("execution_id")
    if isinstance(execution_id, str) and execution_id:
        return execution_id
    task_id = summary.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise OracleSchemaError("Agentic backend summary has no task_id")
    return _scoped_run_id(
        f"{task_id}:{run_dir.name}",
        repeat_id=repeat_id,
        budget_level=budget_level,
    )


def _action_usage(run_dir: Path, *, expected_budget: int | None) -> int:
    action = _object(run_dir / "public_action_log.json", "public action log")
    used = action.get("used")
    if isinstance(used, bool) or not isinstance(used, int) or used < 0:
        raise OracleSchemaError("Agentic action log has invalid used actions")
    logged_budget = action.get("budget")
    if expected_budget is None:
        if logged_budget is not None:
            raise OracleSchemaError(
                "unbounded Agentic baseline must record budget=null"
            )
    elif logged_budget != expected_budget:
        raise OracleSchemaError(
            "Agentic action-log budget does not match the response-curve point"
        )
    return used


def response_curve_point_from_agentic_outputs(
    *,
    run_dir: str | Path,
    scoring_dir: str | Path,
    problem: ProblemRecord,
    model_key: str,
    budget: int,
    repeat_id: int | None = None,
    budget_level: int | None = None,
    allow_unjudged: bool = False,
) -> ResponseCurvePoint:
    """Build one action-budget response-curve point from a scored episode."""

    run = Path(run_dir)
    scores = _score_rows(Path(scoring_dir) / "judge_results.jsonl")
    if set(scores) != {problem.problem_id}:
        raise OracleSchemaError(
            "single-problem Agentic scoring must contain exactly the bound problem"
        )
    score = scores[problem.problem_id]
    parse_status, judge_status = _statuses(score, allow_unjudged=allow_unjudged)
    observed = _action_usage(run, expected_budget=budget)
    if observed > budget:
        raise OracleSchemaError("observed Agentic cost exceeds configured budget")
    return ResponseCurvePoint(
        domain=problem.domain,
        model_key=model_key,
        setting="agentic",
        budget_unit="counted_actions",
        mode="single_problem",
        problem_id=problem.problem_id,
        suite_id=problem.suite_id,
        problem_index=problem.problem_index,
        problem_label="ABCDEF"[problem.problem_index - 1],
        budget=budget,
        observed_cost=observed,
        reward=_reward(score, allow_unjudged=allow_unjudged),
        parse_status=parse_status,  # type: ignore[arg-type]
        judge_status=judge_status,  # type: ignore[arg-type]
        source_run_id=_run_identity(
            run,
            repeat_id=repeat_id,
            budget_level=budget_level,
        ),
        repeat_id=repeat_id,
        budget_level=budget_level,
    )


def contest_results_from_agentic_outputs(
    *,
    run_dir: str | Path,
    scoring_dir: str | Path,
    problems: Iterable[ProblemRecord],
    model_key: str,
    rho: float,
    contest_budget: int,
    repeat_id: int | None = None,
    allow_unjudged: bool = False,
) -> tuple[ContestProblemResult, ...]:
    """Build six canonical contest rows from post-episode Agentic scoring."""

    run = Path(run_dir)
    problem_rows = tuple(problems)
    scores = _score_rows(Path(scoring_dir) / "judge_results.jsonl")
    expected = {problem.problem_id for problem in problem_rows}
    if len(problem_rows) != 6 or set(scores) != expected:
        raise OracleSchemaError(
            "Agentic contest conversion requires six matching scored problems"
        )
    _action_usage(run, expected_budget=contest_budget)
    source_run_id = _run_identity(
        run, repeat_id=repeat_id, budget_level=None
    )
    results: list[ContestProblemResult] = []
    for problem in sorted(problem_rows, key=lambda item: item.problem_index):
        score = scores[problem.problem_id]
        parse_status, judge_status = _statuses(score, allow_unjudged=allow_unjudged)
        results.append(
            ContestProblemResult(
                domain=problem.domain,
                model_key=model_key,
                setting="agentic",
                budget_unit="counted_actions",
                mode="contest",
                problem_id=problem.problem_id,
                suite_id=problem.suite_id,
                problem_index=problem.problem_index,
                problem_label="ABCDEF"[problem.problem_index - 1],
                rho=rho,
                formal_contest_budget=contest_budget,
                reward=_reward(score, allow_unjudged=allow_unjudged),
                parse_status=parse_status,  # type: ignore[arg-type]
                judge_status=judge_status,  # type: ignore[arg-type]
                source_run_id=source_run_id,
                repeat_id=repeat_id,
            )
        )
    return tuple(results)


def _scoped_run_id(
    source_run_id: str,
    *,
    repeat_id: int | None,
    budget_level: int | None,
) -> str:
    suffix = ""
    if budget_level is not None:
        suffix += f":level_{budget_level}"
    if repeat_id is not None:
        suffix += f":repeat_{repeat_id}"
    return f"{source_run_id}{suffix}"


__all__ = [
    "contest_results_from_agentic_outputs",
    "response_curve_point_from_agentic_outputs",
]
