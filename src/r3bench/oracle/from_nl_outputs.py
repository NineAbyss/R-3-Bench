"""Convert standardized Tool-Free generation/scoring outputs into Oracle inputs."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from r3bench.common.io import read_json, read_jsonl
from r3bench.common.schema import ProblemRecord
from r3bench.oracle.response_curve_schema import (
    ContestProblemResult,
    OracleSchemaError,
    ResponseCurvePoint,
)


def _index_unique(
    rows: Iterable[Mapping[str, Any]], field: str, kind: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise OracleSchemaError(f"{kind} row has no {field}")
        if value in result:
            raise OracleSchemaError(f"duplicate {kind} row for {value!r}")
        result[value] = row
    return result


def _reward(row: Mapping[str, Any], *, allow_unjudged: bool) -> int:
    parse_status = row.get("parse_status")
    judge_status = row.get("judge_status")
    if judge_status == "judge_error":
        return 0
    if judge_status != "judged":
        if parse_status in {"missing", "parse_error"} or allow_unjudged:
            return 0
        raise OracleSchemaError("postprocessing requires judged parsed outputs")
    return int(row.get("correct") is True or float(row.get("score", 0.0)) >= 1.0)


def _statuses(row: Mapping[str, Any], *, allow_unjudged: bool) -> tuple[str, str]:
    parse_status = row.get("parse_status")
    judge_status = row.get("judge_status")
    if parse_status not in {"parsed", "missing", "parse_error"}:
        raise OracleSchemaError("unsupported scoring parse_status")
    if judge_status not in {"judged", "not_judged", "judge_error"}:
        raise OracleSchemaError("unsupported scoring judge_status")
    if (
        judge_status == "not_judged"
        and parse_status == "parsed"
        and not allow_unjudged
    ):
        raise OracleSchemaError("postprocessing requires judged parsed outputs")
    return str(parse_status), str(judge_status)


def _attempt_costs(
    attempts: Iterable[Mapping[str, Any]],
    *,
    stage1_only: bool,
) -> tuple[dict[str, int], dict[str, str]]:
    costs: dict[str, int] = defaultdict(int)
    run_ids: dict[str, str] = {}
    for row in attempts:
        problem_id = row.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id:
            continue
        stage = row.get("stage")
        if stage1_only and stage != "stage1":
            continue
        usage = row.get("usage")
        if not isinstance(usage, Mapping):
            raise OracleSchemaError("attempt usage must be an object")
        output = usage.get("output_tokens", 0)
        reasoning = usage.get("reasoning_tokens", 0)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (output, reasoning)
        ):
            raise OracleSchemaError("attempt token usage must be non-negative integers")
        costs[problem_id] += output + reasoning
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise OracleSchemaError("attempt row has no run_id")
        if problem_id in run_ids and run_ids[problem_id] != run_id:
            raise OracleSchemaError("problem attempts contain inconsistent run IDs")
        run_ids[problem_id] = run_id
    return dict(costs), run_ids


def response_curve_points_from_outputs(
    *,
    run_dir: str | Path,
    scoring_dir: str | Path,
    problems: Iterable[ProblemRecord],
    domain: str,
    model_key: str,
    budget: int,
    stage1_only: bool,
    repeat_id: int | None = None,
    budget_level: int | None = None,
    allow_unjudged: bool = False,
) -> tuple[ResponseCurvePoint, ...]:
    run = Path(run_dir)
    attempts = read_jsonl(run / "attempts.jsonl")
    scores = _index_unique(
        read_jsonl(Path(scoring_dir) / "judge_results.jsonl"),
        "problem_id",
        "scoring",
    )
    costs, run_ids = _attempt_costs(attempts, stage1_only=stage1_only)
    rows: list[ResponseCurvePoint] = []
    for problem in problems:
        if problem.problem_id not in scores:
            raise OracleSchemaError(
                f"missing scored response-curve point for {problem.problem_id!r}"
            )
        score = scores[problem.problem_id]
        parse_status, judge_status = _statuses(score, allow_unjudged=allow_unjudged)
        if problem.problem_id not in run_ids:
            raise OracleSchemaError(
                f"missing generation attempt for {problem.problem_id!r}"
            )
        rows.append(
            ResponseCurvePoint(
                domain=domain,  # type: ignore[arg-type]
                model_key=model_key,
                setting="tool_free",
                budget_unit="output_tokens",
                mode="single_problem",
                problem_id=problem.problem_id,
                suite_id=problem.suite_id,
                problem_index=problem.problem_index,
                problem_label="ABCDEF"[problem.problem_index - 1],
                budget=budget,
                observed_cost=costs.get(problem.problem_id, 0),
                reward=_reward(score, allow_unjudged=allow_unjudged),
                parse_status=parse_status,  # type: ignore[arg-type]
                judge_status=judge_status,  # type: ignore[arg-type]
                source_run_id=_source_run_identity(
                    run,
                    run_ids[problem.problem_id],
                    repeat_id=repeat_id,
                    budget_level=budget_level,
                ),
                repeat_id=repeat_id,
                budget_level=budget_level,
            )
        )
    return tuple(rows)


def contest_results_from_outputs(
    *,
    run_dir: str | Path,
    scoring_dir: str | Path,
    domain: str,
    model_key: str,
    rho: float,
    contest_budget: int,
    repeat_id: int | None = None,
    allow_unjudged: bool = False,
) -> tuple[ContestProblemResult, ...]:
    run = Path(run_dir)
    parsed = _index_unique(
        read_jsonl(run / "parsed_answers.jsonl"),
        "problem_id",
        "parsed answer",
    )
    scores = _index_unique(
        read_jsonl(Path(scoring_dir) / "judge_results.jsonl"),
        "problem_id",
        "scoring",
    )
    if set(parsed) != set(scores):
        raise OracleSchemaError("parsed and scored contest problem IDs must match")
    rows: list[ContestProblemResult] = []
    for problem_id, parsed_row in parsed.items():
        score = scores[problem_id]
        parse_status, judge_status = _statuses(score, allow_unjudged=allow_unjudged)
        suite_id = parsed_row.get("suite_id")
        label = parsed_row.get("problem_label")
        request_id = parsed_row.get("run_id")
        if (
            not isinstance(suite_id, str)
            or not suite_id
            or label not in tuple("ABCDEF")
            or not isinstance(request_id, str)
            or not request_id
        ):
            raise OracleSchemaError("contest parsed row has invalid public identity")
        rows.append(
            ContestProblemResult(
                domain=domain,  # type: ignore[arg-type]
                model_key=model_key,
                setting="tool_free",
                budget_unit="output_tokens",
                mode="contest",
                problem_id=problem_id,
                suite_id=suite_id,
                problem_index="ABCDEF".index(str(label)) + 1,
                problem_label=str(label),
                rho=rho,
                formal_contest_budget=contest_budget,
                reward=_reward(score, allow_unjudged=allow_unjudged),
                parse_status=parse_status,  # type: ignore[arg-type]
                judge_status=judge_status,  # type: ignore[arg-type]
                source_run_id=_source_run_identity(
                    run,
                    request_id,
                    repeat_id=repeat_id,
                    budget_level=None,
                ),
                repeat_id=repeat_id,
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.suite_id, row.problem_index)))


def _source_run_identity(
    run_dir: Path,
    source_run_id: str,
    *,
    repeat_id: int | None,
    budget_level: int | None,
) -> str:
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        execution_id = summary.get("execution_id") if isinstance(summary, Mapping) else None
        if isinstance(execution_id, str) and execution_id:
            return execution_id
    suffix = ""
    if budget_level is not None:
        suffix += f":level_{budget_level}"
    if repeat_id is not None:
        suffix += f":repeat_{repeat_id}"
    return f"{source_run_id}{suffix}"


__all__ = [
    "contest_results_from_outputs",
    "response_curve_points_from_outputs",
]
