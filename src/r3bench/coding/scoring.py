"""Saved-output Coding scoring with runtime-only verifier configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from r3bench.coding.parser import extract_cpp_code
from r3bench.coding.verifier import (
    CodingVerifier,
    CodingVerifierResult,
    LightCPVerifierConfig,
    LightCPVerifierExecutor,
    MockCodingVerifier,
    load_lightcpverifier_config,
    validate_lightcpverifier_config,
    verify_saved_solution,
)
from r3bench.common.io import read_jsonl_snapshot
from r3bench.common.loader import load_single_problems
from r3bench.common.schema import ProblemRecord
from r3bench.common.scorer_registry import (
    ScorerProfile,
    scorer_profile_contract,
    scorer_profile_contract_sha256,
)


class CodingScoringError(ValueError):
    """Raised when saved Coding outputs violate the scoring contract."""


_PROVENANCE_FIELDS = (
    "run_id",
    "request_id",
    "stage",
    "stage1_request_id",
    "stage2_request_id",
    "source_setting",
    "execution_id",
    "task_id",
    "model_key",
    "repeat_id",
)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _effective_strict(data_source: str | Path, strict: bool) -> bool:
    parts = Path(data_source).parts
    return strict and not (
        len(parts) >= 2
        and parts[-2:] == ("data", "coding.jsonl")
        and "examples" in parts
    )


def _parsed_solution(row: Mapping[str, Any], line_number: int) -> str | None:
    if "parsed_answer" in row:
        value = row["parsed_answer"]
        if value is None:
            return None
        if not isinstance(value, str):
            raise CodingScoringError(
                f"prediction row {line_number} has non-string parsed_answer"
            )
        return extract_cpp_code(value)
    for field in ("prediction", "response_text"):
        if field in row:
            value = row[field]
            if not isinstance(value, str):
                raise CodingScoringError(
                    f"prediction row {line_number} has non-string {field}"
                )
            return extract_cpp_code(value)
    raise CodingScoringError(
        f"prediction row {line_number} requires parsed_answer, prediction, or response_text"
    )


def _prepare(
    rows: list[dict[str, Any]],
    problems: Mapping[str, ProblemRecord],
    *,
    max_problems: int | None,
) -> list[tuple[ProblemRecord, str | None, dict[str, Any]]]:
    if max_problems is not None and max_problems <= 0:
        raise CodingScoringError("max_problems must be positive")
    selected = rows if max_problems is None else rows[:max_problems]
    seen: set[str] = set()
    prepared: list[tuple[ProblemRecord, str | None, dict[str, Any]]] = []
    for line_number, row in enumerate(selected, start=1):
        domain = row.get("domain")
        if domain not in {None, "coding"}:
            raise CodingScoringError(
                f"prediction row {line_number} declares a non-Coding domain"
            )
        problem_id = row.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise CodingScoringError(
                f"prediction row {line_number} has no non-empty problem_id"
            )
        if problem_id in seen:
            raise CodingScoringError(f"duplicate prediction for {problem_id!r}")
        seen.add(problem_id)
        problem = problems.get(problem_id)
        if problem is None:
            raise CodingScoringError(f"unknown problem_id {problem_id!r}")
        provenance: dict[str, Any] = {}
        for field in _PROVENANCE_FIELDS:
            if field not in row:
                continue
            value = row[field]
            invalid = (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                if field == "repeat_id"
                else value is not None and not isinstance(value, str)
            )
            if invalid:
                raise CodingScoringError(
                    f"prediction row {line_number} has invalid {field}"
                )
            provenance[field] = value
        prepared.append((problem, _parsed_solution(row, line_number), provenance))
    return prepared


def _result_row(
    problem: ProblemRecord,
    result: CodingVerifierResult,
    *,
    parse_status: str,
    mode: str,
    evaluator: str,
) -> dict[str, Any]:
    operational = {
        "not_configured",
        "service_unreachable",
        "assets_unavailable",
        "invalid_config",
        "verifier_error",
    }
    status = str(result.status)
    if mode == "dry-run":
        judge_status = "not_judged"
        correct: bool | None = None
        score: float | None = None
    elif status in operational:
        judge_status = "judge_error"
        correct = False
        score = 0.0
    elif status == "missing_solution" or parse_status != "parsed":
        judge_status = "not_judged"
        correct = False
        score = 0.0
    else:
        judge_status = "judged"
        correct = result.accepted
        score = 1.0 if result.accepted else 0.0
    return {
        "domain": "coding",
        "problem_id": problem.problem_id,
        "suite_id": problem.suite_id,
        "problem_index": problem.problem_index,
        "problem_label": "ABCDEF"[problem.problem_index - 1],
        "upstream_id": result.upstream_id,
        "parse_status": parse_status,
        "parsed_output": None,
        "judge_status": judge_status,
        "correct": correct,
        "score": score,
        "verdict": result.verdict,
        "detail": result.detail,
        "evaluator": evaluator,
        "scoring_mode": mode,
        "verifier_status": status,
        "accepted": result.accepted,
        "diagnostic_status": judge_status,
        "error_type": status if status in operational else None,
    }


def score_coding_saved_outputs(
    *,
    predictions_path: str | Path,
    data_source: str | Path,
    output_dir: str | Path,
    mode: str,
    verifier_config_path: str | Path | None = None,
    split: str = "test",
    strict: bool = True,
    max_problems: int | None = None,
    mock_verifier: CodingVerifier | None = None,
    production_executor: LightCPVerifierExecutor | None = None,
    scorer_profile: ScorerProfile | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score previously generated Coding outputs without invoking generation."""

    if mode not in {"mock", "production", "dry-run"}:
        raise CodingScoringError("mode must be mock, production, or dry-run")
    config: LightCPVerifierConfig | None = None
    if verifier_config_path is not None:
        config = load_lightcpverifier_config(verifier_config_path)
    if mode == "production":
        if config is None:
            raise CodingScoringError(
                "production Coding scoring requires --verifier-config"
            )
        validate_lightcpverifier_config(config, production=True)
        if (
            scorer_profile is None
            or scorer_profile.domain != "coding"
            or scorer_profile.scorer_type != "external_verifier"
            or scorer_profile.config.get("verifier") != "LightCPVerifier"
            or scorer_profile.config.get("binding") != "lightcp_http_v1"
            or config.mode != "service"
        ):
            raise CodingScoringError(
                "production Coding scoring differs from the released scorer contract"
            )

    problems = load_single_problems(
        "coding",
        split,
        data_source,
        strict=_effective_strict(data_source, strict),
    )
    input_bytes, prediction_rows = read_jsonl_snapshot(predictions_path)
    prepared = _prepare(
        prediction_rows,
        {problem.problem_id: problem for problem in problems},
        max_problems=max_problems,
    )

    verifier = mock_verifier or MockCodingVerifier()
    evaluator = (
        scorer_profile.profile_id
        if scorer_profile is not None
        else ("mock" if mode == "mock" else "LightCPVerifier")
    )
    rows: list[dict[str, Any]] = []
    verifier_calls = 0
    for problem, parsed_answer, provenance in prepared:
        parse_status = "parsed" if parsed_answer is not None else "parse_error"
        if mode == "dry-run":
            upstream_id = problem.domain_payload.get("upstream_id")
            if not isinstance(upstream_id, str) or not upstream_id:
                raise CodingScoringError(
                    f"problem {problem.problem_id!r} has no upstream_id"
                )
            result = CodingVerifierResult(
                upstream_id=upstream_id,
                accepted=False,
                verdict=(
                    "not_configured"
                    if config is None or config.status != "configured"
                    else "verifier_error"
                ),
                detail="configuration and data contract validated; verifier not called",
                status=(
                    "not_configured"
                    if config is None or config.status != "configured"
                    else "verifier_error"
                ),
            )
        elif mode == "mock":
            if parsed_answer is None:
                result = verify_saved_solution(problem, None, None)
            else:
                verifier_calls += 1
                result = verifier.verify(problem, parsed_answer)
        else:
            if parsed_answer is not None:
                verifier_calls += 1
            result = verify_saved_solution(
                problem,
                parsed_answer,
                config,
                executor=production_executor,
            )
        rows.append(
            {
                **_result_row(
                    problem,
                    result,
                    parse_status=parse_status,
                    mode=mode,
                    evaluator=evaluator,
                ),
                **provenance,
            }
        )

    target = Path(output_dir)
    results_sha256 = _write_jsonl(target / "judge_results.jsonl", rows)
    accepted = sum(row["accepted"] is True for row in rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["verifier_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "schema_version": "1.0",
        "domain": "coding",
        "mode": mode,
        "scoring_mode": mode,
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "results_sha256": results_sha256,
        "input_count": len(rows),
        "parsed_count": sum(row["parse_status"] == "parsed" for row in rows),
        "accepted_count": accepted,
        "verifier_status_counts": status_counts,
        "verifier_call_count": verifier_calls,
        "generation_provider_called": False,
        "external_verifier_started": False,
        "docker_started": False,
        "runtime_paths_serialized": False,
        "status": "dry_run" if mode == "dry-run" else "complete",
        "scorer_profile": scorer_profile.profile_id if scorer_profile else "mock",
        "scorer_contract": (
            scorer_profile_contract(scorer_profile) if scorer_profile else None
        ),
        "scorer_contract_sha256": (
            scorer_profile_contract_sha256(scorer_profile) if scorer_profile else None
        ),
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "scoring_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Coding Saved-Output Scoring",
        "",
        f"- Mode: `{mode}`",
        f"- Inputs: {summary['input_count']}",
        f"- Parsed: {summary['parsed_count']}",
        f"- Accepted: {summary['accepted_count']}",
        f"- Verifier calls: {summary['verifier_call_count']}",
        f"- Status counts: `{json.dumps(status_counts, sort_keys=True)}`",
        "",
        "Generation was not invoked. Hidden assets, verifier locations, and",
        "service endpoints are not serialized in these outputs.",
        "",
    ]
    (target / "scoring_report.md").write_text("\n".join(report), encoding="utf-8")
    return rows, summary


__all__ = ["CodingScoringError", "score_coding_saved_outputs"]
