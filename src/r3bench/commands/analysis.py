#!/usr/bin/env python3
"""Build public analysis inputs and compute equal/Oracle/gap results."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from r3bench.agentic.external_backend import (
    ExternalAgenticBackendError,
    resolve_agentic_execution_profile,
    validate_external_backend_handoff,
)
from r3bench.agentic.scoring_handoff import (
    collect_agentic_saved_outputs,
)
from r3bench.common.budget import (
    BudgetResolutionError,
    OfficialBudgetProfile,
    load_official_budget_profiles,
    resolve_official_budget_profile,
)
from r3bench.common.io import read_json, read_jsonl, read_jsonl_snapshot
from r3bench.common.loader import load_contest_suites, load_single_problems
from r3bench.common.profile_registry import ModelProfile, load_model_profiles
from r3bench.common.result_schema import to_public_dict
from r3bench.common.scorer_registry import (
    ScorerProfile,
    load_scorer_profiles,
    scorer_profile_contract,
    scorer_profile_contract_sha256,
)
from r3bench.common.two_stage_profile import (
    TwoStageProfile,
    load_two_stage_profiles,
)
from r3bench.oracle.from_agentic_outputs import (
    contest_results_from_agentic_outputs,
    response_curve_point_from_agentic_outputs,
)
from r3bench.oracle.from_nl_outputs import (
    contest_results_from_outputs,
    response_curve_points_from_outputs,
)
from r3bench.oracle.protocol_v3 import (
    ConditionBudget,
    run_condition_analysis,
    write_condition_budget_document,
)
from r3bench.oracle.response_curve_schema import OracleSchemaError
from r3bench.resource_paths import resolve_path, resource_path


def _grid(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("grid must contain integers") from exc
    if not parsed or any(item < 0 for item in parsed):
        raise argparse.ArgumentTypeError("grid must contain non-negative integers")
    if tuple(sorted(parsed)) != parsed:
        raise argparse.ArgumentTypeError("grid must be nondecreasing")
    return parsed


def _safe_condition(
    args: argparse.Namespace, *, expected_role: str
) -> tuple[str | None, float | None, OfficialBudgetProfile | None]:
    profile = args.budget_profile
    rho = args.rho
    if args.condition_kind == "official_profile" and not profile:
        raise OracleSchemaError("official_profile requires --budget-profile")
    if args.condition_kind == "custom" and (profile is not None or rho is not None):
        raise OracleSchemaError("custom conditions cannot set profile or rho")
    if args.condition_kind == "custom":
        return None, None, None
    try:
        resolved = resolve_official_budget_profile(
            profile,
            setting=args.setting,
            domain=args.domain,
            model_key=args.model,
        )
    except BudgetResolutionError as exc:
        raise OracleSchemaError(str(exc)) from exc
    if resolved.role != expected_role or rho != resolved.rho:
        raise OracleSchemaError(
            "official budget profile role or rho does not match the analysis input"
        )
    return profile, rho, resolved


def _official_curve_grid(profile: OfficialBudgetProfile) -> tuple[int, ...]:
    matches = [
        candidate
        for candidate in load_official_budget_profiles().values()
        if candidate.model_key == profile.model_key
        and candidate.domain == profile.domain
        and candidate.setting == profile.setting
        and candidate.role == "single_problem_response_curve"
    ]
    if len(matches) != 1:
        raise OracleSchemaError(
            "official response-curve profile is missing or ambiguous"
        )
    return matches[0].budget_grid


def _selected_problems(
    domain: str,
    data: str,
    *,
    problem_ids: set[str] | None,
    strict: bool,
    limit: int | None = None,
):
    problems = load_single_problems(domain, "test", resolve_path(data), strict=strict)
    if problem_ids is not None:
        selected = tuple(
            problem for problem in problems if problem.problem_id in problem_ids
        )
        if {problem.problem_id for problem in selected} != problem_ids:
            raise OracleSchemaError("scoring output contains unknown problem IDs")
        return selected
    return tuple(problems[:limit]) if limit is not None else tuple(problems)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], append: bool) -> int:
    serialized = [
        json.dumps(dict(row), ensure_ascii=False, allow_nan=False, sort_keys=True)
        + "\n"
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        handle.writelines(serialized)
    return len(serialized)


def _execution_metadata(
    run_dir: str | Path,
    *,
    setting: str,
    repeat_id: int | None,
    budget_level: int | None = None,
    require_repeat: bool = False,
    require_budget_level: bool = False,
) -> tuple[int | None, int | None]:
    filename = "run_summary.json" if setting == "tool_free" else "backend_summary.json"
    summary_path = Path(run_dir) / filename
    if not summary_path.is_file():
        if require_repeat or require_budget_level:
            raise OracleSchemaError(
                f"official analysis requires recorded metadata in {filename}"
            )
        return repeat_id, budget_level
    summary = read_json(summary_path)
    if not isinstance(summary, Mapping):
        raise OracleSchemaError(f"{filename} must contain an object")
    recorded_repeat = summary.get("repeat_id")
    recorded_level = summary.get("budget_level")
    if repeat_id is not None and recorded_repeat != repeat_id:
        raise OracleSchemaError("--repeat-id disagrees with the run summary")
    if budget_level is not None and recorded_level != budget_level:
        raise OracleSchemaError("--budget-level disagrees with the run summary")
    if require_repeat and recorded_repeat is None:
        raise OracleSchemaError("official analysis requires recorded repeat_id")
    if require_budget_level and recorded_level is None:
        raise OracleSchemaError("official analysis requires recorded budget_level")
    if require_repeat or require_budget_level:
        execution_id = summary.get("execution_id")
        if not isinstance(execution_id, str) or not execution_id:
            raise OracleSchemaError("official analysis requires recorded execution_id")
    effective_repeat = (
        recorded_repeat
        if require_repeat
        else (repeat_id if repeat_id is not None else recorded_repeat)
    )
    effective_level = (
        recorded_level
        if require_budget_level
        else (budget_level if budget_level is not None else recorded_level)
    )
    for value, field in (
        (effective_repeat, "repeat_id"),
        (effective_level, "budget_level"),
    ):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise OracleSchemaError(f"{field} must be a positive integer")
    return effective_repeat, effective_level


def _official_tool_free_profiles(
    profile: OfficialBudgetProfile,
) -> tuple[ModelProfile, TwoStageProfile | None]:
    try:
        model = load_model_profiles()[profile.model_key]
    except (KeyError, OSError, ValueError) as exc:
        raise OracleSchemaError(
            "official Tool-Free model profile is missing or invalid"
        ) from exc
    if not isinstance(model.thinking_enabled, bool):
        raise OracleSchemaError(
            "official Tool-Free model has unresolved thinking_enabled"
        )
    if not model.thinking_enabled:
        return model, None

    try:
        candidates = [
            candidate
            for candidate in load_two_stage_profiles(
                resource_path("configs", "two_stage_profiles.yaml")
            ).values()
            if profile.model_key in candidate.applicable_models
            and profile.domain in candidate.applicable_domains
        ]
    except (OSError, ValueError) as exc:
        raise OracleSchemaError(
            "official two-stage profile registry is invalid"
        ) from exc
    if len(candidates) != 1:
        raise OracleSchemaError("official two-stage profile is missing or ambiguous")
    two_stage = candidates[0]
    if (
        two_stage.stage1_model_key != model.model_key
        or two_stage.stage1_thinking_enabled is not True
        or two_stage.stage2_thinking_enabled is not False
        or two_stage.stage2_protocol.include_original_problems is not False
        or "reasoning_content" not in two_stage.stage2_protocol.handoff_channels
        or two_stage.stage2_protocol.prompt_assembly
        not in {"coding_reasoning_visible_trace", "reasoning_visible_trace"}
    ):
        raise OracleSchemaError(
            "official two-stage profile is not a trace-only "
            "thinking/finalization protocol"
        )
    return model, two_stage


def _effective_stage1_only(requested: bool, profile: OfficialBudgetProfile) -> bool:
    model, two_stage = _official_tool_free_profiles(profile)
    if not model.thinking_enabled:
        if requested:
            raise OracleSchemaError(
                "official one-stage analysis cannot use --stage1-only"
            )
        return False
    if two_stage is None:
        raise OracleSchemaError("official thinking model has no two-stage profile")
    return True


def _contains_fields(value: object, expected: Mapping[str, Any]) -> bool:
    return isinstance(value, Mapping) and all(
        key in value
        and (
            value[key] is expected_value
            if isinstance(expected_value, bool)
            else value[key] == expected_value
        )
        for key, expected_value in expected.items()
    )


def _official_scorer_profile(domain: str) -> ScorerProfile:
    matches = [
        profile
        for profile in load_scorer_profiles().values()
        if profile.domain == domain
    ]
    if len(matches) != 1:
        raise OracleSchemaError(
            "official scorer profile is missing or ambiguous for the domain"
        )
    return matches[0]


def _validate_official_scoring_contract(
    summary: Mapping[str, Any],
    *,
    domain: str,
    results_bytes: bytes,
) -> ScorerProfile:
    try:
        profile = _official_scorer_profile(domain)
    except (OSError, ValueError) as exc:
        raise OracleSchemaError(
            "official scorer profile registry is missing or invalid"
        ) from exc
    expected_contract = scorer_profile_contract(profile)
    expected_contract_sha256 = scorer_profile_contract_sha256(profile)
    if (
        summary.get("scorer_profile") != profile.profile_id
        or summary.get("scorer_contract") != expected_contract
        or summary.get("scorer_contract_sha256") != expected_contract_sha256
    ):
        raise OracleSchemaError(
            "official analysis requires the released scorer contract"
        )
    if summary.get("results_sha256") != hashlib.sha256(results_bytes).hexdigest():
        raise OracleSchemaError(
            "official scoring result digest does not bind to judge_results.jsonl"
        )
    return profile


def _attempt_item_key(row: Mapping[str, Any]) -> tuple[str, str]:
    problem_id = row.get("problem_id")
    if isinstance(problem_id, str) and problem_id:
        return ("problem", problem_id)
    suite_id = row.get("suite_id")
    if isinstance(suite_id, str) and suite_id:
        return ("suite", suite_id)
    raise OracleSchemaError("official Tool-Free attempt has no item identity")


def _validate_tool_free_identity_chain(
    summary: Mapping[str, Any],
    attempts: list[Mapping[str, Any]],
    generation_rows: list[Mapping[str, Any]],
    *,
    expected_final_stage: str,
) -> None:
    run_id = summary.get("run_id")
    execution_id = summary.get("execution_id")
    repeat_id = summary.get("repeat_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not isinstance(execution_id, str)
        or not execution_id
        or isinstance(repeat_id, bool)
        or not isinstance(repeat_id, int)
        or repeat_id <= 0
    ):
        raise OracleSchemaError(
            "official Tool-Free summary has incomplete execution identity"
        )
    for row in (*attempts, *generation_rows):
        if (
            row.get("run_id") != run_id
            or row.get("execution_id") != execution_id
            or row.get("repeat_id") != repeat_id
            or row.get("source_setting") != "tool_free"
        ):
            raise OracleSchemaError(
                "official Tool-Free artifacts do not bind to the selected execution"
            )

    by_stage: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    for row in attempts:
        stage = row.get("stage")
        if not isinstance(stage, str):
            raise OracleSchemaError("official Tool-Free attempt has no stage")
        key = _attempt_item_key(row)
        stage_rows = by_stage.setdefault(stage, {})
        if key in stage_rows:
            raise OracleSchemaError(
                "official Tool-Free attempts duplicate one item and stage"
            )
        stage_rows[key] = row

    final_rows = by_stage.get(expected_final_stage, {})
    if not final_rows:
        raise OracleSchemaError("official Tool-Free run has no final-stage attempts")
    if expected_final_stage == "one_stage":
        if set(by_stage) != {"one_stage"}:
            raise OracleSchemaError("official one-stage attempts contain another stage")
    else:
        stage1_rows = by_stage.get("stage1", {})
        if set(by_stage) != {"stage1", "stage2"} or set(stage1_rows) != set(final_rows):
            raise OracleSchemaError(
                "official two-stage attempts are not a one-to-one item mapping"
            )
        for key, stage2 in final_rows.items():
            stage1 = stage1_rows[key]
            stage1_id = stage1.get("request_id")
            stage2_id = stage2.get("request_id")
            if (
                not isinstance(stage1_id, str)
                or not isinstance(stage2_id, str)
                or stage1.get("stage1_request_id") != stage1_id
                or stage1.get("stage2_request_id") is not None
                or stage2.get("parent_request_id") != stage1_id
                or stage2.get("stage1_request_id") != stage1_id
                or stage2.get("stage2_request_id") != stage2_id
            ):
                raise OracleSchemaError(
                    "official two-stage attempts do not form a one-to-one chain"
                )

    used_final_keys: set[tuple[str, str]] = set()
    for row in generation_rows:
        problem_key = ("problem", str(row.get("problem_id")))
        suite_key = ("suite", str(row.get("suite_id")))
        key = problem_key if problem_key in final_rows else suite_key
        final = final_rows.get(key)
        if final is None:
            raise OracleSchemaError(
                "official parsed output has no matching final-stage attempt"
            )
        used_final_keys.add(key)
        if (
            row.get("stage") != expected_final_stage
            or row.get("request_id") != final.get("request_id")
            or row.get("stage1_request_id") != final.get("stage1_request_id")
            or row.get("stage2_request_id") != final.get("stage2_request_id")
        ):
            raise OracleSchemaError(
                "official parsed output does not bind to its final-stage attempt"
            )
    if used_final_keys != set(final_rows):
        raise OracleSchemaError(
            "official final-stage attempt has no parsed output binding"
        )


def _validate_official_tool_free_protocol(
    summary: Mapping[str, Any],
    profile: OfficialBudgetProfile,
    *,
    run_dir: str | Path,
    score_rows: Iterable[Mapping[str, Any]],
    attempt_rows: Iterable[Mapping[str, Any]] | None = None,
    generation_rows: Iterable[Mapping[str, Any]] | None = None,
) -> None:
    model, two_stage = _official_tool_free_profiles(profile)
    expected_kind = "two_stage" if model.thinking_enabled else "one_stage"
    expected_final_stage = "stage2" if two_stage is not None else "one_stage"
    expected_final_model = (
        two_stage.stage2_model_key if two_stage is not None else model.model_key
    )
    final_models = load_model_profiles()
    try:
        expected_provider = final_models[expected_final_model].provider_profile
    except KeyError as exc:
        raise OracleSchemaError(
            "official final-stage model profile is missing"
        ) from exc

    protocol = summary.get("protocol")
    common = {
        "kind": expected_kind,
        "model_key": model.model_key,
        "model_thinking_enabled": model.thinking_enabled,
        "official_rho": profile.rho,
    }
    if (
        summary.get("stage") != expected_final_stage
        or summary.get("model_key") != model.model_key
        or summary.get("model_name") != expected_final_model
        or summary.get("provider_name") != expected_provider
        or not _contains_fields(protocol, common)
    ):
        raise OracleSchemaError(
            "official Tool-Free run stage or model protocol is inconsistent"
        )
    assert isinstance(protocol, Mapping)

    if two_stage is None:
        if not _contains_fields(
            protocol,
            {
                "two_stage_profile": None,
                "budget_accounting": "single_stage_output_tokens",
                "stage1": None,
                "stage2": None,
            },
        ):
            raise OracleSchemaError(
                "official non-thinking model requires one-stage metadata"
            )
    else:
        if (
            not _contains_fields(
                protocol,
                {
                    "two_stage_profile": two_stage.profile_id,
                    "budget_accounting": two_stage.budget_accounting,
                },
            )
            or not _contains_fields(
                protocol.get("stage1"),
                {
                    "model_key": two_stage.stage1_model_key,
                    "thinking_enabled": True,
                    "accounting": "reported_output_tokens",
                },
            )
            or not _contains_fields(
                protocol.get("stage2"),
                {
                    "model_key": two_stage.stage2_model_key,
                    "thinking_enabled": False,
                    "accounting": "not_counted",
                    "practical_output_token_cap": two_stage.stage2_practical_cap,
                    "handoff_channels": list(
                        two_stage.stage2_protocol.handoff_channels
                    ),
                    "prompt_assembly": two_stage.stage2_protocol.prompt_assembly,
                    "include_original_problems": False,
                    "trace_only": True,
                },
            )
        ):
            raise OracleSchemaError(
                "official two-stage run lacks trace-only Stage 2 metadata"
            )

    attempts_path = Path(run_dir) / "attempts.jsonl"
    if not attempts_path.is_file():
        raise OracleSchemaError("official Tool-Free run requires recorded attempts")
    attempts = (
        list(attempt_rows) if attempt_rows is not None else read_jsonl(attempts_path)
    )
    request_ids = [row.get("request_id") for row in attempts]
    if (
        not attempts
        or summary.get("attempt_count") != len(attempts)
        or any(not isinstance(value, str) or not value for value in request_ids)
        or len(set(request_ids)) != len(request_ids)
    ):
        raise OracleSchemaError("official Tool-Free attempt provenance is incomplete")
    if two_stage is None:
        if any(
            row.get("stage") != "one_stage"
            or row.get("stage_input_kind") != "public_prompt"
            or row.get("parent_request_id") is not None
            for row in attempts
        ):
            raise OracleSchemaError(
                "official one-stage attempts disagree with run_summary"
            )
    else:
        stage1 = [row for row in attempts if row.get("stage") == "stage1"]
        stage2 = [row for row in attempts if row.get("stage") == "stage2"]
        stage1_ids = {row.get("request_id") for row in stage1}
        if (
            not stage1
            or len(stage1) != len(stage2)
            or len(stage1) + len(stage2) != len(attempts)
            or any(
                row.get("stage_input_kind") != "public_prompt"
                or row.get("parent_request_id") is not None
                or row.get("stage1_request_id") != row.get("request_id")
                or row.get("stage2_request_id") is not None
                for row in stage1
            )
            or any(
                row.get("stage_input_kind") != "stage1_output"
                or row.get("parent_request_id") not in stage1_ids
                or row.get("stage1_request_id") != row.get("parent_request_id")
                or row.get("stage2_request_id") != row.get("request_id")
                for row in stage2
            )
        ):
            raise OracleSchemaError(
                "official two-stage attempts do not form a complete Stage 1/2 chain"
            )
    if any(row.get("stage") != expected_final_stage for row in score_rows):
        raise OracleSchemaError(
            "official scoring rows do not target the protocol final stage"
        )
    if generation_rows is not None:
        generation = list(generation_rows)
        parsed_count = sum(row.get("parse_status") == "parsed" for row in generation)
        if (
            summary.get("problem_count") != len(generation)
            or summary.get("parsed_count") != parsed_count
        ):
            raise OracleSchemaError(
                "official Tool-Free output counts disagree with run_summary"
            )
        _validate_tool_free_identity_chain(
            summary,
            attempts,
            generation,
            expected_final_stage=expected_final_stage,
        )


def _validate_official_provenance(
    args: argparse.Namespace,
    profile: OfficialBudgetProfile,
    *,
    expected_budget: int,
) -> None:
    scoring_summary = read_json(Path(args.scoring_dir) / "scoring_summary.json")
    if not isinstance(scoring_summary, Mapping) or (
        scoring_summary.get("status") != "complete"
        or scoring_summary.get("domain") != args.domain
        or scoring_summary.get("scoring_mode") != "production"
    ):
        raise OracleSchemaError(
            "official analysis requires complete production scoring provenance"
        )
    score_bytes, score_rows = read_jsonl_snapshot(
        Path(args.scoring_dir) / "judge_results.jsonl"
    )
    scorer_profile = _validate_official_scoring_contract(
        scoring_summary,
        domain=args.domain,
        results_bytes=score_bytes,
    )
    if not score_rows or any(
        row.get("scoring_mode") != "production"
        or row.get("evaluator") != scorer_profile.profile_id
        for row in score_rows
    ):
        raise OracleSchemaError(
            "official analysis requires production scoring on every result row"
        )

    saved_name = (
        "parsed_answers.jsonl" if args.setting == "tool_free" else "saved_outputs.jsonl"
    )
    saved_path = Path(args.run_dir) / saved_name
    if not saved_path.is_file():
        raise OracleSchemaError(
            "official scoring input digest does not bind to the selected run"
        )
    saved_bytes, generation_rows = read_jsonl_snapshot(saved_path)
    if scoring_summary.get("input_sha256") != hashlib.sha256(saved_bytes).hexdigest():
        raise OracleSchemaError(
            "official scoring input digest does not bind to the selected run"
        )
    generation_by_id = {
        row.get("problem_id"): row
        for row in generation_rows
        if isinstance(row.get("problem_id"), str)
    }
    score_by_id = {
        row.get("problem_id"): row
        for row in score_rows
        if isinstance(row.get("problem_id"), str)
    }
    if (
        len(generation_by_id) != len(generation_rows)
        or len(score_by_id) != len(score_rows)
        or set(generation_by_id) != set(score_by_id)
    ):
        raise OracleSchemaError(
            "official scoring rows do not bind one-to-one to generation outputs"
        )
    binding_fields = (
        (
            "run_id",
            "request_id",
            "stage",
            "stage1_request_id",
            "stage2_request_id",
            "execution_id",
            "repeat_id",
            "source_setting",
        )
        if args.setting == "tool_free"
        else ("execution_id", "task_id", "model_key", "source_setting")
    )
    for problem_id, generation_row in generation_by_id.items():
        score_row = score_by_id[problem_id]
        if any(
            score_row.get(field) != generation_row.get(field)
            for field in binding_fields
        ):
            raise OracleSchemaError(
                "official scoring provenance differs from the selected generation run"
            )
        required_fields = (
            ("run_id", "request_id", "stage", "execution_id", "source_setting")
            if args.setting == "tool_free"
            else ("execution_id", "task_id", "model_key")
        )
        if any(
            not isinstance(generation_row.get(field), str) or not generation_row[field]
            for field in required_fields
        ):
            raise OracleSchemaError(
                "official generation output has incomplete scoring provenance"
            )
        if args.setting == "tool_free":
            repeat_id = generation_row.get("repeat_id")
            if (
                isinstance(repeat_id, bool)
                or not isinstance(repeat_id, int)
                or repeat_id <= 0
                or generation_row.get("source_setting") != "tool_free"
            ):
                raise OracleSchemaError(
                    "official generation output has incomplete repeat provenance"
                )

    summary_name = (
        "run_summary.json" if args.setting == "tool_free" else "backend_summary.json"
    )
    summary = read_json(Path(args.run_dir) / summary_name)
    if not isinstance(summary, Mapping):
        raise OracleSchemaError("official run summary must contain an object")
    if args.setting == "tool_free":
        attempts_path = Path(args.run_dir) / "attempts.jsonl"
        if not attempts_path.is_file():
            raise OracleSchemaError("official Tool-Free run requires recorded attempts")
        attempt_bytes, attempt_rows = read_jsonl_snapshot(attempts_path)
        if (
            summary.get("attempts_sha256") != hashlib.sha256(attempt_bytes).hexdigest()
            or summary.get("parsed_answers_sha256")
            != hashlib.sha256(saved_bytes).hexdigest()
        ):
            raise OracleSchemaError(
                "official Tool-Free artifact digests do not bind to run_summary"
            )
        provider = summary.get("provider_name")
        budget = summary.get("budget")
        if (
            provider in {"mock", "replay"}
            or not isinstance(provider, str)
            or summary.get("domain") != args.domain
            or not isinstance(budget, Mapping)
            or budget.get("condition_kind") != "official_profile"
            or budget.get("profile_id") != profile.profile_id
            or budget.get("unit") != "output_tokens"
            or budget.get("value") != expected_budget
        ):
            raise OracleSchemaError(
                "official Tool-Free analysis requires matching real-run provenance"
            )
        _validate_official_tool_free_protocol(
            summary,
            profile,
            run_dir=args.run_dir,
            score_rows=score_rows,
            attempt_rows=attempt_rows,
            generation_rows=generation_rows,
        )
    else:
        try:
            execution_profile = resolve_agentic_execution_profile(args.model)
        except ExternalAgenticBackendError as exc:
            raise OracleSchemaError(
                "official Agentic model execution profile is invalid"
            ) from exc
        budget = summary.get("budget_resolution")
        if (
            summary.get("backend") != "harbor"
            or summary.get("environment") != "docker"
            or summary.get("agent") != "terminus-2"
            or summary.get("paper_equivalent_runtime") is not True
            or summary.get("domain") != args.domain
            or summary.get("model_key") != args.model
            or summary.get("public_model_id") != execution_profile.public_model_id
            or summary.get("execution_profile") != execution_profile.to_dict()
            or summary.get("model_api_called") is not True
            or summary.get("raw_trajectory_saved") is not True
            or summary.get("trajectory_complete") is not True
            or summary.get("trajectory_format") != "ATIF"
            or not isinstance(summary.get("trajectory_sha256"), str)
            or len(summary["trajectory_sha256"]) != 64
            or not isinstance(budget, Mapping)
            or budget.get("condition_kind") != "official_profile"
            or budget.get("profile_id") != profile.profile_id
            or budget.get("unit") != "counted_actions"
            or budget.get("value") != expected_budget
        ):
            raise OracleSchemaError(
                "official Agentic analysis requires Harbor/Terminus-2 provenance"
            )
        try:
            handoff = validate_external_backend_handoff(
                args.run_dir,
                Path(args.run_dir) / "task_binding",
                expected_execution_profile=execution_profile,
            )
            expected_generation = collect_agentic_saved_outputs(args.run_dir)
        except (ExternalAgenticBackendError, OSError, ValueError) as exc:
            raise OracleSchemaError(
                "official Agentic handoff could not be fully revalidated"
            ) from exc
        if any(summary.get(key) != value for key, value in handoff.items()):
            raise OracleSchemaError(
                "official Agentic trajectory digest differs from the validated handoff"
            )
        if generation_rows != list(expected_generation):
            raise OracleSchemaError(
                "official Agentic saved outputs differ from validated final artifacts"
            )
        if any(
            row.get("execution_id") != summary.get("execution_id")
            or row.get("task_id") != summary.get("task_id")
            or row.get("model_key") != summary.get("model_key")
            or row.get("source_setting") != "agentic"
            for row in generation_rows
        ):
            raise OracleSchemaError(
                "official Agentic scoring does not bind to the selected episode"
            )


def _build_response_curve(args: argparse.Namespace) -> int:
    profile, _, official_profile = _safe_condition(
        args, expected_role="single_problem_response_curve"
    )
    if official_profile is not None and args.condition_id != profile:
        raise OracleSchemaError(
            "official response-curve condition_id must equal its budget profile ID"
        )
    if args.budget < 0:
        raise OracleSchemaError("budget must be non-negative")
    if not args.run_dir or not args.scoring_dir:
        raise OracleSchemaError(
            "all response-curve points, including budget zero, require "
            "--run-dir and --scoring-dir"
        )
    else:
        repeat_id, budget_level = _execution_metadata(
            args.run_dir,
            setting=args.setting,
            repeat_id=args.repeat_id,
            budget_level=args.budget_level,
            require_repeat=official_profile is not None,
            require_budget_level=official_profile is not None,
        )
        if official_profile is not None:
            if args.allow_unjudged:
                raise OracleSchemaError(
                    "official response curves cannot use --allow-unjudged"
                )
            assert budget_level is not None
            if official_profile.budget_grid[budget_level - 1] != args.budget:
                raise OracleSchemaError(
                    "official response-curve budget does not match its recorded level"
                )
            _validate_official_provenance(
                args, official_profile, expected_budget=args.budget
            )
        score_rows = read_jsonl(Path(args.scoring_dir) / "judge_results.jsonl")
        ids = {row.get("problem_id") for row in score_rows}
        if not ids or any(not isinstance(value, str) for value in ids):
            raise OracleSchemaError("scoring results have invalid problem IDs")
        problems = _selected_problems(
            args.domain,
            args.data,
            problem_ids={str(value) for value in ids},
            strict=not args.relaxed,
        )
        if args.setting == "tool_free":
            stage1_only = args.stage1_only
            if official_profile is not None:
                stage1_only = _effective_stage1_only(args.stage1_only, official_profile)
            points = response_curve_points_from_outputs(
                run_dir=args.run_dir,
                scoring_dir=args.scoring_dir,
                problems=problems,
                domain=args.domain,
                model_key=args.model,
                budget=args.budget,
                stage1_only=stage1_only,
                repeat_id=repeat_id,
                budget_level=budget_level,
                allow_unjudged=args.allow_unjudged,
            )
        else:
            if len(problems) != 1:
                raise OracleSchemaError(
                    "one Agentic response-curve episode must bind exactly one problem"
                )
            points = (
                response_curve_point_from_agentic_outputs(
                    run_dir=args.run_dir,
                    scoring_dir=args.scoring_dir,
                    problem=problems[0],
                    model_key=args.model,
                    budget=args.budget,
                    repeat_id=repeat_id,
                    budget_level=budget_level,
                    allow_unjudged=args.allow_unjudged,
                ),
            )
    rows = []
    for point in points:
        row = to_public_dict(point)
        row.update(
            {
                "schema_version": "3.0",
                "condition_id": args.condition_id,
                "condition_kind": args.condition_kind,
            }
        )
        rows.append(row)
    count = _write_jsonl(args.output, rows, args.append)
    print(f"wrote {count} response-curve point(s) to {args.output}")
    return 0


def _contest_problems(args: argparse.Namespace):
    summary = read_json(Path(args.run_dir) / "backend_summary.json")
    if not isinstance(summary, Mapping) or not isinstance(summary.get("suite_id"), str):
        raise OracleSchemaError("Agentic backend summary has no suite_id")
    suites = load_contest_suites(
        args.domain, "test", resolve_path(args.data), strict=not args.relaxed
    )
    matches = [suite for suite in suites if suite.suite_id == summary["suite_id"]]
    if len(matches) != 1:
        raise OracleSchemaError("Agentic run suite is absent from public data")
    return matches[0].problems


def _build_contest(args: argparse.Namespace) -> int:
    profile, rho, official_profile = _safe_condition(
        args,
        expected_role=("budgeted_rho_0p2" if args.rho == 0.2 else "budgeted_rho_0p8"),
    )
    if args.contest_budget < 0:
        raise OracleSchemaError("contest budget must be non-negative")
    repeat_id, _ = _execution_metadata(
        args.run_dir,
        setting=args.setting,
        repeat_id=args.repeat_id,
        require_repeat=official_profile is not None,
    )
    if official_profile is not None:
        if args.allow_unjudged:
            raise OracleSchemaError("official contests cannot use --allow-unjudged")
        if (
            official_profile.budget_value != args.contest_budget
            or _official_curve_grid(official_profile) != args.response_curve_grid
        ):
            raise OracleSchemaError(
                "official contest budget or response-curve grid differs from its profile"
            )
        _validate_official_provenance(
            args, official_profile, expected_budget=args.contest_budget
        )
    if args.setting == "tool_free":
        legacy = contest_results_from_outputs(
            run_dir=args.run_dir,
            scoring_dir=args.scoring_dir,
            domain=args.domain,
            model_key=args.model,
            rho=0.2,
            contest_budget=args.contest_budget,
            repeat_id=repeat_id,
            allow_unjudged=args.allow_unjudged,
        )
    else:
        legacy = contest_results_from_agentic_outputs(
            run_dir=args.run_dir,
            scoring_dir=args.scoring_dir,
            problems=_contest_problems(args),
            model_key=args.model,
            rho=0.2,
            contest_budget=args.contest_budget,
            repeat_id=repeat_id,
            allow_unjudged=args.allow_unjudged,
        )
    rows: list[dict[str, Any]] = []
    for record in legacy:
        row = to_public_dict(record)
        row.pop("formal_contest_budget")
        row.pop("rho")
        row.update(
            {
                "schema_version": "3.0",
                "condition_id": args.condition_id,
                "condition_kind": args.condition_kind,
                "contest_budget": args.contest_budget,
                "budget_profile": profile,
                "rho": rho,
            }
        )
        rows.append(row)
    count = _write_jsonl(args.output, rows, args.append)
    if args.budget_output:
        record = ConditionBudget(
            domain=args.domain,
            model_key=args.model,
            setting=args.setting,
            budget_unit=(
                "output_tokens" if args.setting == "tool_free" else "counted_actions"
            ),
            condition_id=args.condition_id,
            condition_kind=args.condition_kind,
            contest_budget=args.contest_budget,
            response_curve_grid=args.response_curve_grid,
            budget_profile=profile,
            rho=rho,
        )
        Path(args.budget_output).parent.mkdir(parents=True, exist_ok=True)
        write_condition_budget_document(args.budget_output, (record,))
    print(f"wrote {count} contest result row(s) to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    response = sub.add_parser("build-response-curve")
    response.add_argument("--setting", choices=("tool_free", "agentic"), required=True)
    response.add_argument(
        "--domain", choices=("coding", "math", "abstract_reasoning"), required=True
    )
    response.add_argument("--model", required=True)
    response.add_argument("--data", required=True)
    response.add_argument("--run-dir")
    response.add_argument("--scoring-dir")
    response.add_argument("--budget", type=int, required=True)
    response.add_argument("--condition-id", default="response_curve")
    response.add_argument(
        "--condition-kind", choices=("custom", "official_profile"), default="custom"
    )
    response.add_argument("--budget-profile")
    response.add_argument("--rho", type=float)
    response.add_argument("--source-run-id")
    response.add_argument("--repeat-id", type=int)
    response.add_argument("--budget-level", type=int, choices=range(1, 7))
    response.add_argument("--limit-problems", type=int, default=1)
    response.add_argument("--stage1-only", action="store_true")
    response.add_argument("--allow-unjudged", action="store_true")
    response.add_argument("--relaxed", action="store_true")
    response.add_argument("--append", action="store_true")
    response.add_argument("--output", type=Path, required=True)

    contest = sub.add_parser("build-contest-results")
    contest.add_argument("--setting", choices=("tool_free", "agentic"), required=True)
    contest.add_argument(
        "--domain", choices=("coding", "math", "abstract_reasoning"), required=True
    )
    contest.add_argument("--model", required=True)
    contest.add_argument("--data")
    contest.add_argument("--run-dir", required=True)
    contest.add_argument("--scoring-dir", required=True)
    contest.add_argument("--contest-budget", type=int, required=True)
    contest.add_argument("--condition-id", required=True)
    contest.add_argument(
        "--condition-kind", choices=("custom", "official_profile"), default="custom"
    )
    contest.add_argument("--budget-profile")
    contest.add_argument("--rho", type=float)
    contest.add_argument("--response-curve-grid", type=_grid, required=True)
    contest.add_argument("--repeat-id", type=int)
    contest.add_argument("--budget-output", type=Path)
    contest.add_argument("--allow-unjudged", action="store_true")
    contest.add_argument("--relaxed", action="store_true")
    contest.add_argument("--append", action="store_true")
    contest.add_argument("--output", type=Path, required=True)

    compare = sub.add_parser("compare")
    compare.add_argument("--response-curve", required=True)
    compare.add_argument("--contest-results", required=True)
    compare.add_argument("--budgets", required=True)
    compare.add_argument("--output-dir", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "build-response-curve":
            return _build_response_curve(args)
        if args.action == "build-contest-results":
            if args.setting == "agentic" and not args.data:
                raise OracleSchemaError("Agentic contest conversion requires --data")
            return _build_contest(args)
        summary = run_condition_analysis(
            response_curve_path=args.response_curve,
            contest_results_path=args.contest_results,
            budgets_path=args.budgets,
            output_dir=args.output_dir,
        )
        print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError, OracleSchemaError) as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
