#!/usr/bin/env python3
"""Run one bounded R3Bench protocol condition through the shared evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

import yaml

from r3bench.agentic.external_backend import (
    check_external_backend_readiness,
    load_external_backend_config,
    resolve_agentic_execution_profile,
    run_external_agentic_backend,
)
from r3bench.agentic.runtime import (
    AgenticEpisodeConfig,
    MockAgenticProvider,
    ReplayAgenticProvider,
    default_mock_responses,
    run_agentic_episode,
)
from r3bench.agentic.scoring_handoff import write_agentic_saved_outputs
from r3bench.agentic.task_export import (
    export_agentic_response_curve_tasks,
    export_agentic_tasks,
)
from r3bench.commands import nl_contest, nl_single
from r3bench.common.budget import (
    BudgetResolutionError,
    ResolvedBudget,
    resolve_budget,
    resolve_budget_grid,
)
from r3bench.common.profile_registry import (
    ModelProfile,
    RunProfile,
    load_model_profiles,
    load_run_profiles,
    validate_run_profile_applicability,
)
from r3bench.common.io import read_jsonl
from r3bench.common.two_stage_profile import (
    TwoStageProfile,
    load_two_stage_profiles,
)
from r3bench.resource_paths import resolve_path


class UnifiedEvaluationError(ValueError):
    """Raised when unified selection cannot resolve one safe condition."""


def _data_source(domain: str, explicit: str | None) -> str:
    if explicit:
        path = Path(explicit)
        if path.is_absolute() or ".." in path.parts:
            raise UnifiedEvaluationError(
                "--data must be a safe relative or hf:// source"
            )
        return explicit
    filename = (
        "abstract_reasoning.jsonl"
        if domain == "abstract_reasoning"
        else f"{domain}.jsonl"
    )
    return f"public_data/{filename}"


def _prompt(domain: str, mode: str) -> tuple[str, str | None]:
    suffix = "contest" if mode == "contest" else "single"
    system = (
        f"prompts/{domain}/{suffix}_nl_system.txt"
        if domain in {"math", "abstract_reasoning"}
        else None
    )
    return f"prompts/{domain}/{suffix}_nl.txt", system


def _model_profile(args: argparse.Namespace) -> ModelProfile | None:
    if args.provider != "real":
        return None
    profiles = load_model_profiles(resolve_path(args.model_profiles))
    try:
        return profiles[args.model]
    except KeyError as exc:
        raise UnifiedEvaluationError(
            "real execution requires the selected model in --model-profiles"
        ) from exc


def _run_profile(
    args: argparse.Namespace,
    model: ModelProfile,
) -> RunProfile:
    profiles = load_run_profiles(resolve_path(args.run_profiles))
    if args.run_profile:
        try:
            selected = profiles[args.run_profile]
        except KeyError as exc:
            raise UnifiedEvaluationError("unknown --run-profile") from exc
        validate_run_profile_applicability(
            selected,
            model_key=model.model_key,
            provider_profile=model.provider_profile,
            domain=args.domain,
            setting=args.setting,
        )
        return selected
    candidates: list[RunProfile] = []
    for profile in profiles.values():
        limits = profile.request_safety_limits
        if isinstance(limits, Mapping) and limits.get("generation_disabled") is True:
            continue
        try:
            validate_run_profile_applicability(
                profile,
                model_key=model.model_key,
                provider_profile=model.provider_profile,
                domain=args.domain,
                setting=args.setting,
            )
        except ValueError:
            continue
        candidates.append(profile)
    if len(candidates) != 1:
        raise UnifiedEvaluationError(
            "run profile is ambiguous or missing; provide --run-profile"
        )
    return candidates[0]


def _two_stage_profile(args: argparse.Namespace) -> TwoStageProfile:
    profiles = load_two_stage_profiles(resolve_path(args.two_stage_profiles))
    if args.two_stage_profile:
        try:
            return profiles[args.two_stage_profile]
        except KeyError as exc:
            raise UnifiedEvaluationError("unknown --two-stage-profile") from exc
    candidates = [
        profile
        for profile in profiles.values()
        if args.model in profile.applicable_models
        and args.domain in profile.applicable_domains
    ]
    if len(candidates) != 1:
        raise UnifiedEvaluationError(
            "two-stage profile is ambiguous or missing; provide --two-stage-profile"
        )
    return candidates[0]


def _official_tool_free_protocol_metadata(
    args: argparse.Namespace,
    *,
    official_rho: float | None,
) -> dict[str, Any]:
    """Bind an official Tool-Free condition to the released model protocol."""

    profiles = load_model_profiles(resolve_path(args.model_profiles))
    try:
        model = profiles[args.model]
    except KeyError as exc:
        raise UnifiedEvaluationError(
            "official Tool-Free execution requires the selected model in "
            "--model-profiles"
        ) from exc
    if not isinstance(model.thinking_enabled, bool):
        raise UnifiedEvaluationError(
            "official Tool-Free execution requires resolved thinking_enabled"
        )
    expected_kind = "two_stage" if model.thinking_enabled else "one_stage"
    if args.protocol != expected_kind:
        raise UnifiedEvaluationError(
            f"official Tool-Free model {args.model!r} requires --protocol "
            f"{expected_kind}"
        )

    metadata: dict[str, Any] = {
        "kind": expected_kind,
        "model_key": model.model_key,
        "model_thinking_enabled": model.thinking_enabled,
        "two_stage_profile": None,
        "official_rho": official_rho,
        "budget_accounting": "single_stage_output_tokens",
        "stage1": None,
        "stage2": None,
    }
    if not model.thinking_enabled:
        if args.two_stage_profile is not None:
            raise UnifiedEvaluationError(
                "official one-stage execution cannot select --two-stage-profile"
            )
        return metadata

    profile = _two_stage_profile(args)
    if (
        profile.stage1_model_key != model.model_key
        or profile.stage1_thinking_enabled is not True
        or profile.stage2_thinking_enabled is not False
        or profile.stage2_protocol.include_original_problems is not False
        or "reasoning_content" not in profile.stage2_protocol.handoff_channels
        or profile.stage2_protocol.prompt_assembly
        not in {"coding_reasoning_visible_trace", "reasoning_visible_trace"}
    ):
        raise UnifiedEvaluationError(
            "official two-stage profile is not a trace-only thinking/finalization "
            "protocol"
        )
    args.two_stage_profile = profile.profile_id
    metadata.update(
        {
            "two_stage_profile": profile.profile_id,
            "budget_accounting": profile.budget_accounting,
            "stage1": {
                "model_key": profile.stage1_model_key,
                "thinking_enabled": profile.stage1_thinking_enabled,
                "accounting": "reported_output_tokens",
            },
            "stage2": {
                "model_key": profile.stage2_model_key,
                "thinking_enabled": profile.stage2_thinking_enabled,
                "accounting": "not_counted",
                "practical_output_token_cap": profile.stage2_practical_cap,
                "handoff_channels": list(profile.stage2_protocol.handoff_channels),
                "prompt_assembly": profile.stage2_protocol.prompt_assembly,
                "include_original_problems": (
                    profile.stage2_protocol.include_original_problems
                ),
                "trace_only": True,
            },
        }
    )
    return metadata


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _pure_config(
    args: argparse.Namespace,
    *,
    mode: str,
    max_tokens: int | None,
    model: ModelProfile | None,
) -> dict[str, Any]:
    prompt, system = _prompt(args.domain, mode)
    provider_name = model.provider_profile if model else args.provider
    provider: dict[str, Any] = {"name": provider_name, "model": args.model}
    if model is not None:
        provider["api_key_env"] = model.api_key_env
    prompt_config: dict[str, Any] = {"template_path": prompt}
    if system is not None:
        prompt_config["system_template_path"] = system
    return {
        "name": f"unified_{args.setting}_{args.domain}_{mode}_{args.model}",
        "domain": args.domain,
        "mode": mode,
        "visibility": args.visibility,
        "setting": "tool_free",
        "stage": "one_stage",
        "split": "test",
        "strict_data": not args.toy,
        "data_source": _data_source(args.domain, args.data),
        "provider": provider,
        "budget": {
            "max_tokens": max_tokens,
            "temperature": (model.temperature if model is not None else 0.0),
            "top_p": model.top_p if model is not None else None,
            "action_budget": None,
        },
        "prompt": prompt_config,
        "judge": {"profile_name": f"mock_{args.domain}"},
    }


def _repeat_ids(args: argparse.Namespace) -> tuple[int, ...]:
    if args.repeat_id is not None:
        return (args.repeat_id,)
    return tuple(range(1, args.repeats + 1))


def _repeat_output_dir(
    parent: Path,
    repeat_id: int,
    *,
    preserve_single_layout: bool,
) -> Path:
    return parent if preserve_single_layout else parent / f"repeat_{repeat_id}"


def _annotate_repeat_metadata(
    path: Path,
    *,
    repeat_id: int,
    repeat_count: int,
    budget_level: int | None = None,
    budget_override: Mapping[str, Any] | None = None,
    model_key: str | None = None,
    protocol_override: Mapping[str, Any] | None = None,
) -> None:
    execution_id = uuid4().hex
    artifact_digests: dict[str, str] = {}
    for filename in ("attempts.jsonl", "parsed_answers.jsonl"):
        artifact_path = path / filename
        if not artifact_path.is_file():
            continue
        rows = read_jsonl(artifact_path)
        for row in rows:
            row["execution_id"] = execution_id
            row["repeat_id"] = repeat_id
            row["source_setting"] = "tool_free"
        payload = "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
            for row in rows
        ).encode("utf-8")
        artifact_path.write_bytes(payload)
        artifact_digests[f"{filename.removesuffix('.jsonl')}_sha256"] = hashlib.sha256(
            payload
        ).hexdigest()
    for filename in ("run_summary.json", "backend_summary.json"):
        summary_path = path / filename
        if not summary_path.is_file():
            continue
        value = json.loads(summary_path.read_text(encoding="utf-8"))
        value["repeat_id"] = repeat_id
        value["repeat_count"] = repeat_count
        value["execution_id"] = execution_id
        value.update(artifact_digests)
        if budget_level is not None:
            value["budget_level"] = budget_level
        if budget_override is not None:
            value["budget"] = dict(budget_override)
        if model_key is not None:
            value["model_key"] = model_key
        if protocol_override is not None:
            value["protocol"] = dict(protocol_override)
        summary_path.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _publish_primary_repeat(target: Path, primary: Path) -> None:
    """Mirror repeat 1's flat artifacts for pre-repeat consumers."""

    if target == primary:
        return
    for source in primary.iterdir():
        if source.is_file():
            shutil.copyfile(source, target / source.name)


def _nl_tail(
    args: argparse.Namespace,
    *,
    config_path: Path,
    output_dir: Path,
    budget_value: int | None,
    budget_profile: str | None,
    mode: str,
    model: ModelProfile | None,
    run: RunProfile | None,
) -> list[str]:
    tail = [
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--provider",
        args.provider,
        "--condition-id",
        args.condition_id or (budget_profile or f"custom_output_tokens_{budget_value}"),
    ]
    if budget_value is not None:
        tail.extend(["--output-token-budget", str(budget_value)])
    elif budget_profile is not None:
        tail.extend(["--budget-profile", budget_profile])
    if args.provider == "replay":
        if not args.replay_file:
            raise UnifiedEvaluationError("replay provider requires --replay-file")
        tail.extend(["--replay-file", args.replay_file])
    if args.protocol == "two_stage":
        tail.append("--two-stage")
        profile = None
        if args.two_stage_profile or args.provider == "real" or budget_profile:
            profile = _two_stage_profile(args)
        if profile is not None:
            tail.extend(
                [
                    "--two-stage-profiles",
                    str(resolve_path(args.two_stage_profiles)),
                    "--two-stage-profile",
                    profile.profile_id,
                ]
            )
            if args.provider != "real":
                tail.extend(["--model-key", args.model])
    if args.provider == "real":
        assert model is not None and run is not None
        provider_path = (
            args.provider_profile or f"configs/providers/{model.provider_profile}.yaml"
        )
        tail.extend(
            [
                "--provider-profile",
                str(resolve_path(provider_path)),
                "--model-profiles",
                str(resolve_path(args.model_profiles)),
                "--evaluator-profiles",
                str(resolve_path(args.evaluator_profiles)),
                "--run-profiles",
                str(resolve_path(args.run_profiles)),
                "--run-profile",
                run.profile_id,
                "--model-key",
                args.model,
            ]
        )
        if args.dry_run:
            tail.append("--dry-run")
        elif args.allow_real_api:
            tail.append("--allow-real-api")
        else:
            raise UnifiedEvaluationError(
                "real execution requires --dry-run or --allow-real-api"
            )
    if args.confirm_full_run:
        tail.append("--confirm-full-run")
    if mode == "contest":
        tail.extend(["--limit-suites", str(args.limit_suites)])
    else:
        tail.extend(["--limit", str(args.limit_problems)])
    return tail


def _run_tool_free(args: argparse.Namespace) -> dict[str, Any]:
    model = _model_profile(args)
    run = _run_profile(args, model) if model is not None else None
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    repeat_ids = _repeat_ids(args)
    preserve_single_layout = args.repeats == 1 and args.repeat_id is None
    if args.mode == "response_curve":
        grid = resolve_budget_grid(
            setting="tool_free",
            explicit_values=args.budget_grid,
            profile_id=args.budget_profile,
            domain=args.domain,
            model_key=args.model,
        )
        official_protocol = (
            _official_tool_free_protocol_metadata(args, official_rho=None)
            if args.budget_profile and args.budget_grid is None
            else None
        )
        if (
            args.provider == "real"
            and not args.dry_run
            and len(grid) * len(repeat_ids) > 1
            and not args.confirm_full_run
        ):
            raise UnifiedEvaluationError(
                "real response-curve execution with multiple episodes requires "
                "--confirm-full-run"
            )
        records: list[dict[str, Any]] = []
        for budget_level, value in enumerate(grid, start=1):
            level_dir = target / f"level_{budget_level}_budget_{value}"
            config_path = target / (
                f"resolved_experiment_level_{budget_level}_budget_{value}.yaml"
            )
            _write_yaml(
                config_path,
                _pure_config(
                    args, mode="single_problem", max_tokens=value, model=model
                ),
            )
            for repeat_id in repeat_ids:
                child = _repeat_output_dir(
                    level_dir,
                    repeat_id,
                    preserve_single_layout=preserve_single_layout,
                )
                tail = _nl_tail(
                    args,
                    config_path=config_path,
                    output_dir=child,
                    budget_value=value,
                    budget_profile=None,
                    mode="single_problem",
                    model=model,
                    run=run,
                )
                code = nl_single.main(tail)
                if code != 0:
                    raise UnifiedEvaluationError(
                        "Tool-Free response-curve episode failed at "
                        f"level {budget_level}, budget {value}, repeat {repeat_id}"
                    )
                _annotate_repeat_metadata(
                    child,
                    repeat_id=repeat_id,
                    repeat_count=args.repeats,
                    budget_level=budget_level,
                    budget_override=(
                        {
                            "value": value,
                            "unit": "output_tokens",
                            "source": "official_profile",
                            "condition_id": args.budget_profile,
                            "condition_kind": "official_profile",
                            "profile_id": args.budget_profile,
                            "rho": None,
                        }
                        if args.budget_profile and args.budget_grid is None
                        else None
                    ),
                    model_key=args.model if official_protocol is not None else None,
                    protocol_override=official_protocol,
                )
                records.append(
                    {
                        "budget_level": budget_level,
                        "budget": value,
                        "repeat_id": repeat_id,
                        "status": "dry_run" if args.dry_run else "complete",
                        "model_api_called": not args.dry_run,
                    }
                )
        summary = {
            "schema_version": "1.0",
            "status": "dry_run" if args.dry_run else "complete",
            "setting": "tool_free",
            "mode": "response_curve",
            "budget_unit": "output_tokens",
            "budget_levels": len(grid),
            "repeat_count": args.repeats,
            "executed_repeat_count": len(repeat_ids),
            "condition_kind": (
                "official_profile"
                if args.budget_profile and args.budget_grid is None
                else "custom"
            ),
            "budget_profile": (
                args.budget_profile
                if args.budget_profile and args.budget_grid is None
                else None
            ),
            "protocol": official_protocol,
            "episodes": records,
        }
    else:
        budget = resolve_budget(
            setting="tool_free",
            explicit_value=args.budget,
            config_value=None,
            profile_id=args.budget_profile,
            condition_id=args.condition_id,
            domain=args.domain,
            model_key=args.model,
        )
        official_protocol = (
            _official_tool_free_protocol_metadata(args, official_rho=budget.rho)
            if budget.source == "official_profile"
            else None
        )
        config_path = target / "resolved_experiment.yaml"
        config_budget = None if budget.source == "official_profile" else budget.value
        _write_yaml(
            config_path,
            _pure_config(args, mode=args.mode, max_tokens=config_budget, model=model),
        )
        command = nl_contest.main if args.mode == "contest" else nl_single.main
        if (
            args.provider == "real"
            and not args.dry_run
            and len(repeat_ids) > 1
            and not args.confirm_full_run
        ):
            raise UnifiedEvaluationError(
                "real repeated execution requires --confirm-full-run"
            )
        episodes: list[dict[str, Any]] = []
        primary_child: Path | None = None
        for repeat_id in repeat_ids:
            child = _repeat_output_dir(
                target,
                repeat_id,
                preserve_single_layout=preserve_single_layout,
            )
            tail = _nl_tail(
                args,
                config_path=config_path,
                output_dir=child,
                budget_value=(
                    budget.value if budget.source != "official_profile" else None
                ),
                budget_profile=budget.profile_id,
                mode=args.mode,
                model=model,
                run=run,
            )
            code = command(tail)
            if code != 0:
                raise UnifiedEvaluationError(
                    f"Tool-Free backend failed for repeat {repeat_id}"
                )
            _annotate_repeat_metadata(
                child,
                repeat_id=repeat_id,
                repeat_count=args.repeats,
                model_key=args.model if official_protocol is not None else None,
                protocol_override=official_protocol,
            )
            if primary_child is None:
                primary_child = child
            episodes.append(
                {
                    "repeat_id": repeat_id,
                    "status": "dry_run" if args.dry_run else "complete",
                }
            )
        assert primary_child is not None
        _publish_primary_repeat(target, primary_child)
        summary = {
            "schema_version": "1.0",
            "status": "complete" if not args.dry_run else "dry_run",
            "setting": "tool_free",
            "mode": args.mode,
            "budget": budget.to_dict(),
            "protocol": official_protocol,
            "repeat_count": args.repeats,
            "executed_repeat_count": len(repeat_ids),
            "episodes": episodes,
        }
    (target / "evaluation_summary.json").write_text(
        json.dumps(
            summary, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _agentic_provider(args: argparse.Namespace, domain: str):
    if args.provider == "mock":
        return MockAgenticProvider(default_mock_responses(domain))
    if args.provider == "replay":
        if not args.replay_file:
            raise UnifiedEvaluationError("replay provider requires --replay-file")
        return ReplayAgenticProvider(Path(args.replay_file))
    return None


def _annotate_agentic_summary(
    path: Path,
    budget: ResolvedBudget,
    *,
    repeat_id: int,
    repeat_count: int,
    budget_level: int | None,
) -> None:
    summary_path = path / "backend_summary.json"
    value = json.loads(summary_path.read_text(encoding="utf-8"))
    value["budget_resolution"] = budget.to_dict()
    value["repeat_id"] = repeat_id
    value["repeat_count"] = repeat_count
    value["execution_id"] = uuid4().hex
    if budget_level is not None:
        value["budget_level"] = budget_level
    summary_path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _run_agentic(args: argparse.Namespace) -> dict[str, Any]:
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    data = resolve_path(_data_source(args.domain, args.data))
    repeat_ids = _repeat_ids(args)
    preserve_single_layout = args.repeats == 1 and args.repeat_id is None
    if args.mode == "response_curve":
        grid = resolve_budget_grid(
            setting="agentic",
            explicit_values=args.budget_grid,
            profile_id=args.budget_profile,
            domain=args.domain,
            model_key=args.model,
        )
        tasks = export_agentic_response_curve_tasks(
            domain=args.domain,
            data_source=data,
            output_dir=target / "tasks",
            budgets=grid,
            repeat_ids=repeat_ids,
            limit_problems=args.limit_problems,
            confirm_full_curve=args.confirm_full_run or args.provider != "real",
            strict_data=not args.toy,
        )
        curve_condition_kind = (
            "official_profile"
            if args.budget_profile and not args.budget_grid
            else "custom"
        )
    else:
        budget = resolve_budget(
            setting="agentic",
            explicit_value=args.budget,
            config_value=None,
            profile_id=args.budget_profile,
            condition_id=args.condition_id,
            domain=args.domain,
            model_key=args.model,
        )
        if args.mode == "contest":
            tasks = export_agentic_tasks(
                domain=args.domain,
                data_source=data,
                output_dir=target / "tasks",
                budget=budget.value,
                limit_suites=args.limit_suites,
                confirm_full_export=args.confirm_full_run,
                strict_data=not args.toy,
            )
        else:
            tasks = export_agentic_response_curve_tasks(
                domain=args.domain,
                data_source=data,
                output_dir=target / "tasks",
                budgets=(budget.value,),
                repeat_ids=repeat_ids,
                limit_problems=args.limit_problems,
                confirm_full_curve=args.confirm_full_run or args.provider != "real",
                strict_data=not args.toy,
            )
    if args.provider == "real":
        execution_profile = resolve_agentic_execution_profile(
            args.model, args.model_profiles
        )
        if not args.agentic_backend_config:
            raise UnifiedEvaluationError(
                "real Agentic execution requires --agentic-backend-config"
            )
        backend_config = load_external_backend_config(args.agentic_backend_config)
        if args.dry_run:
            readiness = check_external_backend_readiness(backend_config, probe=False)
            summary = {
                "schema_version": "1.0",
                "status": "dry_run",
                "setting": "agentic",
                "mode": args.mode,
                "budget_unit": "counted_actions",
                "exported_task_count": (
                    len(tasks) * len(repeat_ids)
                    if args.mode == "contest"
                    else len(tasks)
                ),
                "repeat_count": args.repeats,
                "executed_repeat_count": len(repeat_ids),
                "backend_readiness": readiness,
                "execution_profile": execution_profile.to_dict(),
                "external_backend_called": False,
            }
            (target / "evaluation_summary.json").write_text(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return summary
        if not args.allow_real_api or not args.allow_agentic_backend:
            raise UnifiedEvaluationError(
                "real Agentic execution requires --allow-real-api and "
                "--allow-agentic-backend"
            )
        execution_count = (
            len(tasks) * len(repeat_ids) if args.mode == "contest" else len(tasks)
        )
        if execution_count > 1 and not args.confirm_full_run:
            raise UnifiedEvaluationError(
                "real Agentic execution of multiple tasks requires --confirm-full-run"
            )
    else:
        backend_config = None
        execution_profile = None
    episodes: list[dict[str, Any]] = []
    executions = (
        tuple((task, repeat_id) for task in tasks for repeat_id in repeat_ids)
        if args.mode == "contest"
        else tuple((task, task.repeat_id) for task in tasks)
    )
    for task, repeat_id in executions:
        episode_parent = target / "episodes"
        if args.mode == "contest":
            episode_parent = _repeat_output_dir(
                episode_parent,
                repeat_id,
                preserve_single_layout=preserve_single_layout,
            )
        episode = episode_parent / task.task_dir.name
        if args.mode == "response_curve":
            base_resolution = resolve_budget(
                setting="agentic",
                explicit_value=task.counted_action_budget,
                config_value=None,
                profile_id=None,
                condition_id=args.condition_id
                or (args.budget_profile or "response_curve"),
                domain=args.domain,
                model_key=args.model,
            )
            resolution = ResolvedBudget(
                value=int(task.counted_action_budget or 0),
                unit="counted_actions",
                source=(
                    "official_profile"
                    if curve_condition_kind == "official_profile"
                    else "response_curve_grid"
                ),
                condition_id=base_resolution.condition_id,
                condition_kind=curve_condition_kind,
                profile_id=(
                    args.budget_profile
                    if curve_condition_kind == "official_profile"
                    else None
                ),
                rho=None,
            )
        else:
            resolution = budget
        if args.provider == "real":
            assert backend_config is not None
            assert execution_profile is not None
            run_external_agentic_backend(
                task_dir=task.task_dir,
                output_dir=episode,
                execution_profile=execution_profile,
                config=backend_config,
                allow_real_api=args.allow_real_api,
                allow_agentic_backend=args.allow_agentic_backend,
            )
        else:
            provider = _agentic_provider(args, args.domain)
            assert provider is not None
            run_agentic_episode(
                AgenticEpisodeConfig(
                    task_dir=task.task_dir,
                    output_dir=episode,
                    model_key=args.model,
                    max_turns=args.max_turns,
                ),
                provider,
            )
        _annotate_agentic_summary(
            episode,
            resolution,
            repeat_id=repeat_id,
            repeat_count=args.repeats,
            budget_level=task.budget_level,
        )
        write_agentic_saved_outputs(episode, episode / "saved_outputs.jsonl")
        episodes.append(
            {
                "task_id": task.task_id,
                "budget": task.counted_action_budget,
                "budget_level": task.budget_level,
                "repeat_id": repeat_id,
                "status": "complete",
                "condition_id": resolution.condition_id,
                "condition_kind": resolution.condition_kind,
            }
        )
    summary = {
        "schema_version": "1.0",
        "status": "complete",
        "setting": "agentic",
        "mode": args.mode,
        "budget_unit": "counted_actions",
        "episode_count": len(episodes),
        "repeat_count": args.repeats,
        "executed_repeat_count": len(repeat_ids),
        "episodes": episodes,
        "external_backend_called": args.provider == "real",
    }
    (target / "evaluation_summary.json").write_text(
        json.dumps(
            summary, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _budget_grid(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("budget grid must contain integers") from exc
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError(
            "budget grid must contain non-negative integers"
        )
    if tuple(sorted(result)) != result:
        raise argparse.ArgumentTypeError("budget grid must be nondecreasing")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", choices=("tool_free", "agentic"), required=True)
    parser.add_argument(
        "--domain", choices=("coding", "math", "abstract_reasoning"), required=True
    )
    parser.add_argument(
        "--mode", choices=("single_problem", "contest", "response_curve"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--output-token-budget", type=int)
    parser.add_argument("--counted-action-budget", type=int)
    parser.add_argument("--budget-grid", type=_budget_grid)
    parser.add_argument("--budget-profile")
    parser.add_argument("--condition-id")
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Independent repetitions to execute (paper protocol default: 5).",
    )
    parser.add_argument(
        "--repeat-id",
        type=int,
        help="Execute only this positive repeat ID for distributed orchestration.",
    )
    parser.add_argument(
        "--provider", choices=("mock", "replay", "real"), default="mock"
    )
    parser.add_argument("--replay-file")
    parser.add_argument("--visibility", choices=("hidden", "labeled"), default="hidden")
    parser.add_argument(
        "--protocol", choices=("one_stage", "two_stage"), default="one_stage"
    )
    parser.add_argument("--two-stage-profile")
    parser.add_argument(
        "--two-stage-profiles", default="configs/two_stage_profiles.yaml"
    )
    parser.add_argument("--model-profiles", default="configs/model_profiles.yaml")
    parser.add_argument(
        "--evaluator-profiles", default="configs/evaluator_profiles.yaml"
    )
    parser.add_argument("--run-profiles", default="configs/run_profiles.yaml")
    parser.add_argument("--run-profile")
    parser.add_argument("--provider-profile")
    parser.add_argument("--agentic-backend-config")
    parser.add_argument("--allow-real-api", action="store_true")
    parser.add_argument("--allow-agentic-backend", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--limit-problems", type=int, default=1)
    parser.add_argument("--limit-suites", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--confirm-full-run", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specific = (
        args.output_token_budget
        if args.setting == "tool_free"
        else args.counted_action_budget
    )
    wrong_specific = (
        args.counted_action_budget
        if args.setting == "tool_free"
        else args.output_token_budget
    )
    if wrong_specific is not None:
        print(
            "unified evaluation failed: budget flag does not match setting",
            file=sys.stderr,
        )
        return 2
    if args.budget is not None and specific is not None and args.budget != specific:
        print(
            "unified evaluation failed: conflicting explicit budgets", file=sys.stderr
        )
        return 2
    args.budget = specific if specific is not None else args.budget
    if args.setting == "agentic" and args.protocol != "one_stage":
        print(
            "unified evaluation failed: Agentic uses its native one-stage episode protocol",
            file=sys.stderr,
        )
        return 2
    if args.dry_run and args.provider != "real":
        print(
            "unified evaluation failed: --dry-run applies only to real providers",
            file=sys.stderr,
        )
        return 2
    if args.mode == "response_curve" and args.budget is not None:
        print(
            "unified evaluation failed: response_curve uses --budget-grid",
            file=sys.stderr,
        )
        return 2
    if args.mode != "response_curve" and args.budget_grid is not None:
        print(
            "unified evaluation failed: --budget-grid requires response_curve",
            file=sys.stderr,
        )
        return 2
    if (
        args.limit_problems <= 0
        or args.limit_suites <= 0
        or args.repeats <= 0
        or (args.repeat_id is not None and args.repeat_id <= 0)
        or (args.repeat_id is not None and args.repeat_id > args.repeats)
    ):
        print(
            "unified evaluation failed: limits, repeats, and repeat IDs must be positive",
            file=sys.stderr,
        )
        return 2
    try:
        summary = (
            _run_tool_free(args) if args.setting == "tool_free" else _run_agentic(args)
        )
    except (OSError, RuntimeError, ValueError, BudgetResolutionError) as exc:
        print(f"unified evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
