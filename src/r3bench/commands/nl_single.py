#!/usr/bin/env python3
"""Run or safely preview a shared pure-NL single-problem evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3bench.common.experiment import ExperimentConfig
from r3bench.common.budget import (
    BudgetResolutionError,
    annotate_run_summary,
    apply_budget_to_experiment,
    resolve_budget,
)
from r3bench.common.nl_checkpoint import NLCheckpointError
from r3bench.common.nl_cli_checkpoint import (
    add_checkpoint_arguments,
    finish_checkpoint_run,
    prepare_checkpoint_run,
)
from r3bench.common.nl_runner import (
    build_offline_provider,
    default_stage1_mock_response,
    derive_two_stage_configs,
    run_single_problem_nl,
    run_two_stage_nl,
)
from r3bench.common.provider_runtime import (
    resolve_real_provider_context,
    write_real_provider_dry_run,
    write_two_stage_real_provider_dry_run,
)
from r3bench.common.profile_registry import load_model_profiles
from r3bench.common.two_stage_profile import (
    apply_offline_two_stage_budget,
    derive_profiled_two_stage_configs,
    load_two_stage_profiles,
    resolve_two_stage_profile,
)
from r3bench.providers.errors import ProviderError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Public experiment YAML/JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--provider", choices=("mock", "replay", "real"), default="mock")
    parser.add_argument(
        "--replay-file",
        help="Allowlisted local response JSONL; required for --provider replay",
    )
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--provider-profile")
    parser.add_argument("--model-profiles")
    parser.add_argument("--evaluator-profiles")
    parser.add_argument("--run-profiles")
    parser.add_argument("--run-profile")
    parser.add_argument("--model-key")
    parser.add_argument(
        "--two-stage-profiles",
        default="configs/two_stage_profiles.yaml",
    )
    parser.add_argument("--two-stage-profile")
    parser.add_argument(
        "--output-token-budget",
        "--budget",
        dest="output_token_budget",
        type=int,
        help="Explicit output-token budget; overrides config and named profiles.",
    )
    parser.add_argument("--budget-profile")
    parser.add_argument("--condition-id")
    parser.add_argument("--rho", type=float, choices=(0.2, 0.8))
    parser.add_argument(
        "--thinking-enabled",
        choices=("true", "false"),
        help="Explicit one-stage provider thinking flag from formal provenance",
    )
    parser.add_argument("--allow-real-api", action="store_true")
    parser.add_argument(
        "--confirm-full-run",
        action="store_true",
        help="Required for real execution of more than one problem",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and preview a real request without reading a key or calling network",
    )
    add_checkpoint_arguments(parser)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig.from_file(args.config)
    try:
        budget = resolve_budget(
            setting="tool_free",
            explicit_value=args.output_token_budget,
            config_value=config.max_tokens,
            profile_id=args.budget_profile,
            condition_id=args.condition_id,
            domain=config.domain,
            model_key=args.model_key or config.model_name,
        )
        config = apply_budget_to_experiment(config, budget)
    except BudgetResolutionError as exc:
        print(f"budget resolution failed: {exc}", file=sys.stderr)
        return 2
    effective_rho = args.rho if args.rho is not None else budget.rho
    protocol_metadata = {
        "kind": "two_stage" if args.two_stage else "one_stage",
        "two_stage_profile": args.two_stage_profile,
        "official_rho": effective_rho,
    }
    try:
        checkpoint = prepare_checkpoint_run(config, args, limit=args.limit)
    except NLCheckpointError as exc:
        print(f"checkpoint setup failed: {exc}", file=sys.stderr)
        return 2
    checkpoint_kwargs = checkpoint.runner_kwargs() if checkpoint else {}
    if args.provider == "real":
        required = {
            "--provider-profile": args.provider_profile,
            "--model-profiles": args.model_profiles,
            "--evaluator-profiles": args.evaluator_profiles,
            "--model-key": args.model_key,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            print(f"real provider requires: {', '.join(missing)}", file=sys.stderr)
            return 2
        try:
            if args.two_stage:
                if not args.two_stage_profile:
                    print(
                        "real two-stage execution requires --two-stage-profile",
                        file=sys.stderr,
                    )
                    return 2
                models = load_model_profiles(args.model_profiles)
                profile = resolve_two_stage_profile(
                    args.two_stage_profile,
                    load_two_stage_profiles(args.two_stage_profiles),
                    model_key=args.model_key,
                    domain=config.domain,
                )
                config_stage1, config_stage2 = derive_profiled_two_stage_configs(
                    config,
                    profile,
                    models,
                )
                context_stage1 = resolve_real_provider_context(
                    config_stage1,
                    provider_profile_path=args.provider_profile,
                    model_profiles_path=args.model_profiles,
                    evaluator_profiles_path=args.evaluator_profiles,
                    model_key=profile.stage1_model_key,
                    dry_run=args.dry_run,
                    run_profiles_path=args.run_profiles,
                    run_profile_id=args.run_profile,
                    thinking_enabled_override=profile.stage1_thinking_enabled,
                )
                context_stage2 = resolve_real_provider_context(
                    config_stage2,
                    provider_profile_path=args.provider_profile,
                    model_profiles_path=args.model_profiles,
                    evaluator_profiles_path=args.evaluator_profiles,
                    model_key=profile.stage2_model_key,
                    dry_run=args.dry_run,
                    run_profiles_path=args.run_profiles,
                    run_profile_id=args.run_profile,
                    thinking_enabled_override=profile.stage2_thinking_enabled,
                )
                if args.dry_run:
                    write_two_stage_real_provider_dry_run(
                        config_stage1,
                        config_stage2,
                        context_stage1,
                        context_stage2,
                        args.output_dir,
                        limit=args.limit,
                        protocol=profile.stage2_protocol,
                    )
                    print(
                        f"wrote a network-free two-stage preview to {args.output_dir}"
                    )
                    annotate_run_summary(
                        args.output_dir, budget, protocol=protocol_metadata
                    )
                    return 0
                if not args.allow_real_api:
                    print(
                        "real two-stage execution requires --allow-real-api",
                        file=sys.stderr,
                    )
                    return 2
                if args.limit is None or args.limit <= 0:
                    print(
                        "real two-stage execution requires a positive --limit",
                        file=sys.stderr,
                    )
                    return 2
                if args.limit > 1 and not args.confirm_full_run:
                    print(
                        "real execution of more than one problem requires "
                        "--confirm-full-run",
                        file=sys.stderr,
                    )
                    return 2
                artifacts = run_two_stage_nl(
                    config_stage1,
                    config_stage2,
                    context_stage1.adapter,
                    context_stage2.adapter,
                    limit=args.limit,
                    judge_mode="none",
                    protocol=profile.stage2_protocol,
                    **checkpoint_kwargs,
                )
            else:
                if not args.dry_run and not args.allow_real_api:
                    print("real execution requires --allow-real-api", file=sys.stderr)
                    return 2
                context = resolve_real_provider_context(
                    config,
                    provider_profile_path=args.provider_profile,
                    model_profiles_path=args.model_profiles,
                    evaluator_profiles_path=args.evaluator_profiles,
                    model_key=args.model_key,
                    dry_run=args.dry_run,
                    run_profiles_path=args.run_profiles,
                    run_profile_id=args.run_profile,
                    thinking_enabled_override=(
                        args.thinking_enabled == "true"
                        if args.thinking_enabled is not None
                        else None
                    ),
                )
                if args.dry_run:
                    write_real_provider_dry_run(
                        config,
                        context,
                        args.output_dir,
                        mode="single_problem",
                        limit=args.limit,
                    )
                    print(f"wrote a network-free request preview to {args.output_dir}")
                    annotate_run_summary(
                        args.output_dir, budget, protocol=protocol_metadata
                    )
                    return 0
                if args.limit is None:
                    print(
                        "real provider execution requires an explicit --limit",
                        file=sys.stderr,
                    )
                    return 2
                if args.limit <= 0:
                    print("--limit must be positive", file=sys.stderr)
                    return 2
                if args.limit > 1 and not args.confirm_full_run:
                    print(
                        "real execution of more than one problem requires "
                        "--confirm-full-run",
                        file=sys.stderr,
                    )
                    return 2
                artifacts = run_single_problem_nl(
                    config,
                    context.adapter,
                    limit=args.limit,
                    judge_mode="none",
                    **checkpoint_kwargs,
                )
        except (ProviderError, OSError, ValueError) as exc:
            print(f"provider setup or execution failed: {exc}", file=sys.stderr)
            return 2
    elif args.two_stage:
        if args.dry_run:
            print("--dry-run is reserved for --provider real", file=sys.stderr)
            return 2
        config_stage1, config_stage2 = derive_two_stage_configs(config)
        profile = None
        if args.two_stage_profile:
            if not args.model_key:
                print("offline two-stage profile requires --model-key", file=sys.stderr)
                return 2
            profile = resolve_two_stage_profile(
                args.two_stage_profile,
                load_two_stage_profiles(args.two_stage_profiles),
                model_key=args.model_key,
                domain=config.domain,
            )
        config_stage1, config_stage2, protocol = apply_offline_two_stage_budget(
            config_stage1,
            config_stage2,
            stage1_budget=budget.value,
            profile=profile,
        )
        provider_stage1 = build_offline_provider(
            config_stage1,
            args.provider,
            args.replay_file,
            mock_response=default_stage1_mock_response(config_stage1),
        )
        provider_stage2 = build_offline_provider(
            config_stage2,
            args.provider,
            args.replay_file,
        )
        artifacts = run_two_stage_nl(
            config_stage1,
            config_stage2,
            provider_stage1,
            provider_stage2,
            limit=args.limit,
            protocol=protocol,
            **checkpoint_kwargs,
        )
    else:
        if args.dry_run:
            print("--dry-run is reserved for --provider real", file=sys.stderr)
            return 2
        provider = build_offline_provider(config, args.provider, args.replay_file)
        artifacts = run_single_problem_nl(
            config, provider, limit=args.limit, **checkpoint_kwargs
        )
    summary = finish_checkpoint_run(checkpoint, artifacts, args.output_dir)
    annotate_run_summary(args.output_dir, budget, protocol=protocol_metadata)
    print(
        f"wrote {summary['attempt_count']} offline attempts and "
        f"{summary['problem_count']} problem results to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
