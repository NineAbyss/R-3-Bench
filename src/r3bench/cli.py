"""Unified command-line interface for the R3Bench public evaluator."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from r3bench.benchmark import expand_cells, load_benchmark
from r3bench.commands import (
    analysis,
    agentic_backend,
    run_evaluation,
    validate_assets,
    verifier_check,
)
from r3bench.common.config import load_config
from r3bench.common.budget import load_official_budget_profiles
from r3bench.common.data_source import resolve_public_data_source, sha256_file
from r3bench.common.loader import load_contest_suites, load_single_problems
from r3bench.common.profile_registry import (
    load_evaluator_profiles,
    load_model_profiles,
    load_run_profiles,
)
from r3bench.common.scorer_registry import load_scorer_profiles
from r3bench.common.scoring_dispatch import main as score_main
from r3bench.resource_paths import resource_path, resolve_path


COMMANDS = (
    "profiles",
    "budgets",
    "data",
    "run",
    "analysis",
    "score",
    "verifier",
    "agentic",
    "doctor",
)


def _budgets(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="r3bench budgets")
    parser.add_argument("action", choices=("list", "show"))
    parser.add_argument("profile_id", nargs="?")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        profiles = load_official_budget_profiles()
        if args.action == "show":
            if not args.profile_id:
                parser.error("show requires profile_id")
            try:
                selected = profiles[args.profile_id]
            except KeyError as exc:
                raise ValueError(f"unknown budget profile: {args.profile_id}") from exc
            payload: object = selected.__dict__ if hasattr(selected, "__dict__") else {
                "profile_id": selected.profile_id,
                "model_key": selected.model_key,
                "domain": selected.domain,
                "setting": selected.setting,
                "role": selected.role,
                "budget_unit": selected.budget_unit,
                "budget_value": selected.budget_value,
                "budget_grid": list(selected.budget_grid),
                "rho": selected.rho,
                "resource_policy": selected.resource_policy,
                "provider_safety_cap": selected.provider_safety_cap,
            }
        else:
            payload = {
                "status": "valid",
                "profile_count": len(profiles),
                "profiles": [
                    {
                        "profile_id": profile.profile_id,
                        "model_key": profile.model_key,
                        "domain": profile.domain,
                        "setting": profile.setting,
                        "role": profile.role,
                        "budget_unit": profile.budget_unit,
                        "budget_value": profile.budget_value,
                        "budget_grid": list(profile.budget_grid),
                    }
                    for profile in profiles.values()
                ],
            }
    except (OSError, ValueError) as exc:
        print(f"budget profile validation failed: {exc}", file=sys.stderr)
        return 2
    if args.json or args.action == "show":
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    else:
        assert isinstance(payload, Mapping)
        print(f"Official paper-reference budget profiles: {payload['profile_count']}")
        for row in payload["profiles"]:
            assert isinstance(row, Mapping)
            value = row["budget_value"]
            grid = row["budget_grid"]
            detail = f"budget={value}" if value is not None else f"grid={grid}"
            print(f"{row['profile_id']}: {detail}")
    return 0


def _help_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="r3bench",
        description="R3Bench protocol evaluator and analysis tools.",
    )
    parser.add_argument("--version", action="store_true")
    parser.add_argument("command", nargs="?", choices=COMMANDS)
    parser.epilog = (
        "Run 'r3bench <command> --help' for command-specific help. "
        "No command calls a model API unless run receives both "
        "--provider real and --allow-real-api."
    )
    return parser


def _profiles(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="r3bench profiles")
    parser.add_argument("action", choices=("list", "validate"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        benchmark = load_benchmark()
        cells = expand_cells(benchmark)
        supported_models = sorted(load_model_profiles())
        payload = {
            "status": "valid",
            "supported_models": supported_models,
            "reference_benchmark_models": list(benchmark["reference_models"]),
            "evaluator_profiles": sorted(load_evaluator_profiles()),
            "run_profiles": sorted(load_run_profiles()),
            "scorer_profiles": sorted(load_scorer_profiles()),
            "reference_cell_count": len(cells),
        }
    except (OSError, ValueError) as exc:
        print(f"profile validation failed: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    elif args.action == "list":
        print("Supported models: " + ", ".join(payload["supported_models"]))
        print(
            "Reference benchmark models: "
            + ", ".join(payload["reference_benchmark_models"])
        )
        print(f"Expanded reference cells: {payload['reference_cell_count']}")
    else:
        print(
            "profiles valid: "
            f"{len(payload['supported_models'])} supported models, "
            f"{len(payload['run_profiles'])} run profiles, "
            f"{payload['reference_cell_count']} reference cells"
        )
    return 0


def _data_validate(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="r3bench data validate")
    parser.add_argument(
        "--domain",
        required=True,
        choices=("coding", "math", "abstract_reasoning"),
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Validate bundled six-row example dimensions instead of canonical 300 rows.",
    )
    args = parser.parse_args(argv)
    try:
        problems = load_single_problems(
            args.domain, args.split, args.source, strict=not args.toy
        )
        suites = load_contest_suites(
            args.domain, args.split, args.source, strict=not args.toy
        )
        expected = (6, 1) if args.toy else (300, 50)
        if (len(problems), len(suites)) != expected:
            raise ValueError(
                f"expected {expected[0]} problems/{expected[1]} suites, "
                f"found {len(problems)}/{len(suites)}"
            )
        if any(len(suite.problems) != 6 for suite in suites):
            raise ValueError("every contest suite must contain six problems")
    except (OSError, ValueError) as exc:
        print(f"data validation failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "valid",
                "domain": args.domain,
                "problem_count": len(problems),
                "suite_count": len(suites),
                "strict": not args.toy,
            },
            sort_keys=True,
        )
    )
    return 0


def _data_fetch(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="r3bench data fetch")
    parser.add_argument("--manifest", default="configs/data_manifest.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = load_config(resolve_path(args.manifest))
        repo_id = manifest.get("repo_id")
        revision = manifest.get("revision")
        domains = manifest.get("domains")
        if (
            not isinstance(repo_id, str)
            or not repo_id
            or not isinstance(revision, str)
            or not revision
            or not isinstance(domains, Mapping)
        ):
            raise ValueError("public dataset manifest is incomplete")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for domain, raw in domains.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"{domain} dataset manifest entry is invalid")
            filename = raw["relative_path"]
            source = f"hf://{repo_id}@{revision}::{filename}"
            downloaded = resolve_public_data_source(
                domain, source, cache_dir=args.cache_dir
            )
            expected = raw.get("sha256")
            if not isinstance(expected, str) or sha256_file(downloaded) != expected:
                raise ValueError(f"{domain} dataset digest does not match manifest")
            shutil.copy2(downloaded, args.output_dir / filename)
    except (OSError, ValueError) as exc:
        print(f"data fetch failed: {exc}", file=sys.stderr)
        return 2
    print(f"downloaded and verified three public data files to {args.output_dir}")
    return 0


def _data(argv: Sequence[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: r3bench data {validate,fetch} ...")
        return 0
    if argv[0] == "validate":
        return _data_validate(argv[1:])
    if argv[0] == "fetch":
        return _data_fetch(argv[1:])
    print(f"unknown data action: {argv[0]}", file=sys.stderr)
    return 2


def _verifier(argv: Sequence[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: r3bench verifier {check,validate-assets} ...")
        return 0
    if argv[0] == "check":
        return verifier_check.main(argv[1:])
    if argv[0] == "validate-assets":
        return validate_assets.main(argv[1:])
    print(f"unknown verifier action: {argv[0]}", file=sys.stderr)
    return 2


def _agentic(argv: Sequence[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print("usage: r3bench agentic backend {check,run} ...")
        return 0
    if argv[0] == "backend":
        return agentic_backend.main(argv[1:])
    print(f"unknown agentic action: {argv[0]}", file=sys.stderr)
    return 2


def _doctor(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="r3bench doctor")
    parser.parse_args(argv)
    checks: dict[str, object] = {}
    try:
        checks["reference_cells"] = len(expand_cells())
        checks["supported_models"] = len(load_model_profiles())
        checks["toy_coding"] = len(
            load_single_problems(
                "coding",
                "test",
                resource_path("examples", "data", "coding.jsonl"),
                strict=False,
            )
        )
        checks["package_resources"] = True
        status = "ready"
    except (OSError, ValueError) as exc:
        checks["error"] = str(exc)
        status = "invalid"
    print(json.dumps({"status": status, "checks": checks}, sort_keys=True))
    return 0 if status == "ready" else 2


def main(argv: Iterable[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help"}:
        _help_parser().print_help()
        return 0
    if raw[0] == "--version":
        from r3bench import __version__

        print(__version__)
        return 0
    command, tail = raw[0], raw[1:]
    routes = {
        "profiles": _profiles,
        "budgets": _budgets,
        "data": _data,
        "run": lambda values: run_evaluation.main(values),
        "analysis": lambda values: analysis.main(values),
        "score": lambda values: score_main(values),
        "verifier": _verifier,
        "agentic": _agentic,
        "doctor": _doctor,
    }
    route = routes.get(command)
    if route is None:
        _help_parser().error(f"unknown command: {command}")
    return route(tail)


if __name__ == "__main__":
    raise SystemExit(main())
