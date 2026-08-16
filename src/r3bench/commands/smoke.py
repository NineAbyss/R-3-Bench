"""Run bounded, network-free public evaluator acceptance checks."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Callable, Iterable

from r3bench import __version__
from r3bench.benchmark import expand_cells
from r3bench.agentic.external_backend import (
    check_external_backend_readiness,
    load_external_backend_config,
)
from r3bench.commands.analysis import main as analysis_main
from r3bench.commands.run_evaluation import main as run_evaluation_main
from r3bench.commands.verifier_check import check_readiness
from r3bench.common.loader import load_contest_suites, load_single_problems
from r3bench.common.scoring_dispatch import main as score_main
from r3bench.resource_paths import resource_path


class SmokeError(RuntimeError):
    """Raised when a bounded public smoke check fails."""


def _assert_command(
    name: str, command: Callable[[list[str]], int], argv: list[str]
) -> None:
    code = command(argv)
    if code != 0:
        raise SmokeError(f"{name} returned exit status {code}")


def _validate_examples() -> dict[str, dict[str, int]]:
    sources = {
        "coding": resource_path("examples", "data", "coding.jsonl"),
        "math": resource_path("examples", "data", "math"),
        "abstract_reasoning": resource_path(
            "examples", "data", "abstract_reasoning.jsonl"
        ),
    }
    result: dict[str, dict[str, int]] = {}
    for domain, source in sources.items():
        problems = load_single_problems(domain, "test", source, strict=False)
        suites = load_contest_suites(domain, "test", source, strict=False)
        if len(problems) != 6 or len(suites) != 1:
            raise SmokeError(f"{domain} toy data dimensions are invalid")
        result[domain] = {"problems": len(problems), "suites": len(suites)}
    return result


def run_smoke(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=False)
    checks: list[str] = []

    examples = _validate_examples()
    checks.append("example_data")
    cell_count = len(expand_cells())
    checks.append("profile_expansion")

    _assert_command(
        "Tool-Free single mock",
        run_evaluation_main,
        [
            "--setting",
            "tool_free",
            "--domain",
            "coding",
            "--mode",
            "single_problem",
            "--model",
            "local-mock",
            "--data",
            "examples/data/coding.jsonl",
            "--output-dir",
            str(output_dir / "tool_free_single"),
            "--output-token-budget",
            "2048",
            "--provider",
            "mock",
            "--toy",
            "--limit-problems",
            "1",
            "--repeats",
            "1",
        ],
    )
    checks.append("tool_free_single")
    _assert_command(
        "Tool-Free contest mock",
        run_evaluation_main,
        [
            "--setting",
            "tool_free",
            "--domain",
            "coding",
            "--mode",
            "contest",
            "--model",
            "local-mock",
            "--data",
            "examples/data/coding.jsonl",
            "--output-dir",
            str(output_dir / "tool_free_contest"),
            "--output-token-budget",
            "4096",
            "--provider",
            "mock",
            "--toy",
            "--limit-suites",
            "1",
            "--repeats",
            "1",
        ],
    )
    checks.append("tool_free_contest")
    _assert_command(
        "Tool-Free two-stage mock",
        run_evaluation_main,
        [
            "--setting",
            "tool_free",
            "--domain",
            "math",
            "--mode",
            "contest",
            "--model",
            "local-mock",
            "--data",
            "examples/data/math/problems.jsonl",
            "--output-dir",
            str(output_dir / "tool_free_two_stage"),
            "--output-token-budget",
            "4096",
            "--provider",
            "mock",
            "--protocol",
            "two_stage",
            "--toy",
            "--limit-suites",
            "1",
            "--repeats",
            "1",
        ],
    )
    checks.append("tool_free_two_stage")

    _assert_command(
        "Agentic contest mock",
        run_evaluation_main,
        [
            "--setting",
            "agentic",
            "--domain",
            "coding",
            "--mode",
            "contest",
            "--model",
            "local-mock",
            "--data",
            "examples/data/coding.jsonl",
            "--output-dir",
            str(output_dir / "agentic_contest"),
            "--counted-action-budget",
            "3",
            "--provider",
            "mock",
            "--toy",
            "--limit-suites",
            "1",
            "--repeats",
            "1",
        ],
    )
    checks.append("agentic_contest")

    _assert_command(
        "Agentic single-task mock",
        run_evaluation_main,
        [
            "--setting",
            "agentic",
            "--domain",
            "coding",
            "--mode",
            "single_problem",
            "--model",
            "local-mock",
            "--data",
            "examples/data/coding.jsonl",
            "--output-dir",
            str(output_dir / "agentic_single"),
            "--counted-action-budget",
            "2",
            "--provider",
            "mock",
            "--toy",
            "--limit-problems",
            "1",
            "--repeats",
            "1",
        ],
    )
    checks.append("agentic_single")

    episode_dirs = sorted((output_dir / "agentic_single" / "episodes").iterdir())
    if len(episode_dirs) != 1:
        raise SmokeError("Agentic single-task run did not produce one episode")
    agentic_episode = episode_dirs[0]
    _assert_command(
        "Agentic final-artifact mock scoring",
        score_main,
        [
            "--domain",
            "coding",
            "--data",
            str(resource_path("examples", "data", "coding.jsonl")),
            "--predictions",
            str(agentic_episode / "saved_outputs.jsonl"),
            "--output-dir",
            str(output_dir / "agentic_single_scoring"),
            "--scoring-mode",
            "mock",
            "--relaxed",
        ],
    )
    checks.append("agentic_final_artifact_scoring")
    _assert_command(
        "Agentic response-curve input",
        analysis_main,
        [
            "build-response-curve",
            "--setting",
            "agentic",
            "--domain",
            "coding",
            "--model",
            "local-mock",
            "--data",
            str(resource_path("examples", "data", "coding.jsonl")),
            "--run-dir",
            str(agentic_episode),
            "--scoring-dir",
            str(output_dir / "agentic_single_scoring"),
            "--budget",
            "2",
            "--condition-id",
            "toy_agentic_curve",
            "--relaxed",
            "--output",
            str(output_dir / "agentic_response_curve.jsonl"),
        ],
    )
    checks.append("agentic_response_curve_input")

    backend_readiness = check_external_backend_readiness(
        load_external_backend_config(
            resource_path("configs", "agentic", "external_backend.example.yaml")
        )
    )
    if (
        backend_readiness["status"] != "not_configured"
        or backend_readiness["external_process_started"] is not False
    ):
        raise SmokeError("external Agentic backend example did not fail closed")
    checks.append("agentic_external_backend_readiness")

    _assert_command(
        "Oracle toy demo",
        analysis_main,
        [
            "compare",
            "--response-curve",
            str(
                resource_path(
                    "examples", "inputs", "analysis", "response_curve_points.jsonl"
                )
            ),
            "--contest-results",
            str(
                resource_path("examples", "inputs", "analysis", "contest_results.jsonl")
            ),
            "--budgets",
            str(resource_path("examples", "inputs", "analysis", "budgets.json")),
            "--output-dir",
            str(output_dir / "oracle"),
        ],
    )
    checks.append("oracle")

    readiness = check_readiness(
        data_source=resource_path("examples", "data", "coding.jsonl"),
        config_path=resource_path("configs", "verifiers", "lightcpverifier.toy.yaml"),
    )
    if (
        readiness["status"] != "not_configured"
        or readiness["public_problem_count"] != 6
        or readiness["external_verifier_started"] is not False
    ):
        raise SmokeError("Coding verifier example did not fail closed")
    (output_dir / "verifier_readiness.json").write_text(
        json.dumps(
            {
                "status": readiness["status"],
                "public_problem_count": readiness["public_problem_count"],
                "external_verifier_started": False,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checks.append("coding_verifier_readiness")

    scoring_inputs = {
        "coding": (
            resource_path("examples", "data", "coding.jsonl"),
            resource_path(
                "examples", "inputs", "scoring", "coding_saved_outputs.jsonl"
            ),
        ),
        "math": (
            resource_path("examples", "data", "math"),
            resource_path("examples", "inputs", "scoring", "math_saved_outputs.jsonl"),
        ),
        "abstract_reasoning": (
            resource_path("examples", "data", "abstract_reasoning.jsonl"),
            resource_path(
                "examples",
                "inputs",
                "scoring",
                "abstract_reasoning_saved_outputs.jsonl",
            ),
        ),
    }
    for domain, (data, predictions) in scoring_inputs.items():
        _assert_command(
            f"{domain} mock scoring",
            score_main,
            [
                "--domain",
                domain,
                "--data",
                str(data),
                "--predictions",
                str(predictions),
                "--output-dir",
                str(output_dir / f"scoring_{domain}"),
                "--scoring-mode",
                "mock",
                "--relaxed",
            ],
        )
        checks.append(f"{domain}_mock_scoring")

    summary: dict[str, object] = {
        "schema_version": "1.0",
        "status": "pass",
        "package_version": __version__,
        "network_called": False,
        "model_api_called": False,
        "external_verifier_started": False,
        "container_runtime_started": False,
        "reference_cell_count": cell_count,
        "examples": examples,
        "checks": checks,
    }
    (output_dir / "acceptance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, prog="run_user_acceptance_smoke.py"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/user_acceptance_smoke"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the selected smoke output directory.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir == Path.cwd().resolve() or output_dir == output_dir.parent:
        print("smoke output must be a dedicated subdirectory", file=sys.stderr)
        return 2
    if output_dir.exists():
        if not args.overwrite:
            print(
                "smoke output already exists; use --overwrite to replace it",
                file=sys.stderr,
            )
            return 2
        shutil.rmtree(output_dir)
    try:
        summary = run_smoke(output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"user acceptance smoke failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "check_count": len(summary["checks"]),
                "network_called": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
