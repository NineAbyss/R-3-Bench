#!/usr/bin/env python3
"""Export sanitized public Agentic tasks from canonical contest data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3bench.agentic.task_export import (
    AgenticTaskExportError,
    export_agentic_response_curve_tasks,
    export_agentic_tasks,
)
from r3bench.resource_paths import resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain", required=True, choices=("coding", "math", "abstract_reasoning")
    )
    parser.add_argument(
        "--mode", choices=("contest", "response_curve"), default="contest"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budget", type=int)
    parser.add_argument(
        "--budget-grid",
        help="Comma-separated counted-action budgets for response_curve mode.",
    )
    parser.add_argument(
        "--budget-mode",
        choices=("counted_cap", "unbounded_calibration"),
        default="counted_cap",
        help="Use unbounded_calibration only for the large-cap baseline.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit-suites", type=int, default=1)
    parser.add_argument("--limit-problems", type=int, default=1)
    parser.add_argument(
        "--start-suite-number",
        type=int,
        default=1,
        help="One-based index in canonical loader order.",
    )
    parser.add_argument("--all-suites", action="store_true")
    parser.add_argument("--confirm-full-export", action="store_true")
    parser.add_argument("--confirm-full-curve", action="store_true")
    parser.add_argument("--strict-data", action="store_true")
    parser.add_argument(
        "--policy",
        choices=("compute_tools",),
        help="Formal exports use the paper's compute_tools policy in every domain.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "response_curve":
        try:
            if args.budget_grid:
                budgets = tuple(
                    int(item.strip())
                    for item in args.budget_grid.split(",")
                    if item.strip()
                )
            elif args.budget is not None:
                budgets = (args.budget,)
            else:
                raise AgenticTaskExportError(
                    "response_curve requires --budget or --budget-grid"
                )
            tasks = export_agentic_response_curve_tasks(
                domain=args.domain,
                data_source=resolve_path(args.data),
                output_dir=args.output_dir,
                budgets=budgets,
                split=args.split,
                limit_problems=args.limit_problems,
                confirm_full_curve=args.confirm_full_curve,
                strict_data=args.strict_data,
                action_policy=args.policy,
            )
        except (AgenticTaskExportError, OSError, ValueError) as exc:
            print(f"agentic task export failed: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "status": "exported",
                    "mode": "response_curve",
                    "task_count": len(tasks),
                    "tasks": [
                        {
                            "task_id": task.task_id,
                            "suite_id": task.suite_id,
                            "domain": task.domain,
                            "problem_count": task.problem_count,
                            "counted_action_budget": task.counted_action_budget,
                            "budget_level": task.budget_level,
                            "repeat_id": task.repeat_id,
                            "execution_scope": task.execution_scope,
                        }
                        for task in tasks
                    ],
                },
                ensure_ascii=True,
            )
        )
        return 0
    if args.budget_mode == "counted_cap" and args.budget is None:
        print(
            "agentic task export failed: counted_cap requires --budget",
            file=sys.stderr,
        )
        return 2
    if args.budget_mode == "unbounded_calibration" and args.budget is not None:
        print(
            "agentic task export failed: unbounded_calibration forbids --budget",
            file=sys.stderr,
        )
        return 2
    limit = None if args.all_suites else args.limit_suites
    if args.start_suite_number < 1:
        print("agentic task export failed: start-suite-number must be positive", file=sys.stderr)
        return 2
    try:
        tasks = export_agentic_tasks(
            domain=args.domain,
            data_source=resolve_path(args.data),
            output_dir=args.output_dir,
            budget=args.budget,
            budget_mode=args.budget_mode,
            split=args.split,
            limit_suites=limit,
            suite_offset=args.start_suite_number - 1,
            confirm_full_export=args.confirm_full_export,
            strict_data=args.strict_data,
            action_policy=args.policy,
        )
    except (AgenticTaskExportError, OSError, ValueError) as exc:
        print(f"agentic task export failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "exported",
                "task_count": len(tasks),
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "suite_id": task.suite_id,
                        "domain": task.domain,
                        "problem_count": task.problem_count,
                        "counted_action_budget": task.counted_action_budget,
                        "budget_mode": task.budget_mode,
                        "execution_scope": task.execution_scope,
                        "task_fingerprint_sha256": task.task_fingerprint_sha256,
                    }
                    for task in tasks
                ],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
