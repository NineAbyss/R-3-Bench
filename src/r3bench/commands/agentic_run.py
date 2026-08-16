#!/usr/bin/env python3
"""Run one bounded public Agentic episode with mock or replay tool calls."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from r3bench.agentic.runtime import (
    AgenticEpisodeConfig,
    AgenticRuntimeError,
    MockAgenticProvider,
    ReplayAgenticProvider,
    default_mock_responses,
    run_agentic_episode,
)


def _resolve_task_dir(path: Path) -> Path:
    if (path / "task_config.json").is_file():
        return path
    children = tuple(
        child
        for child in sorted(path.iterdir())
        if child.is_dir() and (child / "task_config.json").is_file()
    )
    if len(children) != 1:
        raise AgenticRuntimeError(
            "task-dir must be one task or contain exactly one exported task"
        )
    return children[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="synthetic-agent")
    parser.add_argument("--provider", choices=("mock", "replay"), default="mock")
    parser.add_argument("--replay-file")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--protocol-failure-limit", type=int, default=3)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.provider == "replay" and not args.replay_file:
        print("agentic run failed: replay provider requires --replay-file", file=sys.stderr)
        return 2
    try:
        task_dir = _resolve_task_dir(Path(args.task_dir))
        task = json.loads((task_dir / "task_config.json").read_text(encoding="utf-8"))
        if not isinstance(task, dict) or not isinstance(task.get("domain"), str):
            raise AgenticRuntimeError("task_config.json is malformed")
        provider = (
            ReplayAgenticProvider(Path(args.replay_file))
            if args.provider == "replay"
            else MockAgenticProvider(default_mock_responses(task["domain"]))
        )
        result = run_agentic_episode(
            AgenticEpisodeConfig(
                task_dir=task_dir,
                output_dir=Path(args.output_dir),
                model_key=args.model,
                max_turns=args.max_turns,
                protocol_failure_limit=args.protocol_failure_limit,
            ),
            provider,
        )
    except (AgenticRuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"agentic run failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(result), ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
