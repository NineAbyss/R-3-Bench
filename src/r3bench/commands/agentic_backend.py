#!/usr/bin/env python3
"""Check or explicitly run the external Agentic backend contract."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Iterable

from r3bench.agentic.external_backend import (
    ExternalAgenticBackendError,
    check_external_backend_readiness,
    load_external_backend_config,
    resolve_agentic_execution_profile,
    run_external_agentic_backend,
)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "run"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--task-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--model")
    parser.add_argument(
        "--model-profiles", default="configs/model_profiles.yaml"
    )
    parser.add_argument("--limit-tasks", type=int, default=1)
    parser.add_argument("--allow-real-api", action="store_true")
    parser.add_argument("--allow-agentic-backend", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_external_backend_config(args.config)
        if args.action == "check":
            payload = check_external_backend_readiness(config, probe=args.probe)
            print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
            return 0 if payload["status"] in {"ready", "not_configured"} else 2
        if args.limit_tasks != 1:
            raise ExternalAgenticBackendError(
                "public external smoke is restricted to --limit-tasks 1"
            )
        if not args.task_dir or not args.output_dir or not args.model:
            raise ExternalAgenticBackendError(
                "run requires --task-dir, --output-dir, and --model"
            )
        execution_profile = resolve_agentic_execution_profile(
            args.model, args.model_profiles
        )
        payload = run_external_agentic_backend(
            task_dir=args.task_dir,
            output_dir=args.output_dir,
            execution_profile=execution_profile,
            config=config,
            allow_real_api=args.allow_real_api,
            allow_agentic_backend=args.allow_agentic_backend,
        )
        print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        return 0
    except (ExternalAgenticBackendError, OSError, ValueError) as exc:
        print(f"external Agentic backend failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
