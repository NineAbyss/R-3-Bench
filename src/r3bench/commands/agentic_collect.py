#!/usr/bin/env python3
"""Collect Agentic final artifacts into the standard saved-output JSONL."""

from __future__ import annotations

import argparse
import sys
from typing import Iterable

from r3bench.agentic.scoring_handoff import (
    AgenticScoringHandoffError,
    write_agentic_saved_outputs,
)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        count = write_agentic_saved_outputs(args.episode_dir, args.output)
    except (AgenticScoringHandoffError, OSError, ValueError) as exc:
        print(f"Agentic scoring handoff failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {count} standard saved-output row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
