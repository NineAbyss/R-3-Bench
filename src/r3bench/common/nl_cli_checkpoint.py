"""Shared CLI plumbing for resumable Pure-NL item execution."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r3bench.common.experiment import ExperimentConfig
from r3bench.common.nl_checkpoint import (
    NLCheckpointError,
    NLCheckpointStore,
    select_checkpoint_items,
)
from r3bench.common.nl_runner import NLRunArtifacts, list_nl_item_ids


@dataclass(frozen=True, slots=True)
class CheckpointRun:
    store: NLCheckpointStore
    selected_item_ids: tuple[str, ...]
    pending_item_ids: tuple[str, ...]

    def runner_kwargs(self) -> dict[str, Any]:
        return {
            "item_ids": self.pending_item_ids,
            "on_item_complete": self.store.record,
        }


def add_checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Persist one atomic checkpoint per problem or suite.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed item checkpoints and run only missing items.",
    )
    parser.add_argument("--include-ids", help="Comma-separated problem/suite IDs.")
    parser.add_argument("--skip-ids", help="Comma-separated problem/suite IDs.")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)


def _csv(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise NLCheckpointError("comma-separated item IDs cannot be empty")
    return result


def prepare_checkpoint_run(
    config: ExperimentConfig,
    args: argparse.Namespace,
    *,
    limit: int | None,
) -> CheckpointRun | None:
    requested = any(
        (
            args.checkpoint,
            args.resume,
            args.include_ids,
            args.skip_ids,
            args.shard_index is not None,
            args.shard_count is not None,
        )
    )
    if not requested:
        return None
    if not args.checkpoint:
        raise NLCheckpointError(
            "--resume, item filters, and sharding require --checkpoint"
        )
    if args.dry_run:
        raise NLCheckpointError("--checkpoint is not used with provider dry-run")
    store = NLCheckpointStore(args.output_dir)
    existing = store.completed_item_ids()
    if existing and not args.resume:
        raise NLCheckpointError(
            "item checkpoints already exist; use --resume or a new output directory"
        )
    selected = select_checkpoint_items(
        list_nl_item_ids(config, limit=limit),
        include_ids=_csv(args.include_ids),
        skip_ids=_csv(args.skip_ids) or (),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    unexpected = existing - set(selected)
    if unexpected:
        raise NLCheckpointError(
            "checkpoint directory contains items outside the current selection"
        )
    pending = tuple(item_id for item_id in selected if item_id not in existing)
    return CheckpointRun(store, selected, pending)


def finish_checkpoint_run(
    checkpoint: CheckpointRun | None,
    artifacts: NLRunArtifacts,
    output_dir: str | Path,
) -> dict[str, object]:
    if checkpoint is None:
        from r3bench.common.nl_runner import write_run_artifacts
        from r3bench.common.result_schema import to_public_dict

        write_run_artifacts(artifacts, output_dir)
        return to_public_dict(artifacts.summary)
    return checkpoint.store.materialize(checkpoint.selected_item_ids)


__all__ = [
    "CheckpointRun",
    "add_checkpoint_arguments",
    "finish_checkpoint_run",
    "prepare_checkpoint_run",
]
