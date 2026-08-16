"""Atomic item-level checkpoints for long Pure-NL generation runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

from r3bench.common.nl_runner import NLRunArtifacts
from r3bench.common.result_schema import to_public_dict


class NLCheckpointError(ValueError):
    """Raised when checkpoint state is incomplete or inconsistent."""


def select_checkpoint_items(
    item_ids: Iterable[str],
    *,
    include_ids: Iterable[str] | None = None,
    skip_ids: Iterable[str] = (),
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> tuple[str, ...]:
    """Select run units while retaining canonical loader order."""

    ordered = tuple(item_ids)
    if len(ordered) != len(set(ordered)):
        raise NLCheckpointError("source item IDs must be unique")
    include = set(include_ids) if include_ids is not None else set(ordered)
    unknown = include - set(ordered)
    if unknown:
        raise NLCheckpointError(f"unknown included item IDs: {sorted(unknown)[:10]}")
    skipped = set(skip_ids)
    unknown_skips = skipped - set(ordered)
    if unknown_skips:
        raise NLCheckpointError(
            f"unknown skipped item IDs: {sorted(unknown_skips)[:10]}"
        )
    if (shard_index is None) != (shard_count is None):
        raise NLCheckpointError("shard_index and shard_count must be provided together")
    if shard_count is not None:
        if shard_count <= 0:
            raise NLCheckpointError("shard_count must be positive")
        if shard_index is None or not 0 <= shard_index < shard_count:
            raise NLCheckpointError("shard_index must be in [0, shard_count)")
    selected = [
        item_id
        for index, item_id in enumerate(ordered)
        if item_id in include
        and item_id not in skipped
        and (shard_count is None or index % shard_count == shard_index)
    ]
    if not selected:
        raise NLCheckpointError("item selection is empty")
    return tuple(selected)


def _safe_name(item_id: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "-", item_id).strip("-") or "item"
    digest = hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:12]
    return f"{label[:80]}-{digest}.json"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


class NLCheckpointStore:
    """Persist sanitized per-problem/per-suite artifacts and materialize a run."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / ".nl_checkpoints"

    def _path(self, item_id: str) -> Path:
        return self.checkpoint_dir / _safe_name(item_id)

    def completed_item_ids(self) -> frozenset[str]:
        if not self.checkpoint_dir.is_dir():
            return frozenset()
        completed: set[str] = set()
        for path in self.checkpoint_dir.glob("*.json"):
            document = self._read(path)
            item_id = document.get("item_id")
            if not isinstance(item_id, str) or not item_id:
                raise NLCheckpointError(f"invalid item checkpoint: {path.name}")
            if item_id in completed:
                raise NLCheckpointError(f"duplicate checkpoint item: {item_id}")
            completed.add(item_id)
        return frozenset(completed)

    def record(self, item_id: str, artifacts: NLRunArtifacts) -> None:
        document = {
            "schema_version": "1.0",
            "item_id": item_id,
            "metadata": to_public_dict(artifacts.metadata),
            "attempts": [to_public_dict(row) for row in artifacts.attempts],
            "parsed_answers": [
                to_public_dict(row) for row in artifacts.parsed_answers
            ],
            "judge_results": [
                to_public_dict(row) for row in artifacts.judge_results
            ],
            "presentation_orders": [
                to_public_dict(row) for row in artifacts.presentation_orders
            ],
            "summary": to_public_dict(artifacts.summary),
        }
        _atomic_text(
            self._path(item_id),
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
        )

    def _read(self, path: Path) -> dict[str, object]:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NLCheckpointError(f"cannot read checkpoint {path.name}") from exc
        if (
            not isinstance(document, dict)
            or document.get("schema_version") != "1.0"
        ):
            raise NLCheckpointError(f"unsupported checkpoint: {path.name}")
        return document

    def materialize(self, item_ids: Iterable[str]) -> dict[str, object]:
        ordered = tuple(item_ids)
        if not ordered or len(ordered) != len(set(ordered)):
            raise NLCheckpointError("materialization requires unique item IDs")
        documents = []
        missing = []
        for item_id in ordered:
            path = self._path(item_id)
            if not path.is_file():
                missing.append(item_id)
                continue
            document = self._read(path)
            if document.get("item_id") != item_id:
                raise NLCheckpointError(f"checkpoint identity mismatch for {item_id}")
            documents.append(document)
        if missing:
            raise NLCheckpointError(
                f"checkpoint run is incomplete; missing {missing[:10]}"
            )

        collection_fields = (
            "attempts",
            "parsed_answers",
            "judge_results",
            "presentation_orders",
        )
        merged: dict[str, list[object]] = {field: [] for field in collection_fields}
        summaries: list[dict[str, object]] = []
        for document in documents:
            for field in collection_fields:
                value = document.get(field)
                if not isinstance(value, list):
                    raise NLCheckpointError(f"checkpoint {field} must be an array")
                merged[field].extend(value)
            summary = document.get("summary")
            if not isinstance(summary, dict):
                raise NLCheckpointError("checkpoint summary must be an object")
            summaries.append(summary)

        first = summaries[0]
        identity_fields = (
            "run_id",
            "domain",
            "mode",
            "visibility",
            "stage",
            "split",
            "model_name",
            "provider_name",
        )
        for summary in summaries[1:]:
            if any(summary.get(field) != first.get(field) for field in identity_fields):
                raise NLCheckpointError("checkpoint run identities do not match")
        count_fields = (
            "attempt_count",
            "problem_count",
            "parsed_count",
            "judged_count",
            "correct_count",
            "error_count",
        )
        summary = {field: first[field] for field in identity_fields}
        for field in count_fields:
            summary[field] = sum(int(row.get(field, 0)) for row in summaries)
        summary["total_score"] = sum(
            float(row.get("total_score", 0.0)) for row in summaries
        )
        summary["created_at"] = min(str(row["created_at"]) for row in summaries)
        summary["status"] = "complete"
        summary["checkpointed"] = True
        summary["completed_item_count"] = len(ordered)

        names = {
            "attempts": "attempts.jsonl",
            "parsed_answers": "parsed_answers.jsonl",
            "judge_results": "judge_results.jsonl",
            "presentation_orders": "presentation_orders.jsonl",
        }
        for field, filename in names.items():
            rows = merged[field]
            path = self.output_dir / filename
            if rows or field != "presentation_orders":
                _atomic_text(
                    path,
                    "".join(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                        )
                        + "\n"
                        for row in rows
                    ),
                )
            elif path.exists():
                path.unlink()
        _atomic_text(
            self.output_dir / "run_summary.json",
            json.dumps(
                summary,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
        )
        state = {
            "schema_version": "1.0",
            "status": "complete",
            "selected_item_ids": list(ordered),
            "completed_item_ids": list(ordered),
            "checkpoint_count": len(documents),
        }
        _atomic_text(
            self.output_dir / "checkpoint_state.json",
            json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
        return summary


__all__ = [
    "NLCheckpointError",
    "NLCheckpointStore",
    "select_checkpoint_items",
]
