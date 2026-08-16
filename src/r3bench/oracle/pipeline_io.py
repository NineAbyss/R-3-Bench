"""Strict JSON I/O for the public Pure-NL Oracle pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, TypeVar

from r3bench.common.io import read_json, read_jsonl
from r3bench.common.result_schema import to_public_dict
from r3bench.oracle.response_curve_schema import (
    ContestProblemResult,
    EqualAllocationSuiteResult,
    FormalBudgetRecord,
    OracleSchemaError,
    OracleSuiteResult,
)


T = TypeVar("T")
CellKey = tuple[str, str, str, float]
SuiteKey = tuple[str, str, str, float, str]


def load_contest_results(
    path: str | Path,
) -> tuple[ContestProblemResult, ...]:
    rows = tuple(ContestProblemResult.from_dict(row) for row in read_jsonl(path))
    if not rows:
        raise OracleSchemaError("contest result input is empty")
    seen: set[tuple[str, str, str, float, str, str]] = set()
    for row in rows:
        key = (
            row.domain,
            row.model_key,
            row.setting,
            row.rho,
            row.suite_id,
            row.problem_id,
        )
        if key in seen:
            raise OracleSchemaError(
                f"duplicate contest result for problem {row.problem_id!r}"
            )
        seen.add(key)
    return rows


def load_formal_budgets(path: str | Path) -> tuple[FormalBudgetRecord, ...]:
    value = read_json(path)
    if not isinstance(value, Mapping) or frozenset(value) != {
        "schema_version",
        "budgets",
    }:
        raise OracleSchemaError(
            "formal budget file requires exactly schema_version and budgets"
        )
    if value["schema_version"] != "2.0":
        raise OracleSchemaError("unsupported formal budget schema_version")
    rows = value["budgets"]
    if not isinstance(rows, list) or not rows:
        raise OracleSchemaError("formal budget file requires a non-empty budgets array")
    records = tuple(FormalBudgetRecord.from_dict(row) for row in rows)
    keys = [(row.domain, row.model_key, row.setting, row.rho) for row in records]
    if len(keys) != len(set(keys)):
        raise OracleSchemaError("formal budget records must have unique cell keys")
    return records


def index_formal_budgets(
    records: Iterable[FormalBudgetRecord],
) -> dict[CellKey, FormalBudgetRecord]:
    indexed: dict[CellKey, FormalBudgetRecord] = {}
    for row in records:
        key = (row.domain, row.model_key, row.setting, row.rho)
        if key in indexed:
            raise OracleSchemaError(f"duplicate formal budget for cell {key!r}")
        indexed[key] = row
    if not indexed:
        raise OracleSchemaError("formal budget collection is empty")
    return indexed


def group_contest_results(
    rows: Iterable[ContestProblemResult],
) -> dict[SuiteKey, tuple[ContestProblemResult, ...]]:
    grouped: dict[SuiteKey, list[ContestProblemResult]] = {}
    for row in rows:
        key = (
            row.domain,
            row.model_key,
            row.setting,
            row.rho,
            row.suite_id,
        )
        grouped.setdefault(key, []).append(row)
    if not grouped:
        raise OracleSchemaError("contest result collection is empty")
    result: dict[SuiteKey, tuple[ContestProblemResult, ...]] = {}
    for key in sorted(grouped):
        ordered = tuple(sorted(grouped[key], key=lambda row: row.problem_index))
        if len(ordered) != 6:
            raise OracleSchemaError(
                f"contest suite {key[-1]!r} must contain exactly six results"
            )
        if tuple(row.problem_index for row in ordered) != (1, 2, 3, 4, 5, 6):
            raise OracleSchemaError(
                f"contest suite {key[-1]!r} must cover positions 1 through 6"
            )
        if len({row.problem_id for row in ordered}) != 6:
            raise OracleSchemaError(
                f"contest suite {key[-1]!r} must contain distinct problem IDs"
            )
        budgets = {row.formal_contest_budget for row in ordered}
        if len(budgets) != 1:
            raise OracleSchemaError(
                f"contest suite {key[-1]!r} has inconsistent formal budgets"
            )
        units = {row.budget_unit for row in ordered}
        if len(units) != 1:
            raise OracleSchemaError(
                f"contest suite {key[-1]!r} has inconsistent budget units"
            )
        result[key] = ordered
    return result


def load_equal_results(
    path: str | Path,
) -> tuple[EqualAllocationSuiteResult, ...]:
    return _load_result_document(path, "equal_replay", EqualAllocationSuiteResult)


def load_oracle_results(
    path: str | Path,
) -> tuple[OracleSuiteResult, ...]:
    return _load_result_document(path, "oracle_results", OracleSuiteResult)


def _load_result_document(
    path: str | Path,
    kind: str,
    record_type: type[T],
) -> tuple[T, ...]:
    value = read_json(path)
    if not isinstance(value, Mapping) or frozenset(value) != {
        "schema_version",
        "kind",
        "results",
    }:
        raise OracleSchemaError(
            f"{kind} file requires schema_version, kind, and results"
        )
    if value["schema_version"] != "2.0" or value["kind"] != kind:
        raise OracleSchemaError(f"invalid {kind} document metadata")
    rows = value["results"]
    if not isinstance(rows, list) or not rows:
        raise OracleSchemaError(f"{kind} results must be a non-empty array")
    parser = getattr(record_type, "from_dict")
    return tuple(parser(row) for row in rows)


def write_jsonl_records(path: str | Path, records: Iterable[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(
            json.dumps(
                to_public_dict(record),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def write_result_document(
    path: str | Path, *, kind: str, results: Iterable[Any]
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": "2.0",
        "kind": kind,
        "results": [to_public_dict(row) for row in results],
    }
    target.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "load_contest_results",
    "load_equal_results",
    "load_formal_budgets",
    "load_oracle_results",
    "group_contest_results",
    "index_formal_budgets",
    "write_jsonl_records",
    "write_result_document",
]
