"""Aggregate actual contest, equal replay, and exact Oracle suite scores."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from r3bench.oracle.pipeline_io import group_contest_results
from r3bench.oracle.response_curve_schema import (
    ContestProblemResult,
    EqualAllocationSuiteResult,
    GapSummary,
    OracleSchemaError,
    OracleSuiteResult,
)


CellKey = tuple[str, str, str, float]
SuiteKey = tuple[str, str, str, float, str]


def _equal_index(
    rows: Iterable[EqualAllocationSuiteResult],
) -> dict[SuiteKey, EqualAllocationSuiteResult]:
    result: dict[SuiteKey, EqualAllocationSuiteResult] = {}
    for row in rows:
        if row.rho is None:
            raise OracleSchemaError("aggregate equal results require rho")
        key = (row.domain, row.model_key, row.setting, row.rho, row.suite_id)
        if key in result:
            raise OracleSchemaError(f"duplicate equal replay suite {key!r}")
        result[key] = row
    return result


def _oracle_index(
    rows: Iterable[OracleSuiteResult],
) -> dict[SuiteKey, OracleSuiteResult]:
    result: dict[SuiteKey, OracleSuiteResult] = {}
    for row in rows:
        if row.rho is None:
            raise OracleSchemaError("aggregate Oracle results require rho")
        key = (row.domain, row.model_key, row.setting, row.rho, row.suite_id)
        if key in result:
            raise OracleSchemaError(f"duplicate Oracle suite {key!r}")
        result[key] = row
    return result


def aggregate_oracle_gap(
    contest_results: Iterable[ContestProblemResult],
    equal_results: Iterable[EqualAllocationSuiteResult],
    oracle_results: Iterable[OracleSuiteResult],
) -> tuple[GapSummary, ...]:
    """Compute mean correct-per-suite scores and resource-rationality gaps."""

    contest_groups = group_contest_results(contest_results)
    equal_index = _equal_index(equal_results)
    oracle_index = _oracle_index(oracle_results)
    contest_keys = set(contest_groups)
    if contest_keys != set(equal_index):
        raise OracleSchemaError(
            "contest and equal replay suite keys must match exactly"
        )
    if contest_keys != set(oracle_index):
        raise OracleSchemaError("contest and Oracle suite keys must match exactly")

    by_cell: dict[
        CellKey,
        list[
            tuple[
                tuple[ContestProblemResult, ...],
                EqualAllocationSuiteResult,
                OracleSuiteResult,
            ]
        ],
    ] = defaultdict(list)
    for suite_key in sorted(contest_keys):
        domain, model_key, setting, rho, _ = suite_key
        contest = contest_groups[suite_key]
        equal = equal_index[suite_key]
        oracle = oracle_index[suite_key]
        budgets = {
            contest[0].formal_contest_budget,
            equal.formal_contest_budget,
            oracle.formal_contest_budget,
        }
        if len(budgets) != 1:
            raise OracleSchemaError(
                f"suite {suite_key[-1]!r} has inconsistent formal capacities"
            )
        units = {
            contest[0].budget_unit,
            equal.budget_unit,
            oracle.budget_unit,
        }
        if len(units) != 1:
            raise OracleSchemaError(
                f"suite {suite_key[-1]!r} has inconsistent budget units"
            )
        by_cell[(domain, model_key, setting, rho)].append((contest, equal, oracle))

    summaries: list[GapSummary] = []
    for cell_key in sorted(by_cell):
        domain, model_key, setting, rho = cell_key
        rows = by_cell[cell_key]
        contest_total = sum(
            sum(problem.reward for problem in contest) for contest, _, _ in rows
        )
        equal_total = sum(equal.equal_score for _, equal, _ in rows)
        oracle_total = sum(oracle.oracle_score for _, _, oracle in rows)
        suite_count = len(rows)
        contest_score = contest_total / suite_count
        equal_score = equal_total / suite_count
        oracle_score = oracle_total / suite_count
        delta = oracle_score - contest_score
        gap_ratio = delta / oracle_score if oracle_score > 0 else None
        summaries.append(
            GapSummary(
                domain=domain,  # type: ignore[arg-type]
                model_key=model_key,
                setting=setting,  # type: ignore[arg-type]
                budget_unit=rows[0][0][0].budget_unit,
                mode="contest",
                rho=rho,
                formal_contest_budget=rows[0][0][0].formal_contest_budget,
                suite_count=suite_count,
                contest_total=contest_total,
                equal_total=equal_total,
                oracle_total=oracle_total,
                contest_score=contest_score,
                equal_score=equal_score,
                oracle_score=oracle_score,
                delta_rr=delta,
                gap_ratio=gap_ratio,
            )
        )
    return tuple(summaries)


__all__ = ["aggregate_oracle_gap"]
