"""Offline equal-allocation replay over observed binary response curves."""

from __future__ import annotations

from typing import Iterable, Sequence

from r3bench.common.schema import ContestSuite
from r3bench.oracle.build_items import validate_oracle_capacity
from r3bench.oracle.pipeline_io import (
    group_contest_results,
    index_formal_budgets,
)
from r3bench.oracle.response_curve_schema import (
    ContestProblemResult,
    EqualAllocationProblemResult,
    EqualAllocationSuiteResult,
    FormalBudgetRecord,
    OracleItem,
    OracleSchemaError,
)


ItemKey = tuple[str, str, str, str, str]


def _index_items(
    items: Iterable[OracleItem], *, require_nonempty: bool = True
) -> dict[ItemKey, OracleItem]:
    indexed: dict[ItemKey, OracleItem] = {}
    for item in items:
        key = (
            item.domain,
            item.model_key,
            item.setting,
            item.budget_unit,
            item.problem_id,
        )
        if key in indexed:
            raise OracleSchemaError(
                f"duplicate Oracle item for problem {item.problem_id!r}"
            )
        indexed[key] = item
    if require_nonempty and not indexed:
        raise OracleSchemaError("Oracle item collection is empty")
    return indexed


def _equal_problem(
    *,
    problem_id: str,
    problem_index: int,
    problem_label: str,
    item: OracleItem | None,
    allocated_budget: int,
) -> EqualAllocationProblemResult:
    selected = item is not None and item.observed_cost <= allocated_budget
    return EqualAllocationProblemResult(
        problem_id=problem_id,
        problem_index=problem_index,
        problem_label=problem_label,
        observed_cost=item.observed_cost if item is not None else None,
        allocated_budget=allocated_budget,
        reward=int(selected),
        selected_by_equal=selected,
        source_run_id=item.source_run_id if item is not None else None,
    )


def compute_equal_allocation_for_suite(
    suite: ContestSuite,
    oracle_items: Iterable[OracleItem],
    total_budget: int,
    *,
    rho: float | None = None,
) -> EqualAllocationSuiteResult:
    """Replay floor(total_budget / 6) for one canonical contest suite."""

    indexed = _index_items(oracle_items)
    profiles = {
        (item.domain, item.model_key, item.setting, item.budget_unit)
        for item in indexed.values()
    }
    if len(profiles) != 1:
        raise OracleSchemaError(
            "single-suite equal replay requires one model/domain/setting"
        )
    domain, model_key, setting, budget_unit = next(iter(profiles))
    if suite.domain != domain:
        raise OracleSchemaError("suite domain does not match Oracle item domain")
    formal_budget = validate_oracle_capacity(total_budget)
    allocated = formal_budget // 6
    rows = tuple(
        _equal_problem(
            problem_id=problem.problem_id,
            problem_index=problem.problem_index,
            problem_label=problem.problem_label or "",
            item=indexed.get(
                (domain, model_key, setting, budget_unit, problem.problem_id)
            ),
            allocated_budget=allocated,
        )
        for problem in suite.problems
    )
    return EqualAllocationSuiteResult(
        domain=domain,  # type: ignore[arg-type]
        model_key=model_key,
        setting=setting,  # type: ignore[arg-type]
        budget_unit=budget_unit,  # type: ignore[arg-type]
        mode="contest",
        suite_id=suite.suite_id,
        rho=rho,
        formal_contest_budget=formal_budget,
        per_problem_budget=allocated,
        equal_score=sum(row.reward for row in rows),
        problem_results=rows,
    )


def _compute_equal_suite_for_profile(
    suite: ContestSuite,
    indexed: dict[ItemKey, OracleItem],
    *,
    domain: str,
    model_key: str,
    setting: str,
    budget_unit: str,
    formal_budget: int,
    rho: float | None,
) -> EqualAllocationSuiteResult:
    if suite.domain != domain:
        raise OracleSchemaError("suite domain does not match budget domain")
    allocated = formal_budget // 6
    rows = tuple(
        _equal_problem(
            problem_id=problem.problem_id,
            problem_index=problem.problem_index,
            problem_label=problem.problem_label or "",
            item=indexed.get(
                (domain, model_key, setting, budget_unit, problem.problem_id)
            ),
            allocated_budget=allocated,
        )
        for problem in suite.problems
    )
    return EqualAllocationSuiteResult(
        domain=domain,  # type: ignore[arg-type]
        model_key=model_key,
        setting=setting,  # type: ignore[arg-type]
        budget_unit=budget_unit,  # type: ignore[arg-type]
        mode="contest",
        suite_id=suite.suite_id,
        rho=rho,
        formal_contest_budget=formal_budget,
        per_problem_budget=allocated,
        equal_score=sum(row.reward for row in rows),
        problem_results=rows,
    )


def compute_equal_allocation_for_contests(
    suites: Sequence[ContestSuite],
    oracle_items: Iterable[OracleItem],
    budgets: Iterable[FormalBudgetRecord],
) -> tuple[EqualAllocationSuiteResult, ...]:
    """Replay each formal cell budget over every supplied canonical suite."""

    items = tuple(oracle_items)
    if not suites:
        raise OracleSchemaError("equal replay requires at least one contest suite")
    results: list[EqualAllocationSuiteResult] = []
    for budget in sorted(
        tuple(budgets),
        key=lambda row: (row.model_key, row.domain, row.rho),
    ):
        matching = tuple(
            item
            for item in items
            if (
                item.domain,
                item.model_key,
                item.setting,
                item.budget_unit,
            )
            == (
                budget.domain,
                budget.model_key,
                budget.setting,
                budget.budget_unit,
            )
        )
        capacity = validate_oracle_capacity(
            budget.formal_contest_budget,
            budget.response_curve_grid,
        )
        indexed = _index_items(matching, require_nonempty=False)
        for suite in suites:
            if suite.domain != budget.domain:
                continue
            results.append(
                _compute_equal_suite_for_profile(
                    suite,
                    indexed,
                    domain=budget.domain,
                    model_key=budget.model_key,
                    setting=budget.setting,
                    budget_unit=budget.budget_unit,
                    formal_budget=capacity,
                    rho=budget.rho,
                )
            )
    if not results:
        raise OracleSchemaError("no equal-allocation results were produced")
    return tuple(results)


def compute_equal_allocation_from_results(
    contest_results: Iterable[ContestProblemResult],
    oracle_items: Iterable[OracleItem],
    budgets: Iterable[FormalBudgetRecord],
) -> tuple[EqualAllocationSuiteResult, ...]:
    """Replay equal allocation using contest rows as suite membership metadata."""

    grouped = group_contest_results(contest_results)
    budget_index = index_formal_budgets(budgets)
    items = _index_items(oracle_items, require_nonempty=False)
    results: list[EqualAllocationSuiteResult] = []
    for key, rows in grouped.items():
        domain, model_key, setting, rho, suite_id = key
        budget_key = (domain, model_key, setting, rho)
        if budget_key not in budget_index:
            raise OracleSchemaError(f"missing formal budget for cell {budget_key!r}")
        budget = budget_index[budget_key]
        if {row.budget_unit for row in rows} != {budget.budget_unit}:
            raise OracleSchemaError(
                f"contest suite {suite_id!r} does not match budget_unit"
            )
        capacity = validate_oracle_capacity(
            budget.formal_contest_budget,
            budget.response_curve_grid,
        )
        if {row.formal_contest_budget for row in rows} != {capacity}:
            raise OracleSchemaError(
                f"contest suite {suite_id!r} does not record the formal capacity"
            )
        allocated = capacity // 6
        problem_results: list[EqualAllocationProblemResult] = []
        for row in rows:
            item = items.get(
                (domain, model_key, setting, row.budget_unit, row.problem_id)
            )
            if item is not None and (
                item.suite_id != suite_id
                or item.problem_index != row.problem_index
                or item.problem_label != row.problem_label
            ):
                raise OracleSchemaError(
                    f"Oracle item identity mismatch for {row.problem_id!r}"
                )
            problem_results.append(
                _equal_problem(
                    problem_id=row.problem_id,
                    problem_index=row.problem_index,
                    problem_label=row.problem_label,
                    item=item,
                    allocated_budget=allocated,
                )
            )
        results.append(
            EqualAllocationSuiteResult(
                domain=domain,  # type: ignore[arg-type]
                model_key=model_key,
                setting=setting,  # type: ignore[arg-type]
                budget_unit=rows[0].budget_unit,
                mode="contest",
                suite_id=suite_id,
                rho=rho,
                formal_contest_budget=capacity,
                per_problem_budget=allocated,
                equal_score=sum(row.reward for row in problem_results),
                problem_results=tuple(problem_results),
            )
        )
    return tuple(results)


__all__ = [
    "compute_equal_allocation_for_contests",
    "compute_equal_allocation_for_suite",
    "compute_equal_allocation_from_results",
]
