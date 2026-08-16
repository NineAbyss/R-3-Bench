"""Deterministic generic knapsack and exact six-problem R3Bench Oracle."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import isclose, isfinite
from typing import Iterable, Sequence

from r3bench.oracle.build_items import validate_oracle_capacity
from r3bench.oracle.pipeline_io import (
    group_contest_results,
    index_formal_budgets,
)
from r3bench.oracle.response_curve_schema import (
    ContestProblemResult,
    FormalBudgetRecord,
    OracleItem,
    OracleProblemSelection,
    OracleSchemaError,
    OracleSuiteResult,
)


@dataclass(frozen=True, slots=True)
class KnapsackItem:
    key: str
    cost: int
    value: float

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("item key must be non-empty")
        if (
            isinstance(self.cost, bool)
            or not isinstance(self.cost, int)
            or self.cost < 0
        ):
            raise ValueError("item cost must be a non-negative integer")
        if not isfinite(self.value):
            raise ValueError("item value must be finite")


@dataclass(frozen=True, slots=True)
class KnapsackResult:
    total_value: float
    total_cost: int
    selected_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MultipleChoiceKnapsackResult:
    total_value: float
    total_cost: int
    selected_keys: tuple[str, ...]
    combination_count: int


def _better(
    candidate: tuple[float, int, tuple[str, ...]],
    incumbent: tuple[float, int, tuple[str, ...]],
) -> bool:
    if not isclose(candidate[0], incumbent[0], rel_tol=1e-12, abs_tol=1e-12):
        return candidate[0] > incumbent[0]
    if candidate[1] != incumbent[1]:
        return candidate[1] < incumbent[1]
    return candidate[2] < incumbent[2]


def solve_knapsack(items: Iterable[KnapsackItem], budget: int) -> KnapsackResult:
    """Maximize value under an integer budget with deterministic tie-breaking."""

    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a non-negative integer")
    ordered = sorted(tuple(items), key=lambda item: item.key)
    keys = [item.key for item in ordered]
    if len(keys) != len(set(keys)):
        raise ValueError("item keys must be unique")

    states: list[tuple[float, int, tuple[str, ...]]] = [
        (0.0, 0, ()) for _ in range(budget + 1)
    ]
    for item in ordered:
        if item.cost > budget:
            continue
        for capacity in range(budget, item.cost - 1, -1):
            prior = states[capacity - item.cost]
            candidate = (
                prior[0] + item.value,
                prior[1] + item.cost,
                prior[2] + (item.key,),
            )
            if _better(candidate, states[capacity]):
                states[capacity] = candidate
    best = states[budget]
    return KnapsackResult(
        total_value=best[0],
        total_cost=best[1],
        selected_keys=best[2],
    )


def solve_multiple_choice_knapsack(
    groups: Iterable[Sequence[KnapsackItem]],
    budget: int,
) -> MultipleChoiceKnapsackResult:
    """Choose exactly one option per group under the configured-cost budget."""

    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a non-negative integer")
    choices = tuple(tuple(sorted(group, key=lambda item: item.key)) for group in groups)
    if not choices or any(not group for group in choices):
        raise ValueError("multiple-choice knapsack requires non-empty groups")
    keys = [item.key for group in choices for item in group]
    if len(keys) != len(set(keys)):
        raise ValueError("multiple-choice item keys must be globally unique")
    best: tuple[float, int, tuple[str, ...]] | None = None
    combination_count = 0
    for selected in product(*choices):
        combination_count += 1
        cost = sum(item.cost for item in selected)
        if cost > budget:
            continue
        candidate = (
            sum(item.value for item in selected),
            cost,
            tuple(item.key for item in selected),
        )
        if best is None or _better(candidate, best):
            best = candidate
    if best is None:
        raise ValueError("multiple-choice knapsack has no feasible combination")
    return MultipleChoiceKnapsackResult(
        total_value=best[0],
        total_cost=best[1],
        selected_keys=best[2],
        combination_count=combination_count,
    )


def solve_six_problem_oracle(
    contest_rows: Sequence[ContestProblemResult],
    oracle_items: Iterable[OracleItem],
    total_budget: int,
    *,
    rho: float | None = None,
) -> OracleSuiteResult:
    """Enumerate all 64 subsets with score/cost/input-order tie-breaking."""

    rows = tuple(sorted(contest_rows, key=lambda row: row.problem_index))
    if len(rows) != 6 or tuple(row.problem_index for row in rows) != (
        1,
        2,
        3,
        4,
        5,
        6,
    ):
        raise OracleSchemaError(
            "exact contest Oracle requires six rows at positions 1 through 6"
        )
    if len({row.problem_id for row in rows}) != 6:
        raise OracleSchemaError("contest problem IDs must be unique")
    profile = {
        (
            row.domain,
            row.model_key,
            row.setting,
            row.budget_unit,
            row.suite_id,
        )
        for row in rows
    }
    if len(profile) != 1:
        raise OracleSchemaError("contest rows must describe one suite and profile")
    domain, model_key, setting, budget_unit, suite_id = next(iter(profile))
    capacity = validate_oracle_capacity(total_budget)

    by_problem: dict[str, OracleItem] = {}
    for item in oracle_items:
        if (
            item.domain,
            item.model_key,
            item.setting,
            item.budget_unit,
        ) != (domain, model_key, setting, budget_unit):
            continue
        if item.problem_id in by_problem:
            raise OracleSchemaError(
                f"duplicate Oracle item for problem {item.problem_id!r}"
            )
        by_problem[item.problem_id] = item

    aligned: list[OracleItem | None] = []
    for row in rows:
        item = by_problem.get(row.problem_id)
        if item is not None and (
            item.suite_id != suite_id
            or item.problem_index != row.problem_index
            or item.problem_label != row.problem_label
        ):
            raise OracleSchemaError(
                f"Oracle item identity mismatch for {row.problem_id!r}"
            )
        aligned.append(item)

    best_score = -1
    best_cost = 0
    best_positions: tuple[int, ...] = ()
    best_mask = 0
    for mask in range(1 << 6):
        selected_positions: list[int] = []
        cost = 0
        valid = True
        for index, item in enumerate(aligned):
            if not mask & (1 << index):
                continue
            if item is None:
                valid = False
                break
            cost += item.observed_cost
            selected_positions.append(index + 1)
        if not valid or cost > capacity:
            continue
        score = len(selected_positions)
        positions = tuple(selected_positions)
        if (
            score > best_score
            or (score == best_score and cost < best_cost)
            or (
                score == best_score and cost == best_cost and positions < best_positions
            )
        ):
            best_score = score
            best_cost = cost
            best_positions = positions
            best_mask = mask

    selections = tuple(
        OracleProblemSelection(
            problem_id=row.problem_id,
            problem_index=row.problem_index,
            problem_label=row.problem_label,
            observed_cost=item.observed_cost if item is not None else None,
            reward=int(bool(best_mask & (1 << index))),
            selected_by_oracle=bool(best_mask & (1 << index)),
            source_run_id=item.source_run_id if item is not None else None,
        )
        for index, (row, item) in enumerate(zip(rows, aligned, strict=True))
    )
    result_rho = rho if rho is not None else rows[0].rho
    return OracleSuiteResult(
        domain=domain,  # type: ignore[arg-type]
        model_key=model_key,
        setting=setting,  # type: ignore[arg-type]
        budget_unit=budget_unit,  # type: ignore[arg-type]
        mode="contest",
        suite_id=suite_id,
        rho=result_rho,
        formal_contest_budget=capacity,
        oracle_score=best_score,
        total_selected_cost=best_cost,
        problem_selections=selections,
    )


def compute_oracle_from_results(
    contest_results: Iterable[ContestProblemResult],
    oracle_items: Iterable[OracleItem],
    budgets: Iterable[FormalBudgetRecord],
) -> tuple[OracleSuiteResult, ...]:
    """Compute exact Oracle selections for every contest suite and formal cell."""

    grouped = group_contest_results(contest_results)
    budget_index = index_formal_budgets(budgets)
    items = tuple(oracle_items)
    results: list[OracleSuiteResult] = []
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
        results.append(
            solve_six_problem_oracle(
                rows,
                items,
                capacity,
                rho=rho,
            )
        )
    return tuple(results)


__all__ = [
    "KnapsackItem",
    "KnapsackResult",
    "MultipleChoiceKnapsackResult",
    "compute_oracle_from_results",
    "solve_knapsack",
    "solve_multiple_choice_knapsack",
    "solve_six_problem_oracle",
]
