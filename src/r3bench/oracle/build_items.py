"""Load response curves and build empirical budget options for the Oracle."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from r3bench.common.io import read_jsonl
from r3bench.common.schema import ContestSuite
from r3bench.oracle.response_curve_schema import (
    OracleItem,
    OracleBudgetOption,
    OracleSchemaError,
    ProblemResponseCurve,
    ResponseCurvePoint,
)


ProblemKey = tuple[str, str, str, str, str]
FORMAL_REPEAT_COUNT = 5
FORMAL_BUDGET_LEVEL_COUNT = 6


def load_response_curve_points(
    path: str | Path,
) -> tuple[ResponseCurvePoint, ...]:
    """Load strict public response-curve JSONL without passing through raw rows."""

    return tuple(ResponseCurvePoint.from_dict(row) for row in read_jsonl(path))


def validate_response_curve_points(
    points: Iterable[ResponseCurvePoint],
) -> tuple[ProblemResponseCurve, ...]:
    """Validate repeat identities and return deterministic problem curves."""

    ordered = tuple(points)
    if not ordered:
        raise OracleSchemaError("response-curve input contains no points")
    seen_runs: set[tuple[str, str, str, str, str, str]] = set()
    grouped: dict[ProblemKey, list[ResponseCurvePoint]] = defaultdict(list)
    identities: dict[ProblemKey, tuple[str, int, str]] = {}
    for point in ordered:
        run_key = (
            point.domain,
            point.model_key,
            point.setting,
            point.budget_unit,
            point.problem_id,
            point.source_run_id,
        )
        if run_key in seen_runs:
            raise OracleSchemaError(
                "duplicate response-curve source_run_id for problem "
                f"{point.problem_id!r}"
            )
        seen_runs.add(run_key)
        key = (
            point.domain,
            point.model_key,
            point.setting,
            point.budget_unit,
            point.problem_id,
        )
        identity = (point.suite_id, point.problem_index, point.problem_label)
        if key in identities and identities[key] != identity:
            raise OracleSchemaError(
                f"inconsistent suite identity for problem {point.problem_id!r}"
            )
        identities[key] = identity
        grouped[key].append(point)

    curves: list[ProblemResponseCurve] = []
    for key in sorted(
        grouped,
        key=lambda item: (
            item[1],
            item[0],
            grouped[item][0].suite_id,
            grouped[item][0].problem_index,
            item[3],
        ),
    ):
        domain, model_key, setting, budget_unit, problem_id = key
        suite_id, problem_index, problem_label = identities[key]
        curves.append(
            ProblemResponseCurve(
                domain=domain,  # type: ignore[arg-type]
                model_key=model_key,
                setting=setting,  # type: ignore[arg-type]
                budget_unit=budget_unit,  # type: ignore[arg-type]
                problem_id=problem_id,
                suite_id=suite_id,
                problem_index=problem_index,
                problem_label=problem_label,
                points=tuple(grouped[key]),
            )
        )
    return tuple(curves)


def build_min_success_cost_items(
    points: Iterable[ResponseCurvePoint],
) -> tuple[OracleItem, ...]:
    """Collapse each problem to its lowest observed full-credit success cost."""

    curves = validate_response_curve_points(points)
    items: list[OracleItem] = []
    for curve in curves:
        indexed_successes = [
            (index, point)
            for index, point in enumerate(curve.points)
            if point.reward == 1
            and point.parse_status == "parsed"
            and point.judge_status == "judged"
        ]
        if not indexed_successes:
            continue
        _, best = min(
            indexed_successes,
            key=lambda pair: (
                pair[1].observed_cost,
                pair[1].budget,
                pair[0],
            ),
        )
        items.append(
            OracleItem(
                domain=best.domain,
                model_key=best.model_key,
                setting=best.setting,
                budget_unit=best.budget_unit,
                problem_id=best.problem_id,
                suite_id=best.suite_id,
                problem_index=best.problem_index,
                problem_label=best.problem_label,
                budget=best.budget,
                observed_cost=best.observed_cost,
                reward=1,
                parse_status="parsed",
                judge_status="judged",
                source_run_id=best.source_run_id,
            )
        )
    return tuple(items)


def build_empirical_budget_options(
    points: Iterable[ResponseCurvePoint],
    *,
    expected_repeats: int = FORMAL_REPEAT_COUNT,
    expected_levels: int = FORMAL_BUDGET_LEVEL_COUNT,
) -> tuple[OracleBudgetOption, ...]:
    """Aggregate formal 6-level curves into five-repeat success probabilities.

    A budget level is an explicit identity rather than a distinct cost: low action
    caps may legitimately round to the same configured budget.  Observed usage is
    deliberately ignored here because the paper prices an option at its configured
    budget.
    """

    if expected_repeats <= 0 or expected_levels <= 0:
        raise ValueError("expected repeat and level counts must be positive")
    curves = validate_response_curve_points(points)
    options: list[OracleBudgetOption] = []
    expected_level_ids = tuple(range(1, expected_levels + 1))
    profile_level_budgets: dict[tuple[str, str, str, str], tuple[int, ...]] = {}
    for curve in curves:
        if any(
            point.repeat_id is None or point.budget_level is None
            for point in curve.points
        ):
            raise OracleSchemaError(
                "formal response curves require repeat_id and budget_level on every point"
            )
        by_level: dict[int, list[ResponseCurvePoint]] = defaultdict(list)
        for point in curve.points:
            assert point.budget_level is not None
            by_level[point.budget_level].append(point)
        if tuple(sorted(by_level)) != expected_level_ids:
            raise OracleSchemaError(
                f"problem {curve.problem_id!r} must contain budget levels "
                f"1 through {expected_levels}"
            )
        level_budgets: list[int] = []
        for level in expected_level_ids:
            rows = sorted(
                by_level[level],
                key=lambda point: int(point.repeat_id or 0),
            )
            if any(
                point.parse_status == "parsed"
                and point.judge_status == "not_judged"
                for point in rows
            ):
                raise OracleSchemaError(
                    "formal response curves cannot contain unresolved judge outcomes"
                )
            repeat_ids = tuple(point.repeat_id for point in rows)
            if repeat_ids != tuple(range(1, expected_repeats + 1)):
                raise OracleSchemaError(
                    f"problem {curve.problem_id!r} budget level {level} must "
                    f"contain repeat_id 1 through {expected_repeats}"
                )
            budgets = {point.budget for point in rows}
            if len(budgets) != 1:
                raise OracleSchemaError(
                    f"problem {curve.problem_id!r} budget level {level} has "
                    "inconsistent configured budgets"
                )
            budget = next(iter(budgets))
            level_budgets.append(budget)
            successes = sum(point.reward for point in rows)
            options.append(
                OracleBudgetOption(
                    domain=curve.domain,
                    model_key=curve.model_key,
                    setting=curve.setting,
                    budget_unit=curve.budget_unit,
                    problem_id=curve.problem_id,
                    suite_id=curve.suite_id,
                    problem_index=curve.problem_index,
                    problem_label=curve.problem_label,
                    budget_level=level,
                    budget=budget,
                    success_rate=successes / expected_repeats,
                    successful_repeats=successes,
                    repeat_count=expected_repeats,
                    source_run_ids=tuple(point.source_run_id for point in rows),
                )
            )
        budget_tuple = tuple(level_budgets)
        if any(left > right for left, right in zip(budget_tuple, budget_tuple[1:])):
            raise OracleSchemaError("configured budgets must be nondecreasing by level")
        if budget_tuple[0] != 0:
            raise OracleSchemaError(
                "formal response curves must begin with budget zero"
            )
        profile_key = (
            curve.domain,
            curve.model_key,
            curve.setting,
            curve.budget_unit,
        )
        if profile_key not in profile_level_budgets:
            profile_level_budgets[profile_key] = budget_tuple
        elif profile_level_budgets[profile_key] != budget_tuple:
            raise OracleSchemaError(
                "all problems in one model/domain/setting must share level budgets"
            )
    return tuple(options)


def load_oracle_items(path: str | Path) -> tuple[OracleItem, ...]:
    items = tuple(OracleItem.from_dict(row) for row in read_jsonl(path))
    seen: set[ProblemKey] = set()
    for item in items:
        key = (
            item.domain,
            item.model_key,
            item.setting,
            item.budget_unit,
            item.problem_id,
        )
        if key in seen:
            raise OracleSchemaError(
                f"duplicate Oracle item for problem {item.problem_id!r}"
            )
        seen.add(key)
    return items


def join_items_to_contest_suites(
    items: Iterable[OracleItem],
    suites: Sequence[ContestSuite],
) -> dict[str, tuple[OracleItem | None, ...]]:
    """Join one model/setting item set to canonical suites in suite order."""

    item_rows = tuple(items)
    if not item_rows:
        raise OracleSchemaError("cannot join an empty Oracle item set")
    profiles = {
        (item.domain, item.model_key, item.setting, item.budget_unit)
        for item in item_rows
    }
    if len(profiles) != 1:
        raise OracleSchemaError(
            "join_items_to_contest_suites requires one model/domain/setting"
        )
    domain, _, _, _ = next(iter(profiles))
    by_problem = {item.problem_id: item for item in item_rows}
    if len(by_problem) != len(item_rows):
        raise OracleSchemaError("Oracle item problem IDs must be unique")

    joined: dict[str, tuple[OracleItem | None, ...]] = {}
    for suite in suites:
        if suite.domain != domain:
            raise OracleSchemaError(f"suite {suite.suite_id!r} has the wrong domain")
        rows: list[OracleItem | None] = []
        for problem in suite.problems:
            item = by_problem.get(problem.problem_id)
            if item is not None and (
                item.suite_id != suite.suite_id
                or item.problem_index != problem.problem_index
                or item.problem_label != problem.problem_label
            ):
                raise OracleSchemaError(
                    f"Oracle item identity mismatch for {problem.problem_id!r}"
                )
            rows.append(item)
        joined[suite.suite_id] = tuple(rows)
    return joined


def validate_oracle_capacity(
    formal_contest_budget: int,
    response_curve_grid: Sequence[int] | None = None,
) -> int:
    """Validate and return the formal capacity without snapping to the grid."""

    if (
        isinstance(formal_contest_budget, bool)
        or not isinstance(formal_contest_budget, int)
        or formal_contest_budget < 0
    ):
        raise OracleSchemaError("formal_contest_budget must be a non-negative integer")
    if response_curve_grid is not None:
        grid = tuple(response_curve_grid)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in grid
        ):
            raise OracleSchemaError(
                "response_curve_grid must contain non-negative integers"
            )
        if tuple(sorted(grid)) != grid:
            raise OracleSchemaError("response_curve_grid must be nondecreasing")
    return formal_contest_budget


__all__ = [
    "build_min_success_cost_items",
    "build_empirical_budget_options",
    "FORMAL_BUDGET_LEVEL_COUNT",
    "FORMAL_REPEAT_COUNT",
    "join_items_to_contest_suites",
    "load_oracle_items",
    "load_response_curve_points",
    "validate_oracle_capacity",
    "validate_response_curve_points",
]
