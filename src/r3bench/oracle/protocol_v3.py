"""Condition-based analysis schema with read compatibility for v2 artifacts."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from r3bench.common.budget import (
    BudgetResolutionError,
    OfficialBudgetProfile,
    load_official_budget_profiles,
    resolve_official_budget_profile,
)
from r3bench.common.io import read_json, read_jsonl
from r3bench.common.result_schema import to_public_dict
from r3bench.common.schema import CONTEST_LABELS, DOMAINS
from r3bench.common.settings import validate_budget_unit
from r3bench.oracle.build_items import (
    FORMAL_BUDGET_LEVEL_COUNT,
    FORMAL_REPEAT_COUNT,
    build_empirical_budget_options,
    build_min_success_cost_items,
)
from r3bench.oracle.knapsack import (
    KnapsackItem,
    solve_knapsack,
    solve_multiple_choice_knapsack,
)
from r3bench.oracle.pipeline_io import load_contest_results, load_formal_budgets
from r3bench.oracle.response_curve_schema import (
    OracleItem,
    OracleBudgetOption,
    OracleSchemaError,
    ResponseCurvePoint,
)


_CONDITION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_CONDITION_KINDS = frozenset({"custom", "official_profile"})
_RESPONSE_V2_FIELDS = frozenset(
    {
        "domain",
        "model_key",
        "setting",
        "budget_unit",
        "mode",
        "problem_id",
        "suite_id",
        "problem_index",
        "problem_label",
        "budget",
        "observed_cost",
        "reward",
        "parse_status",
        "judge_status",
        "source_run_id",
    }
)
_RESPONSE_REPEAT_FIELDS = frozenset({"repeat_id", "budget_level"})
_CONTEST_V3_REQUIRED = frozenset(
    {
        "schema_version",
        "domain",
        "model_key",
        "setting",
        "budget_unit",
        "mode",
        "condition_id",
        "condition_kind",
        "contest_budget",
        "problem_id",
        "suite_id",
        "problem_index",
        "problem_label",
        "reward",
        "parse_status",
        "judge_status",
        "source_run_id",
    }
)
_CONDITION_OPTIONAL = frozenset({"budget_profile", "rho"})
_REPEAT_OPTIONAL = frozenset({"repeat_id"})
_BUDGET_V3_REQUIRED = frozenset(
    {
        "domain",
        "model_key",
        "setting",
        "budget_unit",
        "condition_id",
        "condition_kind",
        "contest_budget",
        "response_curve_grid",
    }
)


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OracleSchemaError(f"{field} must be a non-negative integer")
    return value


def _condition_id(value: object) -> str:
    if not isinstance(value, str) or not _CONDITION_ID.fullmatch(value):
        raise OracleSchemaError("condition_id contains unsupported characters")
    return value


def _condition_kind(value: object) -> str:
    if value not in _CONDITION_KINDS:
        raise OracleSchemaError("condition_kind must be 'custom' or 'official_profile'")
    return str(value)


def _rho(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OracleSchemaError("rho must be numeric or null")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise OracleSchemaError("rho must be finite and within [0, 1]")
    return result


def _identity(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    problem_id = row.get("problem_id")
    suite_id = row.get("suite_id")
    index = row.get("problem_index")
    label = row.get("problem_label")
    if not isinstance(problem_id, str) or not problem_id:
        raise OracleSchemaError("problem_id must be a non-empty string")
    if not isinstance(suite_id, str) or not suite_id:
        raise OracleSchemaError("suite_id must be a non-empty string")
    if isinstance(index, bool) or not isinstance(index, int) or not 1 <= index <= 6:
        raise OracleSchemaError("problem_index must be between 1 and 6")
    if label != CONTEST_LABELS[index - 1]:
        raise OracleSchemaError("problem_label must match problem_index")
    return problem_id, suite_id, index, str(label)


def _profile_fields(row: Mapping[str, Any]) -> tuple[str | None, float | None]:
    profile = row.get("budget_profile")
    if profile is not None and (not isinstance(profile, str) or not profile):
        raise OracleSchemaError("budget_profile must be a non-empty string or null")
    rho = _rho(row.get("rho"))
    kind = _condition_kind(row.get("condition_kind"))
    if kind == "custom" and (profile is not None or rho is not None):
        raise OracleSchemaError("custom conditions cannot claim profile or rho metadata")
    if kind == "official_profile" and (profile is None or rho is None):
        raise OracleSchemaError(
            "official_profile conditions require budget_profile and rho"
        )
    return profile, rho


def _official_response_curve_profile(
    *, domain: str, model_key: str, setting: str, budget_unit: str
) -> OfficialBudgetProfile:
    matches = [
        candidate
        for candidate in load_official_budget_profiles().values()
        if candidate.model_key == model_key
        and candidate.domain == domain
        and candidate.setting == setting
        and candidate.role == "single_problem_response_curve"
    ]
    if len(matches) != 1:
        raise OracleSchemaError(
            "official response-curve profile is missing or ambiguous"
        )
    profile = matches[0]
    if profile.budget_unit != budget_unit:
        raise OracleSchemaError(
            "official response-curve profile has the wrong budget unit"
        )
    return profile


def _validate_official_budget(record: "ConditionBudget") -> None:
    if record.condition_kind != "official_profile":
        return
    assert record.budget_profile is not None
    try:
        profile = resolve_official_budget_profile(
            record.budget_profile,
            setting=record.setting,  # type: ignore[arg-type]
            domain=record.domain,
            model_key=record.model_key,
        )
    except BudgetResolutionError as exc:
        raise OracleSchemaError(str(exc)) from exc
    if (
        profile.role not in {"budgeted_rho_0p2", "budgeted_rho_0p8"}
        or profile.budget_value != record.contest_budget
        or profile.rho != record.rho
    ):
        raise OracleSchemaError(
            "official condition does not match its rho contest budget profile"
        )
    curve = _official_response_curve_profile(
        domain=record.domain,
        model_key=record.model_key,
        setting=record.setting,
        budget_unit=record.budget_unit,
    )
    if curve.budget_grid != record.response_curve_grid:
        raise OracleSchemaError(
            "official condition response_curve_grid differs from its profile"
        )


@dataclass(frozen=True, slots=True)
class ConditionBudget:
    domain: str
    model_key: str
    setting: str
    budget_unit: str
    condition_id: str
    condition_kind: str
    contest_budget: int
    response_curve_grid: tuple[int, ...]
    budget_profile: str | None = None
    rho: float | None = None

    @classmethod
    def from_v3(cls, row: Mapping[str, Any]) -> "ConditionBudget":
        fields = frozenset(row)
        if not _BUDGET_V3_REQUIRED <= fields or fields - (
            _BUDGET_V3_REQUIRED | _CONDITION_OPTIONAL
        ):
            raise OracleSchemaError("v3 budget record has invalid fields")
        domain = row.get("domain")
        setting = row.get("setting")
        model = row.get("model_key")
        if domain not in DOMAINS or setting not in {"tool_free", "agentic"}:
            raise OracleSchemaError("v3 budget has an invalid domain or setting")
        if not isinstance(model, str) or not model:
            raise OracleSchemaError("model_key must be a non-empty string")
        try:
            unit = validate_budget_unit(str(setting), row.get("budget_unit"))
        except ValueError as exc:
            raise OracleSchemaError(str(exc)) from exc
        grid = row.get("response_curve_grid")
        if not isinstance(grid, list):
            raise OracleSchemaError("response_curve_grid must be an array")
        parsed_grid = tuple(
            _nonnegative_int(value, "response_curve_grid") for value in grid
        )
        if not parsed_grid or tuple(sorted(parsed_grid)) != parsed_grid:
            raise OracleSchemaError(
                "response_curve_grid must be non-empty and nondecreasing"
            )
        profile, rho = _profile_fields(row)
        return cls(
            domain=str(domain),
            model_key=model,
            setting=str(setting),
            budget_unit=str(unit),
            condition_id=_condition_id(row.get("condition_id")),
            condition_kind=_condition_kind(row.get("condition_kind")),
            contest_budget=_nonnegative_int(
                row.get("contest_budget"), "contest_budget"
            ),
            response_curve_grid=parsed_grid,
            budget_profile=profile,
            rho=rho,
        )


@dataclass(frozen=True, slots=True)
class ContestResultV3:
    domain: str
    model_key: str
    setting: str
    budget_unit: str
    mode: str
    condition_id: str
    condition_kind: str
    contest_budget: int
    problem_id: str
    suite_id: str
    problem_index: int
    problem_label: str
    reward: int
    parse_status: str
    judge_status: str
    source_run_id: str
    budget_profile: str | None = None
    rho: float | None = None
    repeat_id: int | None = None

    @classmethod
    def from_v3(cls, row: Mapping[str, Any]) -> "ContestResultV3":
        fields = frozenset(row)
        if not _CONTEST_V3_REQUIRED <= fields or fields - (
            _CONTEST_V3_REQUIRED | _CONDITION_OPTIONAL | _REPEAT_OPTIONAL
        ):
            raise OracleSchemaError("v3 contest record has invalid fields")
        if row.get("schema_version") != "3.0" or row.get("mode") != "contest":
            raise OracleSchemaError("invalid v3 contest metadata")
        problem_id, suite_id, index, label = _identity(row)
        budget = ConditionBudget.from_v3(
            {key: row[key] for key in _BUDGET_V3_REQUIRED - {"response_curve_grid"}}
            | {
                "response_curve_grid": [row["contest_budget"]],
                **{key: row[key] for key in _CONDITION_OPTIONAL if key in row},
            }
        )
        reward = row.get("reward")
        parse = row.get("parse_status")
        judge = row.get("judge_status")
        run_id = row.get("source_run_id")
        if isinstance(reward, bool) or reward not in {0, 1}:
            raise OracleSchemaError("reward must be 0 or 1")
        if parse not in {"parsed", "missing", "parse_error"}:
            raise OracleSchemaError("invalid parse_status")
        if judge not in {"judged", "not_judged", "judge_error"}:
            raise OracleSchemaError("invalid judge_status")
        if reward == 1 and (parse != "parsed" or judge != "judged"):
            raise OracleSchemaError("successful contest rows must be parsed and judged")
        if not isinstance(run_id, str) or not run_id:
            raise OracleSchemaError("source_run_id must be a non-empty string")
        repeat_id = (
            _nonnegative_int(row["repeat_id"], "repeat_id")
            if row.get("repeat_id") is not None
            else None
        )
        if repeat_id == 0:
            raise OracleSchemaError("repeat_id must be a positive integer")
        return cls(
            domain=budget.domain,
            model_key=budget.model_key,
            setting=budget.setting,
            budget_unit=budget.budget_unit,
            mode="contest",
            condition_id=budget.condition_id,
            condition_kind=budget.condition_kind,
            contest_budget=budget.contest_budget,
            problem_id=problem_id,
            suite_id=suite_id,
            problem_index=index,
            problem_label=label,
            reward=int(reward),
            parse_status=str(parse),
            judge_status=str(judge),
            source_run_id=run_id,
            budget_profile=budget.budget_profile,
            rho=budget.rho,
            repeat_id=repeat_id,
        )


def _legacy_condition_id(rho: float) -> str:
    return f"official_rho_{str(rho).replace('.', 'p')}"


def load_condition_budgets(path: str | Path) -> tuple[ConditionBudget, ...]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise OracleSchemaError("budget document must be an object")
    version = value.get("schema_version")
    if version == "2.0":
        rows = tuple(
            ConditionBudget(
                domain=row.domain,
                model_key=row.model_key,
                setting=row.setting,
                budget_unit=row.budget_unit,
                condition_id=_legacy_condition_id(row.rho),
                condition_kind="custom",
                contest_budget=row.formal_contest_budget,
                response_curve_grid=row.response_curve_grid,
                budget_profile=None,
                rho=None,
            )
            for row in load_formal_budgets(path)
        )
    elif version == "3.0" and frozenset(value) == {"schema_version", "budgets"}:
        raw = value.get("budgets")
        if not isinstance(raw, list) or not raw:
            raise OracleSchemaError("v3 budget document requires non-empty budgets")
        rows = tuple(ConditionBudget.from_v3(row) for row in raw)
    else:
        raise OracleSchemaError("unsupported budget schema_version")
    keys = [(r.domain, r.model_key, r.setting, r.condition_id) for r in rows]
    if len(keys) != len(set(keys)):
        raise OracleSchemaError("budget conditions must have unique cell keys")
    for row in rows:
        _validate_official_budget(row)
    return rows


def load_contest_results_compatible(path: str | Path) -> tuple[ContestResultV3, ...]:
    raw = read_jsonl(path)
    if not raw:
        raise OracleSchemaError("contest result input is empty")
    if all(row.get("schema_version") == "3.0" for row in raw):
        rows = tuple(ContestResultV3.from_v3(row) for row in raw)
    elif all("schema_version" not in row for row in raw):
        rows = tuple(
            ContestResultV3(
                domain=row.domain,
                model_key=row.model_key,
                setting=row.setting,
                budget_unit=row.budget_unit,
                mode="contest",
                condition_id=_legacy_condition_id(row.rho),
                condition_kind="custom",
                contest_budget=row.formal_contest_budget,
                problem_id=row.problem_id,
                suite_id=row.suite_id,
                problem_index=row.problem_index,
                problem_label=row.problem_label,
                reward=row.reward,
                parse_status=row.parse_status,
                judge_status=row.judge_status,
                source_run_id=row.source_run_id,
                budget_profile=None,
                rho=None,
            )
            for row in load_contest_results(path)
        )
    else:
        raise OracleSchemaError("contest rows cannot mix v2 and v3 schemas")
    seen: set[tuple[str, str, str, str, str, str, int | None]] = set()
    for row in rows:
        key = (
            row.domain,
            row.model_key,
            row.setting,
            row.condition_id,
            row.suite_id,
            row.problem_id,
            row.repeat_id,
        )
        if key in seen:
            raise OracleSchemaError("duplicate contest problem result")
        seen.add(key)
    return rows


def load_response_curve_points_compatible(
    path: str | Path,
) -> tuple[ResponseCurvePoint, ...]:
    raw = read_jsonl(path)
    points: list[ResponseCurvePoint] = []
    for row in raw:
        if row.get("schema_version") == "3.0":
            allowed = (
                _RESPONSE_V2_FIELDS
                | _RESPONSE_REPEAT_FIELDS
                | {
                    "schema_version",
                    "condition_id",
                    "condition_kind",
                }
            )
            required = _RESPONSE_V2_FIELDS | {
                "schema_version",
                "condition_id",
                "condition_kind",
            }
            fields = frozenset(row)
            if not required <= fields or fields - allowed:
                raise OracleSchemaError("v3 response-curve point has invalid fields")
            _condition_id(row.get("condition_id"))
            _condition_kind(row.get("condition_kind"))
            row = {key: value for key, value in row.items() if key != "schema_version"}
        points.append(ResponseCurvePoint.from_dict(row))
    if not points:
        raise OracleSchemaError("response-curve input contains no points")
    return tuple(points)


def _group_contests(
    rows: Iterable[ContestResultV3],
) -> dict[tuple[str, str, str, str, str, int | None], tuple[ContestResultV3, ...]]:
    grouped: dict[
        tuple[str, str, str, str, str, int | None], list[ContestResultV3]
    ] = {}
    for row in rows:
        key = (
            row.domain,
            row.model_key,
            row.setting,
            row.condition_id,
            row.suite_id,
            row.repeat_id,
        )
        grouped.setdefault(key, []).append(row)
    result: dict[
        tuple[str, str, str, str, str, int | None], tuple[ContestResultV3, ...]
    ] = {}
    for key, values in grouped.items():
        ordered = tuple(sorted(values, key=lambda item: item.problem_index))
        if len(ordered) != 6 or tuple(r.problem_index for r in ordered) != tuple(
            range(1, 7)
        ):
            raise OracleSchemaError("each contest suite must contain positions 1-6")
        if len({r.problem_id for r in ordered}) != 6:
            raise OracleSchemaError("contest suite problem IDs must be distinct")
        if len({r.contest_budget for r in ordered}) != 1:
            raise OracleSchemaError("contest suite budgets are inconsistent")
        result[key] = ordered
    return result


def _item_index(
    items: Iterable[OracleItem],
) -> dict[tuple[str, str, str, str, str], OracleItem]:
    return {
        (
            item.domain,
            item.model_key,
            item.setting,
            item.budget_unit,
            item.problem_id,
        ): item
        for item in items
    }


def _option_index(
    options: Iterable[OracleBudgetOption],
) -> dict[tuple[str, str, str, str, str], tuple[OracleBudgetOption, ...]]:
    grouped: dict[tuple[str, str, str, str, str], list[OracleBudgetOption]] = (
        defaultdict(list)
    )
    for option in options:
        key = (
            option.domain,
            option.model_key,
            option.setting,
            option.budget_unit,
            option.problem_id,
        )
        grouped[key].append(option)
    return {
        key: tuple(sorted(rows, key=lambda option: option.budget_level))
        for key, rows in grouped.items()
    }


def _validate_formal_contest_repeats(
    contests: Iterable[ContestResultV3],
) -> None:
    by_suite: dict[
        tuple[str, str, str, str, str], dict[int, set[str]]
    ] = defaultdict(lambda: defaultdict(set))
    for row in contests:
        if row.repeat_id is None:
            raise OracleSchemaError(
                "formal contest results require repeat_id on every row"
            )
        if row.parse_status == "parsed" and row.judge_status == "not_judged":
            raise OracleSchemaError(
                "formal contest results cannot contain unresolved judge outcomes"
            )
        by_suite[
            (
                row.domain,
                row.model_key,
                row.setting,
                row.condition_id,
                row.suite_id,
            )
        ][row.repeat_id].add(row.source_run_id)
    expected = set(range(1, FORMAL_REPEAT_COUNT + 1))
    for key, repeat_sources in by_suite.items():
        if set(repeat_sources) != expected:
            raise OracleSchemaError(
                f"contest suite {key[-1]!r} must contain repeat_id 1 through "
                f"{FORMAL_REPEAT_COUNT}"
            )
        if any(len(sources) != 1 for sources in repeat_sources.values()) or len(
            {next(iter(sources)) for sources in repeat_sources.values()}
        ) != FORMAL_REPEAT_COUNT:
            raise OracleSchemaError(
                f"contest suite {key[-1]!r} requires five unique source runs"
            )


def _formal_options_for_problem(
    indexed: Mapping[tuple[str, str, str, str, str], tuple[OracleBudgetOption, ...]],
    *,
    domain: str,
    model_key: str,
    setting: str,
    budget_unit: str,
    problem_id: str,
) -> tuple[OracleBudgetOption, ...]:
    key = (domain, model_key, setting, budget_unit, problem_id)
    try:
        return indexed[key]
    except KeyError as exc:
        raise OracleSchemaError(
            f"missing formal response curve for problem {problem_id!r}"
        ) from exc


def _curve_cell(
    point: ResponseCurvePoint,
) -> tuple[str, str, str, str]:
    return (
        point.domain,
        point.model_key,
        point.setting,
        point.budget_unit,
    )


def _validate_official_input_bindings(
    points: tuple[ResponseCurvePoint, ...],
    contests: tuple[ContestResultV3, ...],
    budgets: tuple[ConditionBudget, ...],
) -> None:
    if any(point.condition_kind != "official_profile" for point in points):
        raise OracleSchemaError(
            "official analysis requires every response-curve row to use "
            "condition_kind='official_profile'"
        )
    if any(row.condition_kind != "official_profile" for row in contests):
        raise OracleSchemaError(
            "official analysis cannot mix custom contest results"
        )
    if any(row.condition_kind != "official_profile" for row in budgets):
        raise OracleSchemaError("official analysis cannot mix custom budgets")

    curve_conditions: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for point in points:
        assert point.condition_id is not None
        curve_conditions[_curve_cell(point)].add(point.condition_id)
    for cell, condition_ids in curve_conditions.items():
        if len(condition_ids) != 1:
            raise OracleSchemaError(
                "official response-curve cell must bind exactly one condition"
            )
        domain, model_key, setting, budget_unit = cell
        expected = _official_response_curve_profile(
            domain=domain,
            model_key=model_key,
            setting=setting,
            budget_unit=budget_unit,
        )
        if next(iter(condition_ids)) != expected.profile_id:
            raise OracleSchemaError(
                "official response-curve condition does not match its unique "
                "single-problem budget profile"
            )

    curve_cells = set(curve_conditions)
    budget_cells = {
        (row.domain, row.model_key, row.setting, row.budget_unit) for row in budgets
    }
    contest_cells = {
        (row.domain, row.model_key, row.setting, row.budget_unit) for row in contests
    }
    if curve_cells != budget_cells or curve_cells != contest_cells:
        raise OracleSchemaError(
            "official response curves, contests, and budgets must cover the same cells"
        )
    budget_conditions = {
        (row.domain, row.model_key, row.setting, row.condition_id) for row in budgets
    }
    contest_conditions = {
        (row.domain, row.model_key, row.setting, row.condition_id) for row in contests
    }
    if budget_conditions != contest_conditions:
        raise OracleSchemaError(
            "official contest conditions must bind one matching budget condition"
        )


def _validate_official_curve_grids(
    options: tuple[OracleBudgetOption, ...],
) -> None:
    grids: dict[
        tuple[str, str, str, str],
        dict[str, list[OracleBudgetOption]],
    ] = defaultdict(lambda: defaultdict(list))
    for option in options:
        cell = (
            option.domain,
            option.model_key,
            option.setting,
            option.budget_unit,
        )
        grids[cell][option.problem_id].append(option)
    for cell, by_problem in grids.items():
        actual_grids = {
            tuple(
                option.budget
                for option in sorted(rows, key=lambda item: item.budget_level)
            )
            for rows in by_problem.values()
        }
        domain, model_key, setting, budget_unit = cell
        expected = _official_response_curve_profile(
            domain=domain,
            model_key=model_key,
            setting=setting,
            budget_unit=budget_unit,
        )
        if actual_grids != {expected.budget_grid}:
            raise OracleSchemaError(
                "official response-curve observations do not match their profile grid"
            )


def run_condition_analysis(
    *,
    response_curve_path: str | Path,
    contest_results_path: str | Path,
    budgets_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run equal replay, exact knapsack, and gap aggregation for v2/v3 inputs."""

    points = load_response_curve_points_compatible(response_curve_path)
    contests = load_contest_results_compatible(contest_results_path)
    budgets = load_condition_budgets(budgets_path)
    repeat_aware = any(
        point.repeat_id is not None or point.budget_level is not None
        for point in points
    )
    official_declared = any(
        row.condition_kind == "official_profile"
        for row in (*points, *budgets, *contests)
    )
    if official_declared:
        _validate_official_input_bindings(points, contests, budgets)
    formal_curve = repeat_aware or official_declared
    if formal_curve:
        options = build_empirical_budget_options(points)
        items: tuple[OracleItem, ...] = ()
        if official_declared:
            _validate_official_curve_grids(options)
    else:
        options = ()
        items = build_min_success_cost_items(points)
    if formal_curve:
        _validate_formal_contest_repeats(contests)
    elif any(row.repeat_id is not None for row in contests):
        raise OracleSchemaError(
            "repeat-aware contests require repeat-aware response curves"
        )
    budget_index = {
        (row.domain, row.model_key, row.setting, row.condition_id): row
        for row in budgets
    }
    indexed_items = _item_index(items)
    indexed_options = _option_index(options)
    equal_results: list[dict[str, Any]] = []
    oracle_results: list[dict[str, Any]] = []
    aggregate: dict[
        tuple[str, str, str, str],
        list[tuple[str, int | None, float, float, float]],
    ] = defaultdict(list)

    for key, rows in sorted(_group_contests(contests).items()):
        domain, model_key, setting, condition_id, suite_id, repeat_id = key
        cell_key = (domain, model_key, setting, condition_id)
        if cell_key not in budget_index:
            raise OracleSchemaError(f"missing budget condition for suite {suite_id!r}")
        budget = budget_index[cell_key]
        if rows[0].budget_unit != budget.budget_unit:
            raise OracleSchemaError("contest and condition budget units differ")
        if rows[0].contest_budget != budget.contest_budget:
            raise OracleSchemaError("contest and condition budget values differ")
        if any(
            row.condition_kind != budget.condition_kind
            or row.budget_profile != budget.budget_profile
            or row.rho != budget.rho
            for row in rows
        ):
            raise OracleSchemaError("contest and budget condition metadata differ")
        if formal_curve:
            if len(budget.response_curve_grid) != FORMAL_BUDGET_LEVEL_COUNT:
                raise OracleSchemaError(
                    f"formal response_curve_grid must contain exactly "
                    f"{FORMAL_BUDGET_LEVEL_COUNT} levels"
                )
            first_options = _formal_options_for_problem(
                indexed_options,
                domain=domain,
                model_key=model_key,
                setting=setting,
                budget_unit=budget.budget_unit,
                problem_id=rows[0].problem_id,
            )
            if tuple(option.budget for option in first_options) != tuple(
                budget.response_curve_grid
            ):
                raise OracleSchemaError(
                    "condition response_curve_grid does not match configured levels"
                )
        per_problem = budget.contest_budget // 6
        equal_problem_rows: list[dict[str, Any]] = []
        candidates: list[KnapsackItem] = []
        item_by_problem: dict[str, OracleItem] = {}
        option_by_key: dict[str, OracleBudgetOption] = {}
        for row in rows:
            if formal_curve:
                problem_options = _formal_options_for_problem(
                    indexed_options,
                    domain=domain,
                    model_key=model_key,
                    setting=setting,
                    budget_unit=budget.budget_unit,
                    problem_id=row.problem_id,
                )
                if any(
                    option.suite_id != suite_id
                    or option.problem_index != row.problem_index
                    or option.problem_label != row.problem_label
                    for option in problem_options
                ):
                    raise OracleSchemaError(
                        "response-curve and contest identities differ"
                    )
                affordable = tuple(
                    option for option in problem_options if option.budget <= per_problem
                )
                if not affordable:
                    raise OracleSchemaError(
                        "equal allocation has no affordable response-curve level"
                    )
                equal_option = max(
                    affordable,
                    key=lambda option: option.budget_level,
                )
                equal_problem_rows.append(
                    {
                        "problem_id": row.problem_id,
                        "problem_index": row.problem_index,
                        "problem_label": row.problem_label,
                        "allocated_budget": per_problem,
                        "configured_budget": equal_option.budget,
                        "budget_level": equal_option.budget_level,
                        "success_rate": equal_option.success_rate,
                        "successful_repeats": equal_option.successful_repeats,
                        "repeat_count": equal_option.repeat_count,
                        "source_run_ids": list(equal_option.source_run_ids),
                    }
                )
                for option in problem_options:
                    option_key = (
                        f"{row.problem_index}:{row.problem_id}:"
                        f"level_{option.budget_level}"
                    )
                    option_by_key[option_key] = option
                continue
            item = indexed_items.get(
                (domain, model_key, setting, budget.budget_unit, row.problem_id)
            )
            if item is not None:
                if (
                    item.suite_id != suite_id
                    or item.problem_index != row.problem_index
                    or item.problem_label != row.problem_label
                ):
                    raise OracleSchemaError(
                        "response-curve and contest identities differ"
                    )
                item_by_problem[row.problem_id] = item
                candidates.append(KnapsackItem(row.problem_id, item.observed_cost, 1))
            selected = item is not None and item.observed_cost <= per_problem
            equal_problem_rows.append(
                {
                    "problem_id": row.problem_id,
                    "problem_index": row.problem_index,
                    "problem_label": row.problem_label,
                    "observed_cost": item.observed_cost if item else None,
                    "allocated_budget": per_problem,
                    "reward": int(selected),
                    "selected_by_equal": selected,
                    "source_run_id": item.source_run_id if item else None,
                }
            )
        if formal_curve:
            groups = []
            for row in rows:
                problem_options = _formal_options_for_problem(
                    indexed_options,
                    domain=domain,
                    model_key=model_key,
                    setting=setting,
                    budget_unit=budget.budget_unit,
                    problem_id=row.problem_id,
                )
                groups.append(
                    tuple(
                        KnapsackItem(
                            key=(
                                f"{row.problem_index}:{row.problem_id}:"
                                f"level_{option.budget_level}"
                            ),
                            cost=option.budget,
                            value=float(option.successful_repeats),
                        )
                        for option in problem_options
                    )
                )
            solution = solve_multiple_choice_knapsack(groups, budget.contest_budget)
            selected_options = {
                option_by_key[key].problem_id: option_by_key[key]
                for key in solution.selected_keys
            }
            oracle_problem_rows = [
                {
                    "problem_id": row.problem_id,
                    "problem_index": row.problem_index,
                    "problem_label": row.problem_label,
                    "budget_level": selected_options[row.problem_id].budget_level,
                    "configured_budget": selected_options[row.problem_id].budget,
                    "success_rate": selected_options[row.problem_id].success_rate,
                    "successful_repeats": selected_options[
                        row.problem_id
                    ].successful_repeats,
                    "repeat_count": selected_options[row.problem_id].repeat_count,
                    "source_run_ids": list(
                        selected_options[row.problem_id].source_run_ids
                    ),
                }
                for row in rows
            ]
        else:
            solution = solve_knapsack(candidates, budget.contest_budget)
            selected_ids = set(solution.selected_keys)
            oracle_problem_rows = [
                {
                    "problem_id": row.problem_id,
                    "problem_index": row.problem_index,
                    "problem_label": row.problem_label,
                    "observed_cost": (
                        item_by_problem[row.problem_id].observed_cost
                        if row.problem_id in item_by_problem
                        else None
                    ),
                    "reward": int(row.problem_id in selected_ids),
                    "selected_by_oracle": row.problem_id in selected_ids,
                    "source_run_id": (
                        item_by_problem[row.problem_id].source_run_id
                        if row.problem_id in item_by_problem
                        else None
                    ),
                }
                for row in rows
            ]
        common = {
            "domain": domain,
            "model_key": model_key,
            "setting": setting,
            "budget_unit": budget.budget_unit,
            "mode": "contest",
            "condition_id": condition_id,
            "condition_kind": budget.condition_kind,
            "budget_profile": budget.budget_profile,
            "rho": budget.rho,
            "contest_budget": budget.contest_budget,
            "suite_id": suite_id,
            "repeat_id": repeat_id,
        }
        equal_score = sum(
            float(row.get("success_rate", row.get("reward", 0)))
            for row in equal_problem_rows
        )
        equal_results.append(
            {
                **common,
                "per_problem_budget": per_problem,
                "equal_score": equal_score,
                "problem_results": equal_problem_rows,
            }
        )
        oracle_value = (
            float(solution.total_value) / FORMAL_REPEAT_COUNT
            if formal_curve
            else float(solution.total_value)
        )
        oracle_cost = int(solution.total_cost)
        oracle_results.append(
            {
                **common,
                "capacity_source": "contest_budget",
                "oracle_score": oracle_value,
                "total_selected_cost": oracle_cost,
                "problem_selections": oracle_problem_rows,
                "combination_count": getattr(solution, "combination_count", 64),
            }
        )
        aggregate[cell_key].append(
            (
                suite_id,
                repeat_id,
                float(sum(row.reward for row in rows)),
                equal_score,
                oracle_value,
            )
        )

    summaries: list[dict[str, Any]] = []
    for cell_key, values in sorted(aggregate.items()):
        domain, model_key, setting, condition_id = cell_key
        budget = budget_index[cell_key]
        contest_run_count = len(values)
        suite_count = len({value[0] for value in values})
        repeat_count = FORMAL_REPEAT_COUNT if formal_curve else 1
        contest_total = sum(value[2] for value in values)
        equal_total = sum(value[3] for value in values)
        oracle_total = sum(value[4] for value in values)
        contest_score = contest_total / contest_run_count
        oracle_score = oracle_total / contest_run_count
        delta = oracle_score - contest_score
        summaries.append(
            {
                "domain": domain,
                "model_key": model_key,
                "setting": setting,
                "budget_unit": budget.budget_unit,
                "mode": "contest",
                "condition_id": condition_id,
                "condition_kind": budget.condition_kind,
                "budget_profile": budget.budget_profile,
                "rho": budget.rho,
                "contest_budget": budget.contest_budget,
                "suite_count": suite_count,
                "repeat_count": repeat_count,
                "contest_run_count": contest_run_count,
                "contest_total": contest_total,
                "equal_total": equal_total,
                "oracle_total": oracle_total,
                "contest_score": contest_score,
                "equal_score": equal_total / contest_run_count,
                "oracle_score": oracle_score,
                "contest_oracle_gap": delta,
                "gap_ratio": delta / oracle_score if oracle_score > 0 else None,
            }
        )

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    oracle_records: tuple[OracleItem | OracleBudgetOption, ...] = (
        options if formal_curve else items
    )
    (target / "oracle_items.jsonl").write_text(
        "".join(
            json.dumps(
                {"schema_version": "3.0", **to_public_dict(item)},
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
            for item in oracle_records
        ),
        encoding="utf-8",
    )
    for filename, kind, rows in (
        ("equal_replay.json", "equal_replay", equal_results),
        ("oracle_results.json", "oracle_results", oracle_results),
    ):
        (target / filename).write_text(
            json.dumps(
                {"schema_version": "3.0", "kind": kind, "results": rows},
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    gap_document = {
        "schema_version": "3.0",
        "kind": "gap_summary",
        "summaries": summaries,
    }
    (target / "gap_summary.json").write_text(
        json.dumps(
            gap_document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": "3.0",
        "status": "complete",
        "response_curve_point_count": len(points),
        "minimum_success_item_count": len(items),
        "oracle_budget_option_count": len(options),
        "oracle_protocol": (
            "six_level_five_repeat_mckp" if formal_curve else "legacy_binary"
        ),
        "suite_count": len(equal_results),
        "condition_count": len(summaries),
    }


def write_condition_budget_document(
    path: str | Path, records: Iterable[ConditionBudget]
) -> None:
    rows = []
    for record in records:
        value = asdict(record)
        value["response_curve_grid"] = list(record.response_curve_grid)
        rows.append(value)
    Path(path).write_text(
        json.dumps(
            {"schema_version": "3.0", "budgets": rows},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ConditionBudget",
    "ContestResultV3",
    "load_condition_budgets",
    "load_contest_results_compatible",
    "load_response_curve_points_compatible",
    "run_condition_analysis",
    "write_condition_budget_document",
]
