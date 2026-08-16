"""Strict public records for Tool-Free response-curve and Oracle replay."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any, Literal, Mapping

from r3bench.common.schema import CONTEST_LABELS, DOMAINS, Domain
from r3bench.common.settings import BudgetUnit, validate_budget_unit


Setting = Literal["tool_free", "agentic"]
Mode = Literal["single_problem", "contest"]
ParseStatus = Literal["parsed", "missing", "parse_error"]
JudgeStatus = Literal["judged", "not_judged", "judge_error"]

_PARSE_STATUSES = frozenset({"parsed", "missing", "parse_error"})
_JUDGE_STATUSES = frozenset({"judged", "not_judged", "judge_error"})
_PRIVATE_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:home|mnt)/|/tmp/rbench(?:/|\b)")
_CREDENTIAL = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{12,}\b|\bhf_[A-Za-z0-9]{12,}\b|"
    r"\bolp_[A-Za-z0-9]{12,}\b|Bearer\s+[A-Za-z0-9._~+/-]{12,})"
)
_PRIVATE_ENDPOINT = re.compile(
    r"(?i)https?://(?:localhost|127(?:\.\d{1,3}){3}|10(?:\.\d{1,3}){3}|"
    r"192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])"
    r"(?:\.\d{1,3}){2}|[^/\s]+\.internal)(?::\d+)?(?:/|\b)"
)
_CONDITION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


class OracleSchemaError(ValueError):
    """Raised when an Oracle pipeline record violates the public contract."""


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        detail: list[str] = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unknown fields: {', '.join(unknown)}")
        raise OracleSchemaError(f"{context} has invalid fields ({'; '.join(detail)})")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OracleSchemaError(f"{field} must be a non-empty string")
    if _PRIVATE_PATH.search(value):
        raise OracleSchemaError(f"{field} contains a private machine path")
    if _CREDENTIAL.search(value):
        raise OracleSchemaError(f"{field} contains a credential-like value")
    if _PRIVATE_ENDPOINT.search(value):
        raise OracleSchemaError(f"{field} contains a private endpoint")
    return value


def _domain(value: object) -> Domain:
    if value not in DOMAINS:
        raise OracleSchemaError(f"unsupported domain: {value!r}")
    return value  # type: ignore[return-value]


def _setting(value: object) -> Setting:
    if value not in {"tool_free", "agentic"}:
        raise OracleSchemaError("setting must be 'tool_free' or 'agentic'")
    return value  # type: ignore[return-value]


def _budget_unit(value: object, setting: Setting) -> BudgetUnit:
    if value not in {"output_tokens", "counted_actions"}:
        raise OracleSchemaError(
            "budget_unit must be 'output_tokens' or 'counted_actions'"
        )
    try:
        return validate_budget_unit(setting, value)  # type: ignore[arg-type]
    except ValueError as exc:
        raise OracleSchemaError(str(exc)) from exc


def _mode(value: object, expected: Mode) -> Mode:
    if value != expected:
        raise OracleSchemaError(f"mode must be {expected!r}")
    return expected


def _int(
    value: object, field: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OracleSchemaError(f"{field} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = (
            f"between {minimum} and {maximum}"
            if maximum is not None
            else f"at least {minimum}"
        )
        raise OracleSchemaError(f"{field} must be {bound}")
    return value


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _int(value, field)


def _reward(value: object) -> int:
    if isinstance(value, bool) or value not in (0, 1):
        raise OracleSchemaError("reward must be the integer 0 or 1")
    return int(value)


def _rho(value: object, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OracleSchemaError("rho must be 0.2 or 0.8")
    numeric = float(value)
    if numeric not in (0.2, 0.8):
        raise OracleSchemaError("rho must be 0.2 or 0.8")
    return numeric


def _problem_identity(
    *,
    problem_id: object,
    suite_id: object,
    problem_index: object,
    problem_label: object,
) -> tuple[str, str, int, str]:
    parsed_id = _text(problem_id, "problem_id")
    parsed_suite = _text(suite_id, "suite_id")
    parsed_index = _int(problem_index, "problem_index", minimum=1, maximum=6)
    if problem_label != CONTEST_LABELS[parsed_index - 1]:
        raise OracleSchemaError(
            "problem_label must match problem_index using canonical A-F labels"
        )
    return parsed_id, parsed_suite, parsed_index, str(problem_label)


def _statuses(
    parse_status: object, judge_status: object, reward: int
) -> tuple[ParseStatus, JudgeStatus]:
    if parse_status not in _PARSE_STATUSES:
        raise OracleSchemaError(f"unsupported parse_status: {parse_status!r}")
    if judge_status not in _JUDGE_STATUSES:
        raise OracleSchemaError(f"unsupported judge_status: {judge_status!r}")
    if reward == 1 and (parse_status != "parsed" or judge_status != "judged"):
        raise OracleSchemaError(
            "a successful point requires parse_status='parsed' and "
            "judge_status='judged'"
        )
    return parse_status, judge_status  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ResponseCurvePoint:
    domain: Domain
    model_key: str
    setting: Setting
    budget_unit: BudgetUnit
    mode: Literal["single_problem"]
    problem_id: str
    suite_id: str
    problem_index: int
    problem_label: str
    budget: int
    observed_cost: int
    reward: int
    parse_status: ParseStatus
    judge_status: JudgeStatus
    source_run_id: str
    repeat_id: int | None = None
    budget_level: int | None = None
    condition_id: str | None = None
    condition_kind: Literal["custom", "official_profile"] | None = None

    def __post_init__(self) -> None:
        _domain(self.domain)
        _text(self.model_key, "model_key")
        setting = _setting(self.setting)
        _budget_unit(self.budget_unit, setting)
        _mode(self.mode, "single_problem")
        _problem_identity(
            problem_id=self.problem_id,
            suite_id=self.suite_id,
            problem_index=self.problem_index,
            problem_label=self.problem_label,
        )
        _int(self.budget, "budget")
        _int(self.observed_cost, "observed_cost")
        reward = _reward(self.reward)
        _statuses(self.parse_status, self.judge_status, reward)
        _text(self.source_run_id, "source_run_id")
        if self.repeat_id is not None:
            _int(self.repeat_id, "repeat_id", minimum=1)
        if self.budget_level is not None:
            _int(self.budget_level, "budget_level", minimum=1, maximum=6)
        if (self.condition_id is None) != (self.condition_kind is None):
            raise OracleSchemaError(
                "response-curve condition_id and condition_kind must be set together"
            )
        if self.condition_id is not None and not _CONDITION_ID.fullmatch(
            self.condition_id
        ):
            raise OracleSchemaError(
                "response-curve condition_id contains unsupported characters"
            )
        if self.condition_kind not in {None, "custom", "official_profile"}:
            raise OracleSchemaError(
                "response-curve condition_kind must be custom or official_profile"
            )
        if self.budget == 0 and self.observed_cost != 0:
            raise OracleSchemaError("zero-budget points must have zero observed cost")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResponseCurvePoint":
        required = _RESPONSE_POINT_FIELDS - {
            "repeat_id",
            "budget_level",
            "condition_id",
            "condition_kind",
        }
        fields = frozenset(value)
        if not required <= fields or fields - _RESPONSE_POINT_FIELDS:
            missing = sorted(required - fields)
            unknown = sorted(fields - _RESPONSE_POINT_FIELDS)
            detail = []
            if missing:
                detail.append(f"missing fields: {', '.join(missing)}")
            if unknown:
                detail.append(f"unknown fields: {', '.join(unknown)}")
            raise OracleSchemaError(
                f"response-curve point has invalid fields ({'; '.join(detail)})"
            )
        return cls(
            domain=_domain(value["domain"]),
            model_key=_text(value["model_key"], "model_key"),
            setting=_setting(value["setting"]),
            budget_unit=_budget_unit(value["budget_unit"], _setting(value["setting"])),
            mode=_mode(value["mode"], "single_problem"),
            problem_id=value["problem_id"],
            suite_id=value["suite_id"],
            problem_index=value["problem_index"],
            problem_label=value["problem_label"],
            budget=value["budget"],
            observed_cost=value["observed_cost"],
            reward=value["reward"],
            parse_status=value["parse_status"],
            judge_status=value["judge_status"],
            source_run_id=value["source_run_id"],
            repeat_id=value.get("repeat_id"),
            budget_level=value.get("budget_level"),
            condition_id=value.get("condition_id"),
            condition_kind=value.get("condition_kind"),
        )


_RESPONSE_POINT_FIELDS = frozenset(
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
        "repeat_id",
        "budget_level",
        "condition_id",
        "condition_kind",
    }
)


@dataclass(frozen=True, slots=True)
class ProblemResponseCurve:
    domain: Domain
    model_key: str
    setting: Setting
    budget_unit: BudgetUnit
    problem_id: str
    suite_id: str
    problem_index: int
    problem_label: str
    points: tuple[ResponseCurvePoint, ...]

    def __post_init__(self) -> None:
        _domain(self.domain)
        _text(self.model_key, "model_key")
        setting = _setting(self.setting)
        _budget_unit(self.budget_unit, setting)
        _problem_identity(
            problem_id=self.problem_id,
            suite_id=self.suite_id,
            problem_index=self.problem_index,
            problem_label=self.problem_label,
        )
        points = tuple(self.points)
        object.__setattr__(self, "points", points)
        if not points:
            raise OracleSchemaError("a problem response curve requires points")
        for point in points:
            if (
                point.domain,
                point.model_key,
                point.setting,
                point.budget_unit,
                point.problem_id,
                point.suite_id,
                point.problem_index,
                point.problem_label,
            ) != (
                self.domain,
                self.model_key,
                self.setting,
                self.budget_unit,
                self.problem_id,
                self.suite_id,
                self.problem_index,
                self.problem_label,
            ):
                raise OracleSchemaError(
                    "response-curve point identity does not match its problem"
                )


@dataclass(frozen=True, slots=True)
class OracleItem:
    domain: Domain
    model_key: str
    setting: Setting
    budget_unit: BudgetUnit
    problem_id: str
    suite_id: str
    problem_index: int
    problem_label: str
    budget: int
    observed_cost: int
    reward: Literal[1]
    parse_status: Literal["parsed"]
    judge_status: Literal["judged"]
    source_run_id: str

    def __post_init__(self) -> None:
        _domain(self.domain)
        _text(self.model_key, "model_key")
        setting = _setting(self.setting)
        _budget_unit(self.budget_unit, setting)
        _problem_identity(
            problem_id=self.problem_id,
            suite_id=self.suite_id,
            problem_index=self.problem_index,
            problem_label=self.problem_label,
        )
        _int(self.budget, "budget")
        _int(self.observed_cost, "observed_cost")
        if self.reward != 1:
            raise OracleSchemaError("OracleItem reward must be 1")
        if self.parse_status != "parsed" or self.judge_status != "judged":
            raise OracleSchemaError("OracleItem must represent a judged success")
        _text(self.source_run_id, "source_run_id")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OracleItem":
        _exact_fields(value, _ORACLE_ITEM_FIELDS, "Oracle item")
        return cls(**dict(value))


_ORACLE_ITEM_FIELDS = _RESPONSE_POINT_FIELDS - {
    "mode",
    "repeat_id",
    "budget_level",
    "condition_id",
    "condition_kind",
}


@dataclass(frozen=True, slots=True)
class OracleBudgetOption:
    """One configured response-curve level valued by empirical success rate."""

    domain: Domain
    model_key: str
    setting: Setting
    budget_unit: BudgetUnit
    problem_id: str
    suite_id: str
    problem_index: int
    problem_label: str
    budget_level: int
    budget: int
    success_rate: float
    successful_repeats: int
    repeat_count: int
    source_run_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _domain(self.domain)
        _text(self.model_key, "model_key")
        setting = _setting(self.setting)
        _budget_unit(self.budget_unit, setting)
        _problem_identity(
            problem_id=self.problem_id,
            suite_id=self.suite_id,
            problem_index=self.problem_index,
            problem_label=self.problem_label,
        )
        _int(self.budget_level, "budget_level", minimum=1, maximum=6)
        _int(self.budget, "budget")
        successful = _int(self.successful_repeats, "successful_repeats")
        repeats = _int(self.repeat_count, "repeat_count", minimum=1)
        if successful > repeats:
            raise OracleSchemaError("successful_repeats cannot exceed repeat_count")
        if (
            isinstance(self.success_rate, bool)
            or not isinstance(self.success_rate, (int, float))
            or not isfinite(float(self.success_rate))
            or not 0.0 <= float(self.success_rate) <= 1.0
        ):
            raise OracleSchemaError("success_rate must be finite and within [0, 1]")
        if not isclose(float(self.success_rate), successful / repeats):
            raise OracleSchemaError(
                "success_rate must equal successful_repeats / repeat_count"
            )
        run_ids = tuple(self.source_run_ids)
        object.__setattr__(self, "source_run_ids", run_ids)
        if len(run_ids) != repeats or len(set(run_ids)) != repeats:
            raise OracleSchemaError(
                "source_run_ids must contain one unique ID per repeat"
            )
        for run_id in run_ids:
            _text(run_id, "source_run_ids")


@dataclass(frozen=True, slots=True)
class ContestProblemResult:
    domain: Domain
    model_key: str
    setting: Setting
    budget_unit: BudgetUnit
    mode: Literal["contest"]
    problem_id: str
    suite_id: str
    problem_index: int
    problem_label: str
    rho: float
    formal_contest_budget: int
    reward: int
    parse_status: ParseStatus
    judge_status: JudgeStatus
    source_run_id: str
    repeat_id: int | None = None

    def __post_init__(self) -> None:
        _domain(self.domain)
        _text(self.model_key, "model_key")
        setting = _setting(self.setting)
        _budget_unit(self.budget_unit, setting)
        _mode(self.mode, "contest")
        _problem_identity(
            problem_id=self.problem_id,
            suite_id=self.suite_id,
            problem_index=self.problem_index,
            problem_label=self.problem_label,
        )
        _rho(self.rho)
        _int(self.formal_contest_budget, "formal_contest_budget")
        reward = _reward(self.reward)
        _statuses(self.parse_status, self.judge_status, reward)
        _text(self.source_run_id, "source_run_id")
        if self.repeat_id is not None:
            _int(self.repeat_id, "repeat_id", minimum=1)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContestProblemResult":
        required = _CONTEST_RESULT_FIELDS - {"repeat_id"}
        fields = frozenset(value)
        if not required <= fields or fields - _CONTEST_RESULT_FIELDS:
            raise OracleSchemaError("contest result has invalid fields")
        return cls(**dict(value))


_CONTEST_RESULT_FIELDS = frozenset(
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
        "rho",
        "formal_contest_budget",
        "reward",
        "parse_status",
        "judge_status",
        "source_run_id",
        "repeat_id",
    }
)


@dataclass(frozen=True, slots=True)
class FormalBudgetRecord:
    domain: Domain
    model_key: str
    setting: Setting
    budget_unit: BudgetUnit
    rho: float
    formal_contest_budget: int
    response_curve_grid: tuple[int, ...]

    def __post_init__(self) -> None:
        _domain(self.domain)
        _text(self.model_key, "model_key")
        setting = _setting(self.setting)
        _budget_unit(self.budget_unit, setting)
        _rho(self.rho)
        _int(self.formal_contest_budget, "formal_contest_budget")
        grid = tuple(self.response_curve_grid)
        object.__setattr__(self, "response_curve_grid", grid)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in grid
        ):
            raise OracleSchemaError(
                "response_curve_grid must contain non-negative integers"
            )
        if tuple(sorted(grid)) != grid:
            raise OracleSchemaError("response_curve_grid must be nondecreasing")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormalBudgetRecord":
        _exact_fields(value, _FORMAL_BUDGET_FIELDS, "formal budget")
        grid = value["response_curve_grid"]
        if not isinstance(grid, list):
            raise OracleSchemaError("response_curve_grid must be a JSON array")
        return cls(
            domain=value["domain"],
            model_key=value["model_key"],
            setting=value["setting"],
            budget_unit=value["budget_unit"],
            rho=value["rho"],
            formal_contest_budget=value["formal_contest_budget"],
            response_curve_grid=tuple(grid),
        )


_FORMAL_BUDGET_FIELDS = frozenset(
    {
        "domain",
        "model_key",
        "setting",
        "budget_unit",
        "rho",
        "formal_contest_budget",
        "response_curve_grid",
    }
)


@dataclass(frozen=True, slots=True)
class EqualAllocationProblemResult:
    problem_id: str
    problem_index: int
    problem_label: str
    observed_cost: int | None
    allocated_budget: int
    reward: int
    selected_by_equal: bool
    source_run_id: str | None

    def __post_init__(self) -> None:
        _text(self.problem_id, "problem_id")
        index = _int(self.problem_index, "problem_index", minimum=1, maximum=6)
        if self.problem_label != CONTEST_LABELS[index - 1]:
            raise OracleSchemaError("problem_label does not match problem_index")
        _optional_int(self.observed_cost, "observed_cost")
        _int(self.allocated_budget, "allocated_budget")
        reward = _reward(self.reward)
        if not isinstance(self.selected_by_equal, bool):
            raise OracleSchemaError("selected_by_equal must be boolean")
        if reward != int(self.selected_by_equal):
            raise OracleSchemaError(
                "equal-allocation reward must match selected_by_equal"
            )
        if self.source_run_id is not None:
            _text(self.source_run_id, "source_run_id")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EqualAllocationProblemResult":
        _exact_fields(value, _EQUAL_PROBLEM_FIELDS, "equal-allocation problem")
        return cls(**dict(value))


_EQUAL_PROBLEM_FIELDS = frozenset(
    {
        "problem_id",
        "problem_index",
        "problem_label",
        "observed_cost",
        "allocated_budget",
        "reward",
        "selected_by_equal",
        "source_run_id",
    }
)


@dataclass(frozen=True, slots=True)
class EqualAllocationSuiteResult:
    domain: Domain
    model_key: str
    setting: Setting
    budget_unit: BudgetUnit
    mode: Literal["contest"]
    suite_id: str
    rho: float | None
    formal_contest_budget: int
    per_problem_budget: int
    equal_score: int
    problem_results: tuple[EqualAllocationProblemResult, ...]
    capacity_source: Literal["formal_contest_budget"] = "formal_contest_budget"

    def __post_init__(self) -> None:
        _domain(self.domain)
        _text(self.model_key, "model_key")
        setting = _setting(self.setting)
        _budget_unit(self.budget_unit, setting)
        _mode(self.mode, "contest")
        _text(self.suite_id, "suite_id")
        _rho(self.rho, optional=True)
        _int(self.formal_contest_budget, "formal_contest_budget")
        _int(self.per_problem_budget, "per_problem_budget")
        if self.capacity_source != "formal_contest_budget":
            raise OracleSchemaError(
                "equal-allocation capacity_source must be formal_contest_budget"
            )
        if self.per_problem_budget != self.formal_contest_budget // 6:
            raise OracleSchemaError(
                "per_problem_budget must equal floor(formal_contest_budget / 6)"
            )
        _int(self.equal_score, "equal_score", maximum=6)
        results = tuple(self.problem_results)
        object.__setattr__(self, "problem_results", results)
        _validate_six_problem_outputs(results)
        if self.equal_score != sum(row.reward for row in results):
            raise OracleSchemaError("equal_score does not match problem rewards")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EqualAllocationSuiteResult":
        _exact_fields(value, _EQUAL_SUITE_FIELDS, "equal-allocation suite")
        rows = value["problem_results"]
        if not isinstance(rows, list):
            raise OracleSchemaError("problem_results must be a JSON array")
        return cls(
            domain=value["domain"],
            model_key=value["model_key"],
            setting=value["setting"],
            budget_unit=value["budget_unit"],
            mode=value["mode"],
            suite_id=value["suite_id"],
            rho=value["rho"],
            formal_contest_budget=value["formal_contest_budget"],
            per_problem_budget=value["per_problem_budget"],
            equal_score=value["equal_score"],
            problem_results=tuple(
                EqualAllocationProblemResult.from_dict(row) for row in rows
            ),
            capacity_source=value["capacity_source"],
        )


_EQUAL_SUITE_FIELDS = frozenset(
    {
        "domain",
        "model_key",
        "setting",
        "budget_unit",
        "mode",
        "suite_id",
        "rho",
        "formal_contest_budget",
        "per_problem_budget",
        "equal_score",
        "problem_results",
        "capacity_source",
    }
)


@dataclass(frozen=True, slots=True)
class OracleProblemSelection:
    problem_id: str
    problem_index: int
    problem_label: str
    observed_cost: int | None
    reward: int
    selected_by_oracle: bool
    source_run_id: str | None

    def __post_init__(self) -> None:
        _text(self.problem_id, "problem_id")
        index = _int(self.problem_index, "problem_index", minimum=1, maximum=6)
        if self.problem_label != CONTEST_LABELS[index - 1]:
            raise OracleSchemaError("problem_label does not match problem_index")
        _optional_int(self.observed_cost, "observed_cost")
        reward = _reward(self.reward)
        if not isinstance(self.selected_by_oracle, bool):
            raise OracleSchemaError("selected_by_oracle must be boolean")
        if reward != int(self.selected_by_oracle):
            raise OracleSchemaError("Oracle reward must match selected_by_oracle")
        if self.selected_by_oracle and self.observed_cost is None:
            raise OracleSchemaError(
                "an Oracle-selected problem requires an observed success cost"
            )
        if self.source_run_id is not None:
            _text(self.source_run_id, "source_run_id")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OracleProblemSelection":
        _exact_fields(value, _ORACLE_SELECTION_FIELDS, "Oracle selection")
        return cls(**dict(value))


_ORACLE_SELECTION_FIELDS = frozenset(
    {
        "problem_id",
        "problem_index",
        "problem_label",
        "observed_cost",
        "reward",
        "selected_by_oracle",
        "source_run_id",
    }
)


@dataclass(frozen=True, slots=True)
class OracleSuiteResult:
    domain: Domain
    model_key: str
    setting: Setting
    budget_unit: BudgetUnit
    mode: Literal["contest"]
    suite_id: str
    rho: float | None
    formal_contest_budget: int
    oracle_score: int
    total_selected_cost: int
    problem_selections: tuple[OracleProblemSelection, ...]
    capacity_source: Literal["formal_contest_budget"] = "formal_contest_budget"

    def __post_init__(self) -> None:
        _domain(self.domain)
        _text(self.model_key, "model_key")
        setting = _setting(self.setting)
        _budget_unit(self.budget_unit, setting)
        _mode(self.mode, "contest")
        _text(self.suite_id, "suite_id")
        _rho(self.rho, optional=True)
        _int(self.formal_contest_budget, "formal_contest_budget")
        if self.capacity_source != "formal_contest_budget":
            raise OracleSchemaError(
                "Oracle capacity_source must be formal_contest_budget"
            )
        _int(self.oracle_score, "oracle_score", maximum=6)
        _int(self.total_selected_cost, "total_selected_cost")
        rows = tuple(self.problem_selections)
        object.__setattr__(self, "problem_selections", rows)
        _validate_six_problem_outputs(rows)
        if self.oracle_score != sum(row.reward for row in rows):
            raise OracleSchemaError("oracle_score does not match selections")
        expected_cost = sum(
            row.observed_cost or 0 for row in rows if row.selected_by_oracle
        )
        if self.total_selected_cost != expected_cost:
            raise OracleSchemaError("total_selected_cost does not match selections")
        if self.total_selected_cost > self.formal_contest_budget:
            raise OracleSchemaError("Oracle selection exceeds formal contest budget")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OracleSuiteResult":
        _exact_fields(value, _ORACLE_SUITE_FIELDS, "Oracle suite")
        rows = value["problem_selections"]
        if not isinstance(rows, list):
            raise OracleSchemaError("problem_selections must be a JSON array")
        return cls(
            domain=value["domain"],
            model_key=value["model_key"],
            setting=value["setting"],
            budget_unit=value["budget_unit"],
            mode=value["mode"],
            suite_id=value["suite_id"],
            rho=value["rho"],
            formal_contest_budget=value["formal_contest_budget"],
            oracle_score=value["oracle_score"],
            total_selected_cost=value["total_selected_cost"],
            problem_selections=tuple(
                OracleProblemSelection.from_dict(row) for row in rows
            ),
            capacity_source=value["capacity_source"],
        )


_ORACLE_SUITE_FIELDS = frozenset(
    {
        "domain",
        "model_key",
        "setting",
        "budget_unit",
        "mode",
        "suite_id",
        "rho",
        "formal_contest_budget",
        "oracle_score",
        "total_selected_cost",
        "problem_selections",
        "capacity_source",
    }
)


def _validate_six_problem_outputs(rows: tuple[Any, ...]) -> None:
    if len(rows) != 6:
        raise OracleSchemaError("suite output requires exactly six problem rows")
    if tuple(row.problem_index for row in rows) != (1, 2, 3, 4, 5, 6):
        raise OracleSchemaError("suite output must be ordered by positions 1 through 6")
    if len({row.problem_id for row in rows}) != 6:
        raise OracleSchemaError("suite output problem IDs must be unique")


@dataclass(frozen=True, slots=True)
class GapSummary:
    domain: Domain
    model_key: str
    setting: Setting
    budget_unit: BudgetUnit
    mode: Literal["contest"]
    rho: float
    formal_contest_budget: int
    suite_count: int
    contest_total: int
    equal_total: int
    oracle_total: int
    contest_score: float
    equal_score: float
    oracle_score: float
    delta_rr: float
    gap_ratio: float | None

    def __post_init__(self) -> None:
        _domain(self.domain)
        _text(self.model_key, "model_key")
        setting = _setting(self.setting)
        _budget_unit(self.budget_unit, setting)
        _mode(self.mode, "contest")
        _rho(self.rho)
        _int(self.formal_contest_budget, "formal_contest_budget")
        _int(self.suite_count, "suite_count", minimum=1)
        for field in ("contest_total", "equal_total", "oracle_total"):
            _int(getattr(self, field), field)
        for field in (
            "contest_score",
            "equal_score",
            "oracle_score",
            "delta_rr",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OracleSchemaError(f"{field} must be numeric")
            if not isfinite(float(value)):
                raise OracleSchemaError(f"{field} must be finite")
        if self.gap_ratio is not None and (
            isinstance(self.gap_ratio, bool)
            or not isinstance(self.gap_ratio, (int, float))
            or not isfinite(float(self.gap_ratio))
        ):
            raise OracleSchemaError("gap_ratio must be finite or null")
        maximum_total = self.suite_count * 6
        if any(
            total > maximum_total
            for total in (
                self.contest_total,
                self.equal_total,
                self.oracle_total,
            )
        ):
            raise OracleSchemaError(
                "aggregate totals cannot exceed six rewards per suite"
            )
        expected_scores = (
            self.contest_total / self.suite_count,
            self.equal_total / self.suite_count,
            self.oracle_total / self.suite_count,
        )
        actual_scores = (
            self.contest_score,
            self.equal_score,
            self.oracle_score,
        )
        if any(
            not isclose(float(actual), expected)
            for actual, expected in zip(actual_scores, expected_scores, strict=True)
        ):
            raise OracleSchemaError("aggregate scores do not match totals")
        expected_delta = self.oracle_score - self.contest_score
        if not isclose(float(self.delta_rr), expected_delta):
            raise OracleSchemaError("delta_rr must equal oracle - contest")
        if self.oracle_score > 0:
            expected_ratio = expected_delta / self.oracle_score
            if self.gap_ratio is None or not isclose(
                float(self.gap_ratio), expected_ratio
            ):
                raise OracleSchemaError("gap_ratio must equal delta_rr / oracle_score")
        elif self.gap_ratio is not None:
            raise OracleSchemaError("gap_ratio must be null when oracle_score is zero")


__all__ = [
    "ContestProblemResult",
    "EqualAllocationProblemResult",
    "EqualAllocationSuiteResult",
    "FormalBudgetRecord",
    "GapSummary",
    "OracleItem",
    "OracleProblemSelection",
    "OracleSchemaError",
    "OracleSuiteResult",
    "ProblemResponseCurve",
    "ResponseCurvePoint",
]
