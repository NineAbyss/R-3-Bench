"""Setting-aware budget resolution for protocol-reproducible evaluations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from r3bench.benchmark import BenchmarkCell, expand_cells
from r3bench.common.experiment import BudgetConfig, ExperimentConfig
from r3bench.common.settings import BudgetUnit, EvaluationSetting, expected_budget_unit


_CONDITION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")


class BudgetResolutionError(ValueError):
    """Raised when an evaluation has no unambiguous executable budget."""


@dataclass(frozen=True, slots=True)
class OfficialBudgetProfile:
    profile_id: str
    model_key: str
    domain: str
    setting: EvaluationSetting
    role: str
    budget_unit: BudgetUnit
    budget_value: int | None
    budget_grid: tuple[int, ...]
    rho: float | None
    resource_policy: str
    provider_safety_cap: int | None


@dataclass(frozen=True, slots=True)
class ResolvedBudget:
    value: int
    unit: BudgetUnit
    source: str
    condition_id: str
    condition_kind: str
    profile_id: str | None = None
    rho: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _checked_nonnegative(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BudgetResolutionError(f"{field} must be a non-negative integer")
    return value


def _checked_condition_id(value: str) -> str:
    if not isinstance(value, str) or not _CONDITION_ID.fullmatch(value):
        raise BudgetResolutionError(
            "condition_id must use only letters, digits, '.', ':', '_' or '-'"
        )
    return value


def _profile_from_cell(cell: BenchmarkCell) -> OfficialBudgetProfile:
    raw = dict(cell.budget)
    unit = expected_budget_unit(cell.setting)  # type: ignore[arg-type]
    value_field = "output_tokens" if unit == "output_tokens" else "counted_actions"
    value = raw.get(value_field)
    if value is not None:
        value = _checked_nonnegative(value, f"{cell.cell_id}.{value_field}")
    grid_value = raw.get("grid", [])
    if not isinstance(grid_value, list):
        raise BudgetResolutionError(f"{cell.cell_id}.grid must be an array")
    grid = tuple(
        _checked_nonnegative(item, f"{cell.cell_id}.grid") for item in grid_value
    )
    if tuple(sorted(grid)) != grid:
        raise BudgetResolutionError(f"{cell.cell_id}.grid must be nondecreasing")
    safety = raw.get("provider_safety_cap")
    if safety is not None:
        safety = _checked_nonnegative(safety, f"{cell.cell_id}.provider_safety_cap")
    rho = raw.get("rho")
    if rho is not None:
        rho = float(rho)
    return OfficialBudgetProfile(
        profile_id=cell.cell_id,
        model_key=cell.model,
        domain=cell.domain,
        setting=cell.setting,  # type: ignore[arg-type]
        role=cell.role,
        budget_unit=unit,
        budget_value=value,
        budget_grid=grid,
        rho=rho,
        resource_policy=str(raw.get("resource_policy", unit)),
        provider_safety_cap=safety,
    )


def load_official_budget_profiles() -> Mapping[str, OfficialBudgetProfile]:
    """Expose paper-reference budgets as optional, named convenience profiles."""

    profiles = {cell.cell_id: _profile_from_cell(cell) for cell in expand_cells()}
    return dict(sorted(profiles.items()))


def resolve_official_budget_profile(
    profile_id: str,
    *,
    setting: EvaluationSetting,
    domain: str | None = None,
    model_key: str | None = None,
) -> OfficialBudgetProfile:
    try:
        profile = load_official_budget_profiles()[profile_id]
    except KeyError as exc:
        raise BudgetResolutionError(f"unknown budget profile: {profile_id}") from exc
    if profile.setting != setting:
        raise BudgetResolutionError(
            "budget profile does not match the selected setting"
        )
    if domain is not None and profile.domain != domain:
        raise BudgetResolutionError("budget profile does not match the selected domain")
    if model_key is not None and profile.model_key != model_key:
        raise BudgetResolutionError("budget profile does not match the selected model")
    return profile


def resolve_budget(
    *,
    setting: EvaluationSetting,
    explicit_value: int | None,
    config_value: int | None,
    profile_id: str | None,
    condition_id: str | None = None,
    domain: str | None = None,
    model_key: str | None = None,
) -> ResolvedBudget:
    """Resolve CLI > config > official profile, recording the winning source."""

    unit = expected_budget_unit(setting)
    profile = (
        resolve_official_budget_profile(
            profile_id,
            setting=setting,
            domain=domain,
            model_key=model_key,
        )
        if profile_id is not None
        else None
    )
    minimum = 0
    if explicit_value is not None:
        value = _checked_nonnegative(explicit_value, "explicit budget")
        source = "cli_explicit"
        kind = "custom"
        selected_profile = None
        rho = None
    elif config_value is not None:
        value = _checked_nonnegative(config_value, "config budget")
        source = "experiment_config"
        kind = "custom"
        selected_profile = None
        rho = None
    elif profile is not None:
        if profile.budget_value is None:
            raise BudgetResolutionError(
                "selected profile has no scalar budget; use a response-curve grid "
                "or explicitly select unbounded calibration"
            )
        value = profile.budget_value
        source = "official_profile"
        kind = "official_profile"
        selected_profile = profile.profile_id
        rho = profile.rho
    else:
        raise BudgetResolutionError(
            "budget missing: provide an explicit budget, a config budget, or "
            "--budget-profile"
        )
    if value < minimum:
        raise BudgetResolutionError(
            f"{setting} execution budget must be at least {minimum}"
        )
    default_condition = (
        selected_profile if selected_profile is not None else f"custom_{unit}_{value}"
    )
    return ResolvedBudget(
        value=value,
        unit=unit,
        source=source,
        condition_id=_checked_condition_id(condition_id or default_condition),
        condition_kind=kind,
        profile_id=selected_profile,
        rho=rho,
    )


def resolve_budget_grid(
    *,
    setting: EvaluationSetting,
    explicit_values: Iterable[int] | None,
    profile_id: str | None,
    domain: str,
    model_key: str,
) -> tuple[int, ...]:
    if explicit_values is not None:
        values = tuple(
            _checked_nonnegative(value, "budget grid") for value in explicit_values
        )
    elif profile_id is not None:
        profile = resolve_official_budget_profile(
            profile_id,
            setting=setting,
            domain=domain,
            model_key=model_key,
        )
        values = profile.budget_grid
    else:
        raise BudgetResolutionError(
            "response curve requires --budget-grid or an official grid profile"
        )
    if not values or tuple(sorted(values)) != values:
        raise BudgetResolutionError("budget grid must be non-empty and nondecreasing")
    return values


def apply_budget_to_experiment(
    config: ExperimentConfig,
    resolution: ResolvedBudget,
) -> ExperimentConfig:
    if config.setting != "tool_free" or resolution.unit != "output_tokens":
        raise BudgetResolutionError(
            "ExperimentConfig budget overrides apply only to Tool-Free output tokens"
        )
    return replace(
        config,
        budget=BudgetConfig(
            max_tokens=resolution.value,
            temperature=config.budget.temperature,
            top_p=config.budget.top_p,
            action_budget=config.budget.action_budget,
        ),
    )


def annotate_run_summary(
    output_dir: str | Path,
    resolution: ResolvedBudget,
    *,
    protocol: Mapping[str, Any] | None = None,
) -> None:
    path = Path(output_dir) / "run_summary.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BudgetResolutionError("run_summary.json must contain an object")
    value["budget"] = resolution.to_dict()
    if protocol is not None:
        value["protocol"] = dict(protocol)
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "BudgetResolutionError",
    "OfficialBudgetProfile",
    "ResolvedBudget",
    "annotate_run_summary",
    "apply_budget_to_experiment",
    "load_official_budget_profiles",
    "resolve_budget",
    "resolve_budget_grid",
    "resolve_official_budget_profile",
]
