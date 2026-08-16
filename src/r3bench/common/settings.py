"""Setting-level contracts shared by Tool-Free and Agentic evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias


EvaluationSetting: TypeAlias = Literal["tool_free", "agentic"]
BudgetUnit: TypeAlias = Literal["output_tokens", "counted_actions"]
RuntimeKind: TypeAlias = Literal["provider_generation", "agentic_runtime"]

EVALUATION_SETTINGS = frozenset({"tool_free", "agentic"})
BUDGET_UNITS = frozenset({"output_tokens", "counted_actions"})
RUNTIME_KINDS = frozenset({"provider_generation", "agentic_runtime"})

_SETTING_CONTRACT = {
    "tool_free": ("output_tokens", "provider_generation"),
    "agentic": ("counted_actions", "agentic_runtime"),
}


class SettingContractError(ValueError):
    """Raised when a setting, runtime, and resource unit are inconsistent."""


def expected_budget_unit(setting: EvaluationSetting) -> BudgetUnit:
    if setting not in EVALUATION_SETTINGS:
        raise SettingContractError(f"unsupported evaluation setting: {setting!r}")
    return _SETTING_CONTRACT[setting][0]  # type: ignore[return-value]


def expected_runtime_kind(setting: EvaluationSetting) -> RuntimeKind:
    if setting not in EVALUATION_SETTINGS:
        raise SettingContractError(f"unsupported evaluation setting: {setting!r}")
    return _SETTING_CONTRACT[setting][1]  # type: ignore[return-value]


def validate_budget_unit(
    setting: EvaluationSetting,
    budget_unit: BudgetUnit,
) -> BudgetUnit:
    expected = expected_budget_unit(setting)
    if budget_unit != expected:
        raise SettingContractError(
            f"{setting} requires budget_unit={expected!r}, got {budget_unit!r}"
        )
    return budget_unit


@dataclass(frozen=True, slots=True)
class SettingConfig:
    setting: EvaluationSetting
    budget_unit: BudgetUnit
    runtime_kind: RuntimeKind

    def __post_init__(self) -> None:
        if self.setting not in EVALUATION_SETTINGS:
            raise SettingContractError(
                f"unsupported evaluation setting: {self.setting!r}"
            )
        if self.budget_unit not in BUDGET_UNITS:
            raise SettingContractError(f"unsupported budget unit: {self.budget_unit!r}")
        if self.runtime_kind not in RUNTIME_KINDS:
            raise SettingContractError(f"unsupported runtime kind: {self.runtime_kind!r}")
        expected_unit, expected_runtime = _SETTING_CONTRACT[self.setting]
        if self.budget_unit != expected_unit:
            raise SettingContractError(
                f"{self.setting} requires budget_unit={expected_unit!r}"
            )
        if self.runtime_kind != expected_runtime:
            raise SettingContractError(
                f"{self.setting} requires runtime_kind={expected_runtime!r}"
            )


__all__ = [
    "BUDGET_UNITS",
    "EVALUATION_SETTINGS",
    "RUNTIME_KINDS",
    "BudgetUnit",
    "EvaluationSetting",
    "RuntimeKind",
    "SettingConfig",
    "SettingContractError",
    "expected_budget_unit",
    "expected_runtime_kind",
    "validate_budget_unit",
]
