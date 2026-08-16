"""Provider-neutral experiment configuration for future evaluator runners."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from string import Formatter
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

from r3bench.common.config import ConfigError, load_config
from r3bench.common.schema import Domain


Mode: TypeAlias = Literal["single_problem", "contest"]
Visibility: TypeAlias = Literal["hidden", "labeled"]
Setting: TypeAlias = Literal["tool_free", "agentic"]
Stage: TypeAlias = Literal["one_stage", "stage1", "stage2"]
PresentationOrder: TypeAlias = Literal["canonical", "formal_seeded"]

_MODES = frozenset({"single_problem", "contest"})
_VISIBILITIES = frozenset({"hidden", "labeled"})
_SETTINGS = frozenset({"tool_free", "agentic"})
_STAGES = frozenset({"one_stage", "stage1", "stage2"})
_PRESENTATION_ORDERS = frozenset({"canonical", "formal_seeded"})
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value


def _relative_path(value: object, field: str) -> str:
    text = _nonempty(value, field)
    if Path(text).is_absolute():
        raise ConfigError(
            f"{field} must be a relative public path or repository identifier"
        )
    return text


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    name: str
    model: str
    api_key_env: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.name, "provider.name")
        _nonempty(self.model, "provider.model")
        if self.api_key_env is not None and not _ENV_NAME.fullmatch(self.api_key_env):
            raise ConfigError(
                "provider.api_key_env must be an environment-variable name"
            )


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    max_tokens: int | None
    temperature: float | None = 0.0
    top_p: float | None = None
    action_budget: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is not None:
            if isinstance(self.max_tokens, bool) or not isinstance(
                self.max_tokens, int
            ):
                raise ConfigError("budget.max_tokens must be an integer or null")
            if self.max_tokens < 0:
                raise ConfigError(
                    "budget.max_tokens must be non-negative when provided"
                )
        if self.temperature is not None:
            if not isinstance(self.temperature, (int, float)) or isinstance(
                self.temperature, bool
            ):
                raise ConfigError("budget.temperature must be numeric or null")
            if not 0 <= float(self.temperature) <= 2:
                raise ConfigError("budget.temperature must be between 0 and 2")
        if self.top_p is not None:
            if not isinstance(self.top_p, (int, float)) or isinstance(self.top_p, bool):
                raise ConfigError("budget.top_p must be numeric or null")
            if not 0 <= float(self.top_p) <= 1:
                raise ConfigError("budget.top_p must be between 0 and 1")
        if self.action_budget is not None:
            if isinstance(self.action_budget, bool) or not isinstance(
                self.action_budget, int
            ):
                raise ConfigError("budget.action_budget must be an integer or null")
            if self.action_budget <= 0:
                raise ConfigError("budget.action_budget must be positive")


@dataclass(frozen=True, slots=True)
class PromptConfig:
    template_path: str
    system_template_path: str | None = None

    def __post_init__(self) -> None:
        _relative_path(self.template_path, "prompt.template_path")
        if self.system_template_path is not None:
            _relative_path(
                self.system_template_path,
                "prompt.system_template_path",
            )


@dataclass(frozen=True, slots=True)
class JudgeConfig:
    profile_name: str

    def __post_init__(self) -> None:
        _nonempty(self.profile_name, "judge.profile_name")


@dataclass(frozen=True, slots=True)
class PresentationConfig:
    """Contest presentation policy applied after canonical data loading."""

    order: PresentationOrder = "canonical"
    seed: int | None = None
    seed_suite_id_template: str | None = None

    def __post_init__(self) -> None:
        if self.order not in _PRESENTATION_ORDERS:
            raise ConfigError(f"unsupported presentation.order: {self.order!r}")
        if self.order == "canonical":
            if self.seed is not None or self.seed_suite_id_template is not None:
                raise ConfigError(
                    "canonical presentation cannot set a seed or seed_suite_id_template"
                )
            return
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ConfigError(
                "formal_seeded presentation requires a non-negative integer seed"
            )
        if self.seed_suite_id_template is None:
            return
        template = _nonempty(
            self.seed_suite_id_template,
            "presentation.seed_suite_id_template",
        )
        fields = {
            field_name
            for _, field_name, format_spec, conversion in Formatter().parse(template)
            if field_name is not None
            and not format_spec.startswith("{")
            and conversion is None
        }
        if not fields or not fields.issubset({"suite_index", "suite_number"}):
            raise ConfigError(
                "presentation.seed_suite_id_template may use only "
                "{suite_index} or {suite_number}"
            )
        try:
            template.format(suite_index=0, suite_number=1)
        except (IndexError, KeyError, ValueError) as exc:
            raise ConfigError("invalid presentation.seed_suite_id_template") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "seed": self.seed,
            "seed_suite_id_template": self.seed_suite_id_template,
        }


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str
    domain: Domain
    mode: Mode
    visibility: Visibility
    setting: Setting
    stage: Stage
    data_source: str
    provider: ProviderConfig
    budget: BudgetConfig
    prompt: PromptConfig
    judge: JudgeConfig
    split: str = "test"
    strict_data: bool = True
    transport: Mapping[str, Any] | None = None
    presentation: PresentationConfig = field(default_factory=PresentationConfig)

    def __post_init__(self) -> None:
        _nonempty(self.name, "name")
        if self.domain not in {"coding", "math", "abstract_reasoning"}:
            raise ConfigError(f"unsupported domain: {self.domain!r}")
        if self.mode not in _MODES:
            raise ConfigError(f"unsupported mode: {self.mode!r}")
        if self.visibility not in _VISIBILITIES:
            raise ConfigError(f"unsupported visibility: {self.visibility!r}")
        if self.setting not in _SETTINGS:
            raise ConfigError(f"unsupported setting: {self.setting!r}")
        if self.stage not in _STAGES:
            raise ConfigError(f"unsupported stage: {self.stage!r}")
        _relative_path(self.data_source, "data_source")
        if self.split != "test":
            raise ConfigError("the frozen public release supports split='test' only")
        if not isinstance(self.strict_data, bool):
            raise ConfigError("strict_data must be boolean")
        if not isinstance(self.presentation, PresentationConfig):
            raise ConfigError("presentation must be a PresentationConfig")
        if self.mode != "contest" and self.presentation.order != "canonical":
            raise ConfigError(
                "formal_seeded presentation is supported only for contest mode"
            )
        if self.setting != "tool_free" and self.presentation.order != "canonical":
            raise ConfigError(
                "formal_seeded presentation is supported only for tool_free"
            )
        if self.transport is not None:
            from r3bench.common.profile_registry import (
                ProfileError,
                validate_transport_mapping,
            )

            try:
                checked = validate_transport_mapping(
                    self.transport, path="experiment.transport"
                )
            except ProfileError as exc:
                raise ConfigError(str(exc)) from exc
            object.__setattr__(self, "transport", MappingProxyType(checked))

    @property
    def model_name(self) -> str:
        return self.provider.model

    @property
    def max_tokens(self) -> int | None:
        return self.budget.max_tokens

    @property
    def prompt_template_path(self) -> str:
        return self.prompt.template_path

    @property
    def system_prompt_template_path(self) -> str | None:
        return self.prompt.system_template_path

    @property
    def judge_profile_name(self) -> str:
        return self.judge.profile_name

    @property
    def transport_overrides(self) -> Mapping[str, Any] | None:
        return self.transport

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        try:
            provider = value["provider"]
            budget = value["budget"]
            prompt = value["prompt"]
            judge = value["judge"]
            presentation = value.get("presentation", {})
            if not all(
                isinstance(item, Mapping)
                for item in (provider, budget, prompt, judge, presentation)
            ):
                raise ConfigError(
                    "provider, budget, prompt, judge, and presentation must be mappings"
                )
            return cls(
                name=value["name"],
                domain=value["domain"],
                mode=value["mode"],
                visibility=value["visibility"],
                setting=value["setting"],
                stage=value["stage"],
                data_source=value["data_source"],
                provider=ProviderConfig(**provider),
                budget=BudgetConfig(**budget),
                prompt=PromptConfig(**prompt),
                judge=JudgeConfig(**judge),
                split=value.get("split", "test"),
                strict_data=value.get("strict_data", True),
                transport=value.get("transport"),
                presentation=PresentationConfig(**presentation),
            )
        except KeyError as exc:
            raise ConfigError(
                f"missing experiment configuration field: {exc.args[0]}"
            ) from exc
        except TypeError as exc:
            raise ConfigError(
                f"invalid experiment configuration fields: {exc}"
            ) from exc

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_mapping(load_config(path))


__all__ = [
    "BudgetConfig",
    "ExperimentConfig",
    "JudgeConfig",
    "PresentationConfig",
    "PromptConfig",
    "ProviderConfig",
]
