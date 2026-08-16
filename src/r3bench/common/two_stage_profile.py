"""Safe, stage-specific Tool-Free profile resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Mapping

from r3bench.common.config import ConfigError, load_config
from r3bench.common.experiment import (
    BudgetConfig,
    ExperimentConfig,
    PromptConfig,
    ProviderConfig,
)
from r3bench.common.profile_registry import ModelProfile

BudgetAccounting = Literal["stage1_only_with_practical_stage2_cap"]
HandoffChannel = Literal["reasoning_content", "visible_output"]
Stage2PromptAssembly = Literal[
    "coding_reasoning_visible_trace",
    "coding_visible_output_only",
    "reasoning_visible_trace",
]

_HANDOFF_CHANNELS = frozenset({"reasoning_content", "visible_output"})
_STAGE2_ASSEMBLIES = frozenset(
    {
        "coding_reasoning_visible_trace",
        "coding_visible_output_only",
        "reasoning_visible_trace",
    }
)


@dataclass(frozen=True, slots=True)
class TwoStageProtocol:
    """Formal Stage 1-to-Stage 2 data and prompt assembly contract."""

    handoff_channels: tuple[HandoffChannel, ...]
    prompt_assembly: Stage2PromptAssembly
    include_original_problems: bool

    def __post_init__(self) -> None:
        if not self.handoff_channels:
            raise ConfigError("two-stage handoff requires at least one channel")
        if len(set(self.handoff_channels)) != len(self.handoff_channels):
            raise ConfigError("two-stage handoff channels must be unique")
        if any(channel not in _HANDOFF_CHANNELS for channel in self.handoff_channels):
            raise ConfigError("two-stage handoff channel is unsupported")
        if self.prompt_assembly not in _STAGE2_ASSEMBLIES:
            raise ConfigError("Stage 2 prompt assembly is unsupported")
        if (
            self.prompt_assembly in {
                "coding_reasoning_visible_trace",
                "coding_visible_output_only",
                "reasoning_visible_trace",
            }
            and self.include_original_problems
        ):
            raise ConfigError("Stage 2 cannot include original problems")
        if (
            self.prompt_assembly == "coding_visible_output_only"
            and self.handoff_channels != ("visible_output",)
        ):
            raise ConfigError(
                "Coding visible-only Stage 2 must select only visible_output"
            )
        if (
            self.prompt_assembly == "reasoning_visible_trace"
            and self.handoff_channels
            != ("reasoning_content", "visible_output")
        ):
            raise ConfigError(
                "reasoning/visible Stage 2 must select both handoff channels"
            )


@dataclass(frozen=True, slots=True)
class TwoStageProfile:
    profile_id: str
    applicable_models: tuple[str, ...]
    applicable_domains: tuple[str, ...]
    stage1_model_key: str
    stage2_model_key: str
    stage1_thinking_enabled: bool
    stage2_thinking_enabled: bool
    stage2_protocol: TwoStageProtocol
    budget_accounting: BudgetAccounting
    stage2_practical_cap: int | None
    provenance_source: str
    status: str
    notes: str


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    if Path(value).is_absolute() or value.startswith(".env"):
        raise ConfigError(f"{field} contains a private path")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a non-empty array")
    return tuple(_text(item, field) for item in value)


def _handoff_channels(
    value: object,
    field: str,
) -> tuple[HandoffChannel, ...]:
    channels = _string_tuple(value, field)
    if any(channel not in _HANDOFF_CHANNELS for channel in channels):
        raise ConfigError(f"{field} contains an unsupported channel")
    return channels  # type: ignore[return-value]


def load_two_stage_profiles(path: str | Path) -> dict[str, TwoStageProfile]:
    document = load_config(path)
    rows = document.get("profiles")
    if not isinstance(rows, Mapping) or not rows:
        raise ConfigError("two-stage profile registry requires profiles")
    result: dict[str, TwoStageProfile] = {}
    for key, raw in rows.items():
        if not isinstance(raw, Mapping):
            raise ConfigError(f"two-stage profile {key!r} must be an object")
        profile_id = _text(raw.get("profile_id"), f"{key}.profile_id")
        if key != profile_id:
            raise ConfigError("two-stage profile key and profile_id must match")
        accounting = raw.get("budget_accounting")
        if accounting != "stage1_only_with_practical_stage2_cap":
            raise ConfigError(f"{key}.budget_accounting is unsupported")
        practical_cap = raw.get("stage2_practical_cap")
        if practical_cap is not None and (
            isinstance(practical_cap, bool)
            or not isinstance(practical_cap, int)
            or practical_cap <= 0
        ):
            raise ConfigError(
                f"{key}.stage2_practical_cap must be positive or null"
            )
        stage1_thinking = raw.get("stage1_thinking_enabled")
        stage2_thinking = raw.get("stage2_thinking_enabled")
        if not isinstance(stage1_thinking, bool) or not isinstance(
            stage2_thinking, bool
        ):
            raise ConfigError(f"{key} thinking flags must be boolean")
        prompt_assembly = raw.get("stage2_prompt_assembly")
        if prompt_assembly not in _STAGE2_ASSEMBLIES:
            raise ConfigError(f"{key}.stage2_prompt_assembly is unsupported")
        include_original = raw.get("stage2_include_original_problems")
        if not isinstance(include_original, bool):
            raise ConfigError(
                f"{key}.stage2_include_original_problems must be boolean"
            )
        profile = TwoStageProfile(
            profile_id=profile_id,
            applicable_models=_string_tuple(
                raw.get("applicable_models"), f"{key}.applicable_models"
            ),
            applicable_domains=_string_tuple(
                raw.get("applicable_domains"), f"{key}.applicable_domains"
            ),
            stage1_model_key=_text(
                raw.get("stage1_model_key"), f"{key}.stage1_model_key"
            ),
            stage2_model_key=_text(
                raw.get("stage2_model_key"), f"{key}.stage2_model_key"
            ),
            stage1_thinking_enabled=stage1_thinking,
            stage2_thinking_enabled=stage2_thinking,
            stage2_protocol=TwoStageProtocol(
                handoff_channels=_handoff_channels(
                    raw.get("stage2_handoff_channels"),
                    f"{key}.stage2_handoff_channels",
                ),
                prompt_assembly=prompt_assembly,
                include_original_problems=include_original,
            ),
            budget_accounting=accounting,
            stage2_practical_cap=practical_cap,
            provenance_source=_text(
                raw.get("provenance_source"), f"{key}.provenance_source"
            ),
            status=_text(raw.get("status"), f"{key}.status"),
            notes=_text(raw.get("notes"), f"{key}.notes"),
        )
        if "coding" in profile.applicable_domains:
            if profile.stage2_protocol.prompt_assembly not in {
                "coding_reasoning_visible_trace",
                "coding_visible_output_only",
            }:
                raise ConfigError(f"{key} Coding profile has invalid assembly")
        if {"math", "abstract_reasoning"} & set(profile.applicable_domains):
            if (
                profile.stage2_protocol.prompt_assembly
                != "reasoning_visible_trace"
            ):
                raise ConfigError(
                    f"{key} Math/AR profile must use trace-only assembly"
                )
        result[key] = profile
    return result


def resolve_two_stage_profile(
    profile_id: str,
    profiles: Mapping[str, TwoStageProfile],
    *,
    model_key: str,
    domain: str,
) -> TwoStageProfile:
    try:
        profile = profiles[profile_id]
    except KeyError as exc:
        raise ConfigError(f"unknown two-stage profile: {profile_id}") from exc
    if model_key not in profile.applicable_models:
        raise ConfigError("two-stage profile does not apply to the selected model")
    if domain not in profile.applicable_domains:
        raise ConfigError("two-stage profile does not apply to the selected domain")
    return profile


def default_two_stage_protocol(domain: str) -> TwoStageProtocol:
    """Return the formal domain protocol for synthetic/offline test configs."""

    if domain == "coding":
        return TwoStageProtocol(
            handoff_channels=("reasoning_content", "visible_output"),
            prompt_assembly="coding_reasoning_visible_trace",
            include_original_problems=False,
        )
    if domain in {"math", "abstract_reasoning"}:
        return TwoStageProtocol(
            handoff_channels=("reasoning_content", "visible_output"),
            prompt_assembly="reasoning_visible_trace",
            include_original_problems=False,
        )
    raise ConfigError(f"unsupported two-stage domain: {domain}")


def stage_token_caps(
    profile: TwoStageProfile,
    *,
    stage1_budget: int,
) -> tuple[int, int | None]:
    """Return the Stage 1 budget and practical Stage 2 cap."""

    if stage1_budget < 0:
        raise ConfigError("two-stage budget must be non-negative")
    return stage1_budget, profile.stage2_practical_cap


def _sampling_value(value: float | None | str, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise ConfigError(f"{field} is unresolved")


def derive_profiled_two_stage_configs(
    base: ExperimentConfig,
    profile: TwoStageProfile,
    model_profiles: Mapping[str, ModelProfile],
) -> tuple[ExperimentConfig, ExperimentConfig]:
    """Derive stage configs without creating model-specific runner branches."""

    try:
        stage1_model = model_profiles[profile.stage1_model_key]
        stage2_model = model_profiles[profile.stage2_model_key]
    except KeyError as exc:
        raise ConfigError("two-stage profile references an unknown model") from exc
    if base.max_tokens is None:
        raise ConfigError("Stage 1 requires an explicit non-negative token budget")
    stage1_cap, stage2_cap = stage_token_caps(
        profile,
        stage1_budget=base.max_tokens,
    )
    suffix = "contest" if base.mode == "contest" else "single"
    stage2_template = f"prompts/{base.domain}/{suffix}_stage2.txt"
    stage1_system = (
        f"prompts/{base.domain}/{suffix}_stage1_system.txt"
        if base.domain in {"math", "abstract_reasoning"}
        else None
    )
    stage2_system = (
        f"prompts/{base.domain}/{suffix}_stage2_system.txt"
        if base.domain in {"math", "abstract_reasoning"}
        else None
    )
    if profile.stage2_protocol.prompt_assembly == "coding_visible_output_only":
        stage2_template = f"prompts/coding/{suffix}_stage2_visible_only.txt"
    stage1 = replace(
        base,
        name=f"{base.name}_stage1",
        stage="stage1",
        provider=ProviderConfig(
            name=stage1_model.provider_profile,
            model=stage1_model.model_key,
            api_key_env=stage1_model.api_key_env,
        ),
        budget=BudgetConfig(
            max_tokens=stage1_cap,
            temperature=_sampling_value(
                stage1_model.temperature, "Stage 1 temperature"
            ),
            top_p=_sampling_value(stage1_model.top_p, "Stage 1 top_p"),
        ),
        prompt=PromptConfig(
            f"prompts/{base.domain}/{suffix}_stage1.txt",
            stage1_system,
        ),
    )
    stage2 = replace(
        base,
        name=f"{base.name}_stage2",
        stage="stage2",
        provider=ProviderConfig(
            name=stage2_model.provider_profile,
            model=stage2_model.model_key,
            api_key_env=stage2_model.api_key_env,
        ),
        budget=BudgetConfig(
            max_tokens=stage2_cap,
            temperature=_sampling_value(
                stage2_model.temperature, "Stage 2 temperature"
            ),
            top_p=_sampling_value(stage2_model.top_p, "Stage 2 top_p"),
        ),
        prompt=PromptConfig(stage2_template, stage2_system),
    )
    return stage1, stage2


def apply_offline_two_stage_budget(
    stage1: ExperimentConfig,
    stage2: ExperimentConfig,
    *,
    stage1_budget: int,
    profile: TwoStageProfile | None,
) -> tuple[
    ExperimentConfig,
    ExperimentConfig,
    TwoStageProtocol | None,
]:
    """Apply Stage 1 budgeting without changing an offline provider binding."""

    if profile is None:
        return (
            replace(
                stage1,
                budget=replace(stage1.budget, max_tokens=stage1_budget),
            ),
            stage2,
            None,
        )
    stage1_cap, stage2_cap = stage_token_caps(
        profile,
        stage1_budget=stage1_budget,
    )
    return (
        replace(
            stage1,
            budget=replace(stage1.budget, max_tokens=stage1_cap),
        ),
        replace(
            stage2,
            budget=replace(stage2.budget, max_tokens=stage2_cap),
        ),
        profile.stage2_protocol,
    )


__all__ = [
    "TwoStageProtocol",
    "TwoStageProfile",
    "default_two_stage_protocol",
    "apply_offline_two_stage_budget",
    "derive_profiled_two_stage_configs",
    "load_two_stage_profiles",
    "resolve_two_stage_profile",
    "stage_token_caps",
]
