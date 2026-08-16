"""Optional paper-reference configuration and deterministic cell expansion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from r3bench.common.config import load_config
from r3bench.common.profile_registry import (
    load_evaluator_profiles,
    load_model_profiles,
    load_run_profiles,
    validate_run_profile_applicability,
)
from r3bench.common.scorer_registry import load_scorer_profiles
from r3bench.providers.registry import load_provider_profile
from r3bench.resource_paths import resource_path, resolve_path


DEFAULT_BENCHMARK_PATH = resource_path("configs", "benchmark.yaml")
_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "reference_models",
        "domains",
        "settings",
        "roles",
        "reference_cell_count",
        "model_bindings",
        "scorer_bindings",
        "prompt_bindings",
        "budgets",
    }
)
_DOMAINS = ("coding", "math", "abstract_reasoning")
_SETTINGS = ("tool_free", "agentic")
_ROLES = (
    "unbudgeted_baseline",
    "budgeted_rho_0p2",
    "budgeted_rho_0p8",
    "single_problem_response_curve",
)
_RESPONSE_CURVE_LEVEL_COUNT = 6
_RHO_0P2_GRID_INDEX = 3
_RHO_0P8_GRID_INDEX = 5


class BenchmarkConfigError(ValueError):
    """Raised when the compact public benchmark definition is inconsistent."""


@dataclass(frozen=True, slots=True)
class BenchmarkCell:
    cell_id: str
    model: str
    domain: str
    setting: str
    role: str
    mode: str
    evaluator_profile: str
    provider_profile: str
    run_profile: str
    scorer_profile: str
    budget: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "model": self.model,
            "domain": self.domain,
            "setting": self.setting,
            "role": self.role,
            "mode": self.mode,
            "evaluator_profile": self.evaluator_profile,
            "provider_profile": self.provider_profile,
            "run_profile": self.run_profile,
            "scorer_profile": self.scorer_profile,
            "budget": dict(self.budget),
        }


def _string_list(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise BenchmarkConfigError(f"{field} must be a non-empty list")
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise BenchmarkConfigError(f"{field} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise BenchmarkConfigError(f"{field} must not contain duplicates")
    return result


def load_benchmark(path: str | Path = DEFAULT_BENCHMARK_PATH) -> Mapping[str, Any]:
    document = load_config(path)
    if set(document) != _ROOT_FIELDS:
        raise BenchmarkConfigError("benchmark document fields mismatch")
    if document["schema_version"] != "1.0":
        raise BenchmarkConfigError("unsupported benchmark schema_version")
    if document["status"] != "release_reference":
        raise BenchmarkConfigError("benchmark status must be release_reference")
    reference = _string_list(document["reference_models"], "reference_models")
    if _string_list(document["domains"], "domains") != _DOMAINS:
        raise BenchmarkConfigError("domain axis differs from the release contract")
    if _string_list(document["settings"], "settings") != _SETTINGS:
        raise BenchmarkConfigError("setting axis differs from the release contract")
    if _string_list(document["roles"], "roles") != _ROLES:
        raise BenchmarkConfigError("role axis differs from the release contract")
    expected_reference = (
        len(reference) * len(_DOMAINS) * len(_SETTINGS) * len(_ROLES)
    )
    if document["reference_cell_count"] != expected_reference:
        raise BenchmarkConfigError("reference_cell_count is inconsistent")
    return MappingProxyType(document)


def _validate_budget_protocol(
    *,
    setting: str,
    model: str,
    domain: str,
    role_map: Mapping[str, Any],
    thinking_enabled: bool,
) -> None:
    expected_policy = "output_tokens" if setting == "tool_free" else "compute_tools"
    cap_field = "output_tokens" if setting == "tool_free" else "counted_actions"
    context = f"{setting}/{model}/{domain}"

    for role, budget in role_map.items():
        if not isinstance(budget, Mapping):
            raise BenchmarkConfigError(f"budget entry is not a mapping: {context}/{role}")
        if budget.get("resource_policy") != expected_policy:
            raise BenchmarkConfigError(
                f"resource policy must be {expected_policy}: {context}/{role}"
            )

    curve = role_map["single_problem_response_curve"].get("grid")
    if (
        not isinstance(curve, list)
        or len(curve) != _RESPONSE_CURVE_LEVEL_COUNT
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in curve
        )
        or curve[0] != 0
        or any(left > right for left, right in zip(curve, curve[1:]))
    ):
        raise BenchmarkConfigError(
            f"response-curve grid must contain six nondecreasing levels: {context}"
        )
    if setting == "tool_free" and len(set(curve)) != len(curve):
        raise BenchmarkConfigError(
            f"tool-free response-curve levels must be distinct: {context}"
        )
    if setting == "agentic" and any(level < 1 for level in curve[1:]):
        raise BenchmarkConfigError(
            f"positive agentic pressure levels must be clipped at one: {context}"
        )
    if setting == "tool_free":
        expected_protocol = "two_stage" if thinking_enabled else "one_stage"
        actual_protocol = role_map["single_problem_response_curve"].get("protocol")
        if actual_protocol != expected_protocol:
            raise BenchmarkConfigError(
                f"Tool-Free curve protocol must be {expected_protocol}: {context}"
            )

    rho_0p2 = role_map["budgeted_rho_0p2"].get(cap_field)
    rho_0p8 = role_map["budgeted_rho_0p8"].get(cap_field)
    if curve[_RHO_0P2_GRID_INDEX] != rho_0p2:
        raise BenchmarkConfigError(
            f"rho=0.2 response-curve level differs from contest cap: {context}"
        )
    if curve[_RHO_0P8_GRID_INDEX] != rho_0p8:
        raise BenchmarkConfigError(
            f"rho=0.8 response-curve level differs from contest cap: {context}"
        )


def expand_cells(
    benchmark: Mapping[str, Any] | None = None,
) -> tuple[BenchmarkCell, ...]:
    document = benchmark or load_benchmark()
    models = load_model_profiles()
    evaluators = load_evaluator_profiles()
    runs = load_run_profiles()
    scorers = load_scorer_profiles()
    reference = tuple(document["reference_models"])
    if not set(reference).issubset(models):
        raise BenchmarkConfigError(
            "reference models must be a subset of supported model profiles"
        )
    bindings = document["model_bindings"]
    scorer_bindings = document["scorer_bindings"]
    budgets = document["budgets"]
    if not isinstance(bindings, Mapping) or set(bindings) != set(reference):
        raise BenchmarkConfigError("model_bindings do not cover reference models")
    if not isinstance(scorer_bindings, Mapping) or set(scorer_bindings) != set(_DOMAINS):
        raise BenchmarkConfigError("scorer_bindings do not cover all domains")

    cells: list[BenchmarkCell] = []
    for model in reference:
        binding = bindings[model]
        if not isinstance(binding, Mapping):
            raise BenchmarkConfigError(f"model binding for {model} must be a mapping")
        evaluator_key = str(binding.get("evaluator_profile"))
        provider_key = str(binding.get("provider_profile"))
        run_bindings = binding.get("run_profiles")
        if evaluator_key not in evaluators:
            raise BenchmarkConfigError(f"unknown evaluator profile for {model}")
        if models[model].evaluator_profile != evaluator_key:
            raise BenchmarkConfigError(f"model/evaluator binding mismatch for {model}")
        if models[model].provider_profile != provider_key:
            raise BenchmarkConfigError(f"model/provider binding mismatch for {model}")
        provider_path = resource_path("configs", "providers", f"{provider_key}.yaml")
        load_provider_profile(provider_path)
        if not isinstance(run_bindings, Mapping) or set(run_bindings) != set(_DOMAINS):
            raise BenchmarkConfigError(f"run profiles do not cover every domain for {model}")

        for domain in _DOMAINS:
            scorer_key = str(scorer_bindings[domain])
            if scorer_key not in scorers or scorers[scorer_key].domain != domain:
                raise BenchmarkConfigError(f"invalid scorer binding for {domain}")
            run_key = str(run_bindings[domain])
            if run_key not in runs:
                raise BenchmarkConfigError(f"unknown run profile {run_key}")
            validate_run_profile_applicability(
                runs[run_key],
                model_key=model,
                provider_profile=provider_key,
                domain=domain,
                setting="tool_free",
            )
            for setting in _SETTINGS:
                role_map = budgets.get(setting, {}).get(model, {}).get(domain)
                if not isinstance(role_map, Mapping) or set(role_map) != set(_ROLES):
                    raise BenchmarkConfigError(
                        f"budgets do not cover {setting}/{model}/{domain}"
                    )
                _validate_budget_protocol(
                    setting=setting,
                    model=model,
                    domain=domain,
                    role_map=role_map,
                    thinking_enabled=models[model].thinking_enabled,
                )
                for role in _ROLES:
                    budget = role_map[role]
                    if not isinstance(budget, Mapping):
                        raise BenchmarkConfigError("budget entries must be mappings")
                    rho = budget.get("rho")
                    expected_rho = (
                        0.2 if role == "budgeted_rho_0p2"
                        else 0.8 if role == "budgeted_rho_0p8"
                        else None
                    )
                    if rho != expected_rho:
                        raise BenchmarkConfigError(
                            f"rho mismatch for {setting}/{model}/{domain}/{role}"
                        )
                    if role == "single_problem_response_curve":
                        grid = budget.get("grid")
                        assert isinstance(grid, list)
                    mode = (
                        "single_problem"
                        if role == "single_problem_response_curve"
                        else "contest"
                    )
                    cell_id = "_".join(
                        (
                            setting,
                            domain,
                            model.replace(".", "_").replace("-", "_"),
                            role,
                        )
                    )
                    cells.append(
                        BenchmarkCell(
                            cell_id=cell_id,
                            model=model,
                            domain=domain,
                            setting=setting,
                            role=role,
                            mode=mode,
                            evaluator_profile=evaluator_key,
                            provider_profile=provider_key,
                            run_profile=run_key,
                            scorer_profile=scorer_key,
                            budget=MappingProxyType(dict(budget)),
                        )
                    )

    prompt_bindings = document["prompt_bindings"]
    if not isinstance(prompt_bindings, Mapping):
        raise BenchmarkConfigError("prompt_bindings must be a mapping")
    for value in _walk_prompt_paths(prompt_bindings):
        if not resolve_path(value).is_file():
            raise BenchmarkConfigError(f"prompt resource does not exist: {value}")
    if len(cells) != document["reference_cell_count"]:
        raise BenchmarkConfigError(
            "expanded cell count differs from reference_cell_count"
        )
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise BenchmarkConfigError("expanded cell IDs are not unique")
    return tuple(cells)


def _walk_prompt_paths(value: object) -> tuple[str, ...]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for item in value.values():
            result.extend(_walk_prompt_paths(item))
    elif isinstance(value, str) and value.startswith("prompts/"):
        result.append(value)
    return tuple(result)


__all__ = [
    "BenchmarkCell",
    "BenchmarkConfigError",
    "DEFAULT_BENCHMARK_PATH",
    "expand_cells",
    "load_benchmark",
]
