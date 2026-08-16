from __future__ import annotations

from pathlib import Path

import pytest

from r3bench.benchmark import expand_cells, load_benchmark
from r3bench.common.profile_registry import (
    load_evaluator_profiles,
    load_model_profiles,
    load_run_profiles,
    resolve_transport_parameters,
)
from r3bench.common.provider import ModelRequest
from r3bench.common.scorer_registry import load_scorer_profiles
from r3bench.common.two_stage_profile import load_two_stage_profiles
from r3bench.providers.registry import load_provider_profile
from r3bench.providers.openai_compatible import OpenAICompatibleAdapter


def test_benchmark_expands_reference_matrix() -> None:
    benchmark = load_benchmark()
    cells = expand_cells(benchmark)
    assert len(benchmark["reference_models"]) == 4
    assert benchmark["reference_cell_count"] == 96
    assert len(cells) == 96
    assert len({cell.cell_id for cell in cells}) == 96
    assert {cell.domain for cell in cells} == {
        "coding",
        "math",
        "abstract_reasoning",
    }
    assert {cell.setting for cell in cells} == {"tool_free", "agentic"}


def test_reference_response_curves_follow_pressure_grid() -> None:
    benchmark = load_benchmark()
    cells = expand_cells(benchmark)
    indexed = {
        (cell.setting, cell.model, cell.domain, cell.role): cell for cell in cells
    }
    for setting in ("tool_free", "agentic"):
        cap_field = "output_tokens" if setting == "tool_free" else "counted_actions"
        expected_policy = "output_tokens" if setting == "tool_free" else "compute_tools"
        for model in benchmark["reference_models"]:
            for domain in ("coding", "math", "abstract_reasoning"):
                prefix = (setting, model, domain)
                curve = indexed[prefix + ("single_problem_response_curve",)]
                rho_0p2 = indexed[prefix + ("budgeted_rho_0p2",)]
                rho_0p8 = indexed[prefix + ("budgeted_rho_0p8",)]
                grid = curve.budget["grid"]
                assert len(grid) == 6
                assert grid[0] == 0
                assert grid[3] == rho_0p2.budget[cap_field]
                assert grid[5] == rho_0p8.budget[cap_field]
                assert all(
                    indexed[prefix + (role,)].budget["resource_policy"]
                    == expected_policy
                    for role in (
                        "unbudgeted_baseline",
                        "budgeted_rho_0p2",
                        "budgeted_rho_0p8",
                        "single_problem_response_curve",
                    )
                )

    assert indexed[
        ("agentic", "qwen3.7-max", "coding", "budgeted_rho_0p8")
    ].budget["counted_actions"] == 12


def test_release_profiles_are_resolved_and_approved() -> None:
    models = load_model_profiles()
    runs = load_run_profiles()
    scorers = load_scorer_profiles()
    evaluators = load_evaluator_profiles()
    bundled_reference_models = {
        "qwen3.7-max",
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek-v4-pro",
    }
    assert bundled_reference_models.issubset(models)
    assert len(models) in {4, 8}
    assert len(runs) == 2 * len(models)
    assert set(scorers) == {
        "coding_lightcpverifier_external",
        "math_equivalence_judge",
        "abstract_reasoning_reasoning_gym",
    }
    assert set(evaluators) == {"qwen_shared", "deepseek_shared"}
    assert all(not profile.requires_owner_approval for profile in models.values())
    assert all(not profile.requires_owner_approval for profile in runs.values())
    assert models["deepseek-chat"].reasoning_effort is None
    assert models["deepseek-reasoner"].reasoning_effort == "high"


@pytest.mark.parametrize(
    "name",
    [
        "qwen_shared_openai_compatible",
        "deepseek_openai_compatible",
        "mock",
        "replay",
    ],
)
def test_provider_profile_is_secret_free_and_loadable(
    resources: Path, name: str
) -> None:
    profile = load_provider_profile(
        resources / f"configs/providers/{name}.yaml"
    )
    serialized = repr(profile).lower()
    assert "authorization" not in serialized
    assert "bearer " not in serialized


def test_transport_resolution_has_no_unresolved_fields(resources: Path) -> None:
    provider = load_provider_profile(
        resources / "configs/providers/qwen_shared_openai_compatible.yaml"
    )
    run = load_run_profiles()["qwen_coding"]
    transport = resolve_transport_parameters(provider, run)
    assert transport.unresolved_fields == ()
    assert transport.values["timeout_seconds"] > 0


def test_two_stage_profiles_cover_active_reasoning_models(resources: Path) -> None:
    profiles = load_two_stage_profiles(
        resources / "configs/two_stage_profiles.yaml"
    )
    assert set(profiles) == {
        "deepseek_chat_coding_two_stage",
        "qwen_coding_two_stage",
        "deepseek_reasoner_coding_two_stage",
        "deepseek_v4_coding_two_stage",
        "qwen_math_ar_two_stage",
        "deepseek_reasoner_math_ar_two_stage",
        "deepseek_v4_math_ar_two_stage",
    }
    assert all(profile.status == "release" for profile in profiles.values())
    assert {
        profile.budget_accounting for profile in profiles.values()
    } == {"stage1_only_with_practical_stage2_cap"}
    for profile_id in (
        "qwen_math_ar_two_stage",
        "deepseek_reasoner_math_ar_two_stage",
        "deepseek_v4_math_ar_two_stage",
    ):
        protocol = profiles[profile_id].stage2_protocol
        assert protocol.prompt_assembly == "reasoning_visible_trace"
        assert protocol.include_original_problems is False


def test_math_scorer_uses_paper_flash_judge() -> None:
    scorer = load_scorer_profiles()["math_equivalence_judge"]
    assert scorer.config["judge_model"] == "deepseek-v4-flash"


@pytest.mark.parametrize("model_key", ["deepseek-reasoner", "deepseek-v4-pro"])
def test_deepseek_sampling_fields_are_omitted_and_zero_budget_is_preserved(
    resources: Path,
    model_key: str,
) -> None:
    provider = load_provider_profile(
        resources / "configs/providers/deepseek_openai_compatible.yaml"
    )
    model = load_model_profiles()[model_key]
    adapter = OpenAICompatibleAdapter(provider, model)
    payload = adapter.build_payload(
        ModelRequest(
            request_id="deepseek-zero",
            model=model.model_key,
            prompt_text="test",
            max_tokens=0,
            temperature=None,
            top_p=None,
        )
    )
    assert payload["max_tokens"] == 0
    assert "temperature" not in payload
    assert "top_p" not in payload


def test_explicit_sampling_fields_are_still_sent(resources: Path) -> None:
    provider = load_provider_profile(
        resources / "configs/providers/qwen_shared_openai_compatible.yaml"
    )
    model = load_model_profiles()["qwen3.7-max"]
    adapter = OpenAICompatibleAdapter(provider, model)
    payload = adapter.build_payload(
        ModelRequest(
            request_id="qwen-sampling",
            model=model.model_key,
            prompt_text="test",
            max_tokens=10,
            temperature=0.2,
            top_p=0.8,
        )
    )
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.8
    assert payload["max_completion_tokens"] == 10


def test_model_request_rejects_negative_budget() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ModelRequest(
            request_id="negative",
            model="model",
            prompt_text="test",
            max_tokens=-1,
        )
