from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path

import pytest

from r3bench.agentic.external_backend import (
    check_external_backend_readiness,
    load_external_backend_config,
)
from r3bench.commands.run_evaluation import main as run_evaluation_main
from r3bench.common.budget import resolve_budget
from r3bench.common.experiment import (
    BudgetConfig,
    ExperimentConfig,
    JudgeConfig,
    PromptConfig,
    ProviderConfig,
)
from r3bench.common.profile_registry import load_model_profiles
from r3bench.common.nl_runner import (
    _contest_sections,
    prepare_two_stage_requests,
    run_two_stage_nl,
)
from r3bench.common.provider import MockProvider
from r3bench.common.two_stage_profile import (
    apply_offline_two_stage_budget,
    derive_profiled_two_stage_configs,
    load_two_stage_profiles,
    stage_token_caps,
)
from r3bench.coding.scoring import score_coding_saved_outputs
from r3bench.oracle.protocol_v3 import run_condition_analysis


def test_budget_priority_and_custom_condition() -> None:
    resolved = resolve_budget(
        setting="tool_free",
        explicit_value=17,
        config_value=11,
        profile_id=None,
        condition_id="my_condition",
        domain="math",
        model_key="any-compatible-model",
    )
    assert resolved.value == 17
    assert resolved.source == "cli_explicit"
    assert resolved.condition_kind == "custom"
    assert resolved.rho is None


def test_two_stage_budget_applies_only_to_stage1(resources: Path) -> None:
    profile = load_two_stage_profiles(resources / "configs/two_stage_profiles.yaml")[
        "qwen_coding_two_stage"
    ]
    assert stage_token_caps(profile, stage1_budget=0) == (0, 65536)
    assert stage_token_caps(profile, stage1_budget=100) == (100, 65536)


def test_two_stage_registry_has_only_stage1_accounting(resources: Path) -> None:
    profiles = load_two_stage_profiles(resources / "configs/two_stage_profiles.yaml")
    allowed_fields = {
        "profile_id",
        "applicable_models",
        "applicable_domains",
        "stage1_model_key",
        "stage2_model_key",
        "stage1_thinking_enabled",
        "stage2_thinking_enabled",
        "stage2_protocol",
        "budget_accounting",
        "stage2_practical_cap",
        "provenance_source",
        "status",
        "notes",
    }
    assert {profile.budget_accounting for profile in profiles.values()} == {
        "stage1_only_with_practical_stage2_cap"
    }
    assert all(
        {field.name for field in fields(profile)} == allowed_fields
        for profile in profiles.values()
    )


@pytest.mark.parametrize(
    ("profile_id", "stage1_model", "stage2_model"),
    [
        ("qwen_coding_two_stage", "qwen3.7-max", "qwen3.7-max"),
        (
            "deepseek_reasoner_coding_two_stage",
            "deepseek-reasoner",
            "deepseek-chat",
        ),
        (
            "deepseek_v4_coding_two_stage",
            "deepseek-v4-pro",
            "deepseek-chat",
        ),
    ],
)
def test_reasoning_coding_profiles_define_stage_contract(
    resources: Path,
    profile_id: str,
    stage1_model: str,
    stage2_model: str,
) -> None:
    profile = load_two_stage_profiles(resources / "configs/two_stage_profiles.yaml")[
        profile_id
    ]
    assert profile.stage1_model_key == stage1_model
    assert profile.stage2_model_key == stage2_model
    assert profile.stage2_protocol.handoff_channels == (
        "reasoning_content",
        "visible_output",
    )
    assert profile.stage1_thinking_enabled is True
    assert profile.stage2_thinking_enabled is False
    assert profile.stage2_practical_cap == 65536


def _coding_base_config(*, max_tokens: int) -> ExperimentConfig:
    return ExperimentConfig(
        name="coding_two_stage",
        domain="coding",
        mode="contest",
        visibility="hidden",
        setting="tool_free",
        stage="one_stage",
        data_source="examples/data/coding.jsonl",
        provider=ProviderConfig(name="mock", model="local-mock"),
        budget=BudgetConfig(max_tokens=max_tokens, temperature=0.2),
        prompt=PromptConfig("prompts/coding/contest.txt"),
        judge=JudgeConfig("coding_lightcpverifier_external"),
    )


def test_derive_profiled_configs_uses_stage1_budget_and_stage2_cap(
    resources: Path,
) -> None:
    profile = load_two_stage_profiles(resources / "configs/two_stage_profiles.yaml")[
        "deepseek_reasoner_coding_two_stage"
    ]
    stage1, stage2 = derive_profiled_two_stage_configs(
        _coding_base_config(max_tokens=100),
        profile,
        load_model_profiles(),
    )
    assert (stage1.stage, stage1.model_name, stage1.max_tokens) == (
        "stage1",
        "deepseek-reasoner",
        100,
    )
    assert (stage2.stage, stage2.model_name, stage2.max_tokens) == (
        "stage2",
        "deepseek-chat",
        65536,
    )
    assert stage1.budget.temperature is None
    assert stage1.budget.top_p is None
    assert stage2.budget.temperature == 0.0
    assert stage2.budget.top_p == 1.0


def test_apply_offline_budget_always_applies_stage1_budget() -> None:
    stage1 = _coding_base_config(max_tokens=200)
    stage2 = _coding_base_config(max_tokens=300)
    budgeted_stage1, unchanged_stage2, protocol = apply_offline_two_stage_budget(
        stage1,
        stage2,
        stage1_budget=100,
        profile=None,
    )
    assert budgeted_stage1.max_tokens == 100
    assert unchanged_stage2 == stage2
    assert protocol is None


def test_apply_offline_budget_uses_profile_cap_and_returns_protocol(
    resources: Path,
) -> None:
    profile = load_two_stage_profiles(resources / "configs/two_stage_profiles.yaml")[
        "qwen_coding_two_stage"
    ]
    budgeted_stage1, budgeted_stage2, protocol = apply_offline_two_stage_budget(
        _coding_base_config(max_tokens=200),
        _coding_base_config(max_tokens=300),
        stage1_budget=100,
        profile=profile,
    )
    assert budgeted_stage1.max_tokens == 100
    assert budgeted_stage2.max_tokens == 65536
    assert protocol == profile.stage2_protocol


def test_math_stage2_request_contains_trace_but_not_original_problem(
    resources: Path,
) -> None:
    base = ExperimentConfig(
        name="math_two_stage",
        domain="math",
        mode="contest",
        visibility="hidden",
        setting="tool_free",
        stage="one_stage",
        data_source="examples/data/math",
        provider=ProviderConfig(name="mock", model="local-mock"),
        budget=BudgetConfig(max_tokens=100, temperature=0.2),
        prompt=PromptConfig("prompts/math/contest_nl.txt"),
        judge=JudgeConfig("mock_math"),
        strict_data=False,
    )
    profile = load_two_stage_profiles(resources / "configs/two_stage_profiles.yaml")[
        "qwen_math_ar_two_stage"
    ]
    stage1, stage2 = derive_profiled_two_stage_configs(
        base,
        profile,
        load_model_profiles(),
    )
    prepared = prepare_two_stage_requests(
        stage1,
        stage2,
        limit=1,
        stage1_output="TRACE_ONLY_SENTINEL",
        protocol=profile.stage2_protocol,
    )[0]
    stage2_text = "\n".join(
        message.content for message in prepared.stage2_request.messages
    )
    assert "TRACE_ONLY_SENTINEL" in stage2_text
    assert "Toy math problem" not in stage2_text
    assert "{numbered_content}" not in stage2_text
    assert "boxed{No answer}" in stage2_text
    assert prepared.stage2_request.top_p == 0.8


def test_two_stage_no_answer_sentinel_is_missing_and_not_judged(
    resources: Path,
) -> None:
    base = ExperimentConfig(
        name="math_two_stage_missing",
        domain="math",
        mode="contest",
        visibility="hidden",
        setting="tool_free",
        stage="one_stage",
        data_source="examples/data/math",
        provider=ProviderConfig(name="mock", model="local-mock"),
        budget=BudgetConfig(max_tokens=100, temperature=0.2),
        prompt=PromptConfig("prompts/math/contest_nl.txt"),
        judge=JudgeConfig("mock_math"),
        strict_data=False,
    )
    profile = load_two_stage_profiles(resources / "configs/two_stage_profiles.yaml")[
        "qwen_math_ar_two_stage"
    ]
    stage1, stage2 = derive_profiled_two_stage_configs(
        base,
        profile,
        load_model_profiles(),
    )
    no_answers = "\n\n".join(
        f"## Problem {index}\nFinal Answer: \\boxed{{No answer}}"
        for index in range(1, 7)
    )
    artifacts = run_two_stage_nl(
        stage1,
        stage2,
        MockProvider("trace without candidate answers"),
        MockProvider(no_answers),
        limit=1,
        protocol=profile.stage2_protocol,
    )
    assert len(artifacts.parsed_answers) == 6
    assert {row.parse_status for row in artifacts.parsed_answers} == {"missing"}
    assert {row.parsed_answer for row in artifacts.parsed_answers} == {None}
    assert {row.judge_status for row in artifacts.judge_results} == {"not_judged"}
    assert {row.verdict for row in artifacts.judge_results} == {"missing"}


def test_coding_contest_prompt_displays_budget_and_allows_partial_submission(
    resources: Path,
) -> None:
    prompt = (resources / "prompts/coding/contest_nl.txt").read_text(encoding="utf-8")
    assert "{budget_tokens}" in prompt
    assert "Solve all six" not in prompt
    assert "omit problems" in prompt
    assert "First scan" in prompt
    assert "Do not fabricate" in prompt
    assert "code execution" in prompt
    assert "hidden data" in prompt


def test_ar_contest_prompt_allows_subset_without_fabrication(resources: Path) -> None:
    prompt = (resources / "prompts/abstract_reasoning/contest_nl.txt").read_text(
        encoding="utf-8"
    )
    assert "{budget_tokens}" in prompt
    assert "First scan" in prompt
    assert "any subset" in prompt
    assert "Do not fabricate" in prompt
    assert "only for that problem" in prompt
    assert "<answer>...</answer>" in prompt
    assert "external tools" in prompt


def test_math_contest_prompt_allows_subset_without_fabrication(
    resources: Path,
) -> None:
    prompt = (resources / "prompts/math/contest_nl.txt").read_text(encoding="utf-8")
    assert "{budget_tokens}" in prompt
    assert "any subset" in prompt
    assert "do not fabricate" in prompt
    assert "only for that problem" in prompt
    assert "\\boxed{{...}}" in prompt
    assert "external tools" in prompt


@pytest.mark.parametrize("domain", ["math", "abstract_reasoning", "coding"])
def test_tool_free_contest_stage1_has_full_shared_budget_contract(
    resources: Path,
    domain: str,
) -> None:
    prompt = (resources / f"prompts/{domain}/contest_stage1.txt").read_text(
        encoding="utf-8"
    )
    for required in (
        "{budget_tokens}",
        "shared",
        "Stage 2",
        "original problem statements",
        "without solving from scratch",
        "any subset",
        "Do not fabricate",
        "external tools",
        "code execution",
        "local tests",
        "live judge",
        "hidden data",
        "reference solutions",
    ):
        assert required in prompt


def test_tool_free_contest_rejects_old_labeled_answer_wrappers() -> None:
    assert (
        _contest_sections(
            "abstract_reasoning",
            "<answer A>old labeled format</answer A>",
        )
        == {}
    )
    assert _contest_sections(
        "math",
        "## Problem 1\nFinal Answer: 42\n## Problem B\nFinal Answer: 7",
    ) == {
        "A": "Final Answer: 42",
        "B": "Final Answer: 7",
    }


def test_unified_tool_free_custom_budget_is_offline(
    tmp_path: Path,
) -> None:
    target = tmp_path / "pure"
    assert (
        run_evaluation_main(
            [
                "--setting",
                "tool_free",
                "--domain",
                "math",
                "--mode",
                "single_problem",
                "--model",
                "local-mock",
                "--data",
                "examples/data/math/problems.jsonl",
                "--output-dir",
                str(target),
                "--output-token-budget",
                "31",
                "--provider",
                "mock",
                "--toy",
                "--limit-problems",
                "1",
                "--repeats",
                "1",
            ]
        )
        == 0
    )
    summary = json.loads((target / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["budget"]["value"] == 31
    assert summary["budget"]["condition_kind"] == "custom"
    assert summary["provider_name"] == "mock"


def test_zero_budget_curve_calls_model_in_five_independent_repeats(
    tmp_path: Path,
) -> None:
    target = tmp_path / "zero-curve"
    assert (
        run_evaluation_main(
            [
                "--setting",
                "tool_free",
                "--domain",
                "math",
                "--mode",
                "response_curve",
                "--model",
                "local-mock",
                "--data",
                "examples/data/math/problems.jsonl",
                "--output-dir",
                str(target),
                "--budget-grid",
                "0",
                "--provider",
                "mock",
                "--toy",
                "--limit-problems",
                "1",
            ]
        )
        == 0
    )
    evaluation = json.loads(
        (target / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    assert evaluation["repeat_count"] == 5
    assert len(evaluation["episodes"]) == 5
    assert all(row["model_api_called"] for row in evaluation["episodes"])
    execution_ids: set[str] = set()
    for repeat_id in range(1, 6):
        run_dir = target / "level_1_budget_0" / f"repeat_{repeat_id}"
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        assert summary["attempt_count"] == 1
        assert summary["repeat_id"] == repeat_id
        execution_ids.add(summary["execution_id"])
        attempt = json.loads((run_dir / "attempts.jsonl").read_text(encoding="utf-8"))
        parsed = json.loads(
            (run_dir / "parsed_answers.jsonl").read_text(encoding="utf-8")
        )
        for row in (attempt, parsed):
            assert row["execution_id"] == summary["execution_id"]
            assert row["repeat_id"] == repeat_id
            assert row["source_setting"] == "tool_free"
        assert (
            summary["attempts_sha256"]
            == hashlib.sha256((run_dir / "attempts.jsonl").read_bytes()).hexdigest()
        )
        assert (
            summary["parsed_answers_sha256"]
            == hashlib.sha256(
                (run_dir / "parsed_answers.jsonl").read_bytes()
            ).hexdigest()
        )
        assert attempt["usage"]["output_tokens"] == 0
        assert attempt["usage"]["reasoning_tokens"] == 0
    assert len(execution_ids) == 5


def test_unified_agentic_executes_bounded_mock_episode(tmp_path: Path) -> None:
    target = tmp_path / "agentic"
    assert (
        run_evaluation_main(
            [
                "--setting",
                "agentic",
                "--domain",
                "coding",
                "--mode",
                "contest",
                "--model",
                "local-mock",
                "--data",
                "examples/data/coding.jsonl",
                "--output-dir",
                str(target),
                "--counted-action-budget",
                "3",
                "--provider",
                "mock",
                "--toy",
                "--limit-suites",
                "1",
                "--repeats",
                "1",
            ]
        )
        == 0
    )
    episodes = tuple((target / "episodes").iterdir())
    assert len(episodes) == 1
    summary = json.loads(
        (episodes[0] / "backend_summary.json").read_text(encoding="utf-8")
    )
    assert summary["budget_used"] <= 3
    assert summary["correctness_feedback_exposed"] is False
    assert summary["raw_trajectory_saved"] is False
    assert summary["network_called"] is False
    assert not (episodes[0] / "raw_trajectory.jsonl").exists()


def test_external_agentic_example_fails_closed_without_process(resources: Path) -> None:
    config = load_external_backend_config(
        resources / "configs/agentic/external_backend.example.yaml"
    )
    result = check_external_backend_readiness(config)
    assert result["status"] == "not_configured"
    assert result["external_process_started"] is False
    assert result["credential_value_serialized"] is False


def test_coding_scoring_has_common_judge_fields(
    tmp_path: Path, resources: Path
) -> None:
    rows, _ = score_coding_saved_outputs(
        predictions_path=resources
        / "examples/inputs/scoring/coding_saved_outputs.jsonl",
        data_source=resources / "examples/data/coding.jsonl",
        output_dir=tmp_path / "coding_score",
        mode="mock",
        strict=False,
    )
    row = rows[0]
    assert row["domain"] == "coding"
    assert row["judge_status"] == "judged"
    assert isinstance(row["correct"], bool)
    assert row["score"] in {0.0, 1.0}
    assert row["problem_label"] == "A"


def test_coding_scoring_preserves_generation_stage_binding(
    tmp_path: Path, resources: Path
) -> None:
    source = resources / "examples/inputs/scoring/coding_saved_outputs.jsonl"
    prediction = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    prediction.update(
        {
            "run_id": "coding-run",
            "request_id": "coding-stage2",
            "stage": "stage2",
            "stage1_request_id": "coding-stage1",
            "stage2_request_id": "coding-stage2",
        }
    )
    predictions = tmp_path / "coding-bound.jsonl"
    predictions.write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    rows, summary = score_coding_saved_outputs(
        predictions_path=predictions,
        data_source=resources / "examples/data/coding.jsonl",
        output_dir=tmp_path / "coding-bound-score",
        mode="mock",
        strict=False,
    )
    assert rows[0]["stage"] == "stage2"
    assert rows[0]["request_id"] == "coding-stage2"
    assert summary["scoring_mode"] == "mock"


def test_analysis_v3_custom_condition(tmp_path: Path, resources: Path) -> None:
    custom = run_condition_analysis(
        response_curve_path=resources
        / "examples/inputs/analysis/response_curve_points.jsonl",
        contest_results_path=resources
        / "examples/inputs/analysis/contest_results.jsonl",
        budgets_path=resources / "examples/inputs/analysis/budgets.json",
        output_dir=tmp_path / "custom",
    )
    assert custom["condition_count"] == 1
    gap = json.loads(
        (tmp_path / "custom/gap_summary.json").read_text(encoding="utf-8")
    )["summaries"][0]
    assert gap["condition_id"] == "custom_tokens_8"
    assert gap["rho"] is None
    assert gap["contest_oracle_gap"] == 1.0
