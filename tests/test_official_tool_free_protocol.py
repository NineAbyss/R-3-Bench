from __future__ import annotations

import copy
import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Mapping

import pytest

from r3bench.benchmark import BenchmarkConfigError, expand_cells, load_benchmark
from r3bench.commands.analysis import (
    _effective_stage1_only,
    _validate_official_provenance,
    _validate_official_scoring_contract,
    _validate_official_tool_free_protocol,
)
from r3bench.commands.run_evaluation import main as run_evaluation_main
from r3bench.common.budget import resolve_official_budget_profile
from r3bench.common.scorer_registry import (
    load_scorer_profiles,
    scorer_profile_contract,
    scorer_profile_contract_sha256,
)
from r3bench.oracle.response_curve_schema import OracleSchemaError


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def _production_scoring_summary(
    domain: str,
    *,
    input_bytes: bytes,
    results_bytes: bytes,
) -> dict[str, Any]:
    scorer = next(
        profile
        for profile in load_scorer_profiles().values()
        if profile.domain == domain
    )
    return {
        "status": "complete",
        "domain": domain,
        "scoring_mode": "production",
        "scorer_profile": scorer.profile_id,
        "scorer_contract": scorer_profile_contract(scorer),
        "scorer_contract_sha256": scorer_profile_contract_sha256(scorer),
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "results_sha256": hashlib.sha256(results_bytes).hexdigest(),
    }


def test_benchmark_curve_protocols_follow_model_thinking_registry() -> None:
    cells = expand_cells()
    curve_protocols = {
        (cell.model, cell.domain): cell.budget["protocol"]
        for cell in cells
        if cell.setting == "tool_free" and cell.role == "single_problem_response_curve"
    }
    assert curve_protocols[("deepseek-reasoner", "coding")] == "two_stage"
    assert curve_protocols[("qwen3.7-max", "abstract_reasoning")] == "two_stage"

    invalid = copy.deepcopy(dict(load_benchmark()))
    invalid["budgets"]["tool_free"]["deepseek-chat"]["math"][
        "single_problem_response_curve"
    ]["protocol"] = "two_stage"
    with pytest.raises(BenchmarkConfigError, match="must be one_stage"):
        expand_cells(invalid)


def test_official_stage1_accounting_is_derived_from_model_protocol() -> None:
    thinking = resolve_official_budget_profile(
        "tool_free_math_qwen3_7_max_single_problem_response_curve",
        setting="tool_free",
        domain="math",
        model_key="qwen3.7-max",
    )
    chat = resolve_official_budget_profile(
        "tool_free_math_deepseek_chat_single_problem_response_curve",
        setting="tool_free",
        domain="math",
        model_key="deepseek-chat",
    )
    assert _effective_stage1_only(False, thinking) is True
    assert _effective_stage1_only(True, thinking) is True
    assert _effective_stage1_only(False, chat) is False
    with pytest.raises(OracleSchemaError, match="one-stage"):
        _effective_stage1_only(True, chat)


@pytest.mark.parametrize(
    ("model", "profile", "protocol", "required"),
    [
        (
            "qwen3.7-max",
            "tool_free_math_qwen3_7_max_single_problem_response_curve",
            "one_stage",
            "two_stage",
        ),
        (
            "deepseek-chat",
            "tool_free_math_deepseek_chat_single_problem_response_curve",
            "two_stage",
            "one_stage",
        ),
    ],
)
def test_official_run_protocol_is_bound_to_model_thinking(
    tmp_path: Path,
    resources: Path,
    capsys: pytest.CaptureFixture[str],
    model: str,
    profile: str,
    protocol: str,
    required: str,
) -> None:
    code = run_evaluation_main(
        [
            "--setting",
            "tool_free",
            "--domain",
            "math",
            "--mode",
            "response_curve",
            "--model",
            model,
            "--data",
            "examples/data/math/problems.jsonl",
            "--output-dir",
            str(tmp_path / model),
            "--budget-profile",
            profile,
            "--provider",
            "mock",
            "--protocol",
            protocol,
            "--model-profiles",
            str(resources / "configs/model_profiles.yaml"),
            "--two-stage-profiles",
            str(resources / "configs/two_stage_profiles.yaml"),
            "--toy",
            "--repeats",
            "1",
        ]
    )
    assert code == 2
    assert f"requires --protocol {required}" in capsys.readouterr().err


def test_official_thinking_run_records_trace_only_stage2_contract(
    tmp_path: Path,
    resources: Path,
) -> None:
    target = tmp_path / "thinking"
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
                "qwen3.7-max",
                "--data",
                "examples/data/math/problems.jsonl",
                "--output-dir",
                str(target),
                "--budget-profile",
                "tool_free_math_qwen3_7_max_budgeted_rho_0p2",
                "--provider",
                "mock",
                "--protocol",
                "two_stage",
                "--model-profiles",
                str(resources / "configs/model_profiles.yaml"),
                "--two-stage-profiles",
                str(resources / "configs/two_stage_profiles.yaml"),
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
    assert summary["stage"] == "stage2"
    assert summary["model_key"] == "qwen3.7-max"
    assert summary["protocol"]["kind"] == "two_stage"
    assert summary["protocol"]["stage1"] == {
        "accounting": "reported_output_tokens",
        "model_key": "qwen3.7-max",
        "thinking_enabled": True,
    }
    assert summary["protocol"]["stage2"]["trace_only"] is True
    assert summary["protocol"]["stage2"]["include_original_problems"] is False
    assert summary["protocol"]["stage2"]["prompt_assembly"] == (
        "reasoning_visible_trace"
    )
    attempts = [
        json.loads(line)
        for line in (target / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["stage"] for row in attempts] == ["stage1", "stage2"]


def test_official_analysis_accepts_bound_one_stage_provenance(
    tmp_path: Path,
) -> None:
    profile = resolve_official_budget_profile(
        "tool_free_math_deepseek_chat_budgeted_rho_0p2",
        setting="tool_free",
        domain="math",
        model_key="deepseek-chat",
    )
    run_dir = tmp_path / "run"
    scoring_dir = tmp_path / "scoring"
    run_dir.mkdir()
    scoring_dir.mkdir()
    summary = {
        "run_id": "run-one",
        "execution_id": "execution-one",
        "repeat_id": 1,
        "domain": "math",
        "stage": "one_stage",
        "model_key": "deepseek-chat",
        "model_name": "deepseek-chat",
        "provider_name": "deepseek_openai_compatible",
        "attempt_count": 1,
        "problem_count": 1,
        "parsed_count": 1,
        "budget": {
            "condition_kind": "official_profile",
            "profile_id": profile.profile_id,
            "unit": "output_tokens",
            "value": profile.budget_value,
        },
        "protocol": {
            "kind": "one_stage",
            "model_key": "deepseek-chat",
            "model_thinking_enabled": False,
            "two_stage_profile": None,
            "official_rho": 0.2,
            "budget_accounting": "single_stage_output_tokens",
            "stage1": None,
            "stage2": None,
        },
    }
    attempt_row = {
        "problem_id": "problem-one",
        "run_id": "run-one",
        "execution_id": "execution-one",
        "repeat_id": 1,
        "source_setting": "tool_free",
        "request_id": "request-one",
        "stage": "one_stage",
        "stage_input_kind": "public_prompt",
        "parent_request_id": None,
        "stage1_request_id": None,
        "stage2_request_id": None,
    }
    _write_jsonl(
        run_dir / "attempts.jsonl",
        [attempt_row],
    )
    parsed_row = {
        "problem_id": "problem-one",
        "run_id": "run-one",
        "execution_id": "execution-one",
        "repeat_id": 1,
        "source_setting": "tool_free",
        "request_id": "request-one",
        "stage": "one_stage",
        "stage1_request_id": None,
        "stage2_request_id": None,
        "parse_status": "parsed",
    }
    _write_jsonl(run_dir / "parsed_answers.jsonl", [parsed_row])
    summary["attempts_sha256"] = hashlib.sha256(
        (run_dir / "attempts.jsonl").read_bytes()
    ).hexdigest()
    summary["parsed_answers_sha256"] = hashlib.sha256(
        (run_dir / "parsed_answers.jsonl").read_bytes()
    ).hexdigest()
    (run_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    scorer = load_scorer_profiles()["math_equivalence_judge"]
    judge_row = {
        **parsed_row,
        "scoring_mode": "production",
        "evaluator": scorer.profile_id,
    }
    _write_jsonl(scoring_dir / "judge_results.jsonl", [judge_row])
    (scoring_dir / "scoring_summary.json").write_text(
        json.dumps(
            _production_scoring_summary(
                "math",
                input_bytes=(run_dir / "parsed_answers.jsonl").read_bytes(),
                results_bytes=(scoring_dir / "judge_results.jsonl").read_bytes(),
            )
        ),
        encoding="utf-8",
    )
    _validate_official_provenance(
        Namespace(
            setting="tool_free",
            domain="math",
            model="deepseek-chat",
            run_dir=str(run_dir),
            scoring_dir=str(scoring_dir),
        ),
        profile,
        expected_budget=profile.budget_value,
    )

    forged_judge = {**judge_row, "request_id": "another-run"}
    _write_jsonl(scoring_dir / "judge_results.jsonl", [forged_judge])
    (scoring_dir / "scoring_summary.json").write_text(
        json.dumps(
            _production_scoring_summary(
                "math",
                input_bytes=(run_dir / "parsed_answers.jsonl").read_bytes(),
                results_bytes=(scoring_dir / "judge_results.jsonl").read_bytes(),
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(OracleSchemaError, match="selected generation run"):
        _validate_official_provenance(
            Namespace(
                setting="tool_free",
                domain="math",
                model="deepseek-chat",
                run_dir=str(run_dir),
                scoring_dir=str(scoring_dir),
            ),
            profile,
            expected_budget=profile.budget_value,
        )


def test_official_analysis_rejects_forged_or_incomplete_two_stage_metadata(
    tmp_path: Path,
) -> None:
    profile = resolve_official_budget_profile(
        "tool_free_math_qwen3_7_max_budgeted_rho_0p2",
        setting="tool_free",
        domain="math",
        model_key="qwen3.7-max",
    )
    summary = {
        "run_id": "two-stage-run",
        "execution_id": "two-stage-execution",
        "repeat_id": 1,
        "domain": "math",
        "stage": "stage2",
        "model_key": "qwen3.7-max",
        "model_name": "qwen3.7-max",
        "provider_name": "qwen_shared_openai_compatible",
        "attempt_count": 2,
        "problem_count": 1,
        "parsed_count": 1,
        "protocol": {
            "kind": "two_stage",
            "model_key": "qwen3.7-max",
            "model_thinking_enabled": True,
            "two_stage_profile": "qwen_math_ar_two_stage",
            "official_rho": 0.2,
            "budget_accounting": "stage1_only_with_practical_stage2_cap",
            "stage1": {
                "model_key": "qwen3.7-max",
                "thinking_enabled": True,
                "accounting": "reported_output_tokens",
            },
            "stage2": {
                "model_key": "qwen3.7-max",
                "thinking_enabled": False,
                "accounting": "not_counted",
                "practical_output_token_cap": None,
                "handoff_channels": ["reasoning_content", "visible_output"],
                "prompt_assembly": "reasoning_visible_trace",
                "include_original_problems": False,
                "trace_only": True,
            },
        },
    }
    identity = {
        "run_id": "two-stage-run",
        "execution_id": "two-stage-execution",
        "repeat_id": 1,
        "source_setting": "tool_free",
        "problem_id": "problem-one",
    }
    attempts = [
        {
            **identity,
            "request_id": "stage-one",
            "stage": "stage1",
            "stage_input_kind": "public_prompt",
            "parent_request_id": None,
            "stage1_request_id": "stage-one",
            "stage2_request_id": None,
        },
        {
            **identity,
            "request_id": "stage-two",
            "stage": "stage2",
            "stage_input_kind": "stage1_output",
            "parent_request_id": "stage-one",
            "stage1_request_id": "stage-one",
            "stage2_request_id": "stage-two",
        },
    ]
    _write_jsonl(tmp_path / "attempts.jsonl", attempts)
    score_rows = [{"stage": "stage2"}]
    parsed_rows = [
        {
            **identity,
            "request_id": "stage-two",
            "stage": "stage2",
            "stage1_request_id": "stage-one",
            "stage2_request_id": "stage-two",
            "parse_status": "parsed",
        }
    ]
    _validate_official_tool_free_protocol(
        summary,
        profile,
        run_dir=tmp_path,
        score_rows=score_rows,
        attempt_rows=attempts,
        generation_rows=parsed_rows,
    )

    unparsed_summary = copy.deepcopy(summary)
    unparsed_summary["parsed_count"] = 0
    unparsed_rows = copy.deepcopy(parsed_rows)
    unparsed_rows[0]["parse_status"] = "parse_error"
    _validate_official_tool_free_protocol(
        unparsed_summary,
        profile,
        run_dir=tmp_path,
        score_rows=score_rows,
        attempt_rows=attempts,
        generation_rows=unparsed_rows,
    )

    non_bijective = copy.deepcopy(attempts)
    non_bijective[1]["problem_id"] = "problem-two"
    with pytest.raises(OracleSchemaError, match="one-to-one item mapping"):
        _validate_official_tool_free_protocol(
            summary,
            profile,
            run_dir=tmp_path,
            score_rows=score_rows,
            attempt_rows=non_bijective,
            generation_rows=parsed_rows,
        )

    wrong_parsed_request = copy.deepcopy(parsed_rows)
    wrong_parsed_request[0]["request_id"] = "unrelated-stage-two"
    with pytest.raises(OracleSchemaError, match="final-stage attempt"):
        _validate_official_tool_free_protocol(
            summary,
            profile,
            run_dir=tmp_path,
            score_rows=score_rows,
            attempt_rows=attempts,
            generation_rows=wrong_parsed_request,
        )

    missing_trace = copy.deepcopy(summary)
    del missing_trace["protocol"]["stage2"]["trace_only"]
    with pytest.raises(OracleSchemaError, match="trace-only Stage 2 metadata"):
        _validate_official_tool_free_protocol(
            missing_trace,
            profile,
            run_dir=tmp_path,
            score_rows=score_rows,
        )

    forged_stage = copy.deepcopy(summary)
    forged_stage["stage"] = "one_stage"
    with pytest.raises(OracleSchemaError, match="stage or model protocol"):
        _validate_official_tool_free_protocol(
            forged_stage,
            profile,
            run_dir=tmp_path,
            score_rows=score_rows,
        )


@pytest.mark.parametrize(
    ("domain", "field", "wrong_value"),
    [
        ("coding", "verifier", "DifferentVerifier"),
        ("math", "judge_model", "different-math-judge"),
        (
            "abstract_reasoning",
            "reasoning_gym_revision",
            "0" * 40,
        ),
    ],
)
def test_official_analysis_rejects_self_consistent_wrong_scorer_contract(
    domain: str,
    field: str,
    wrong_value: str,
) -> None:
    results_bytes = b'{"scoring_mode":"production"}\n'
    summary = _production_scoring_summary(
        domain,
        input_bytes=b'{"problem_id":"one"}\n',
        results_bytes=results_bytes,
    )
    _validate_official_scoring_contract(
        summary,
        domain=domain,
        results_bytes=results_bytes,
    )

    forged = copy.deepcopy(summary)
    contract = forged["scorer_contract"]
    assert isinstance(contract, dict)
    contract["config"][field] = wrong_value
    forged["scorer_contract_sha256"] = hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(OracleSchemaError, match="released scorer contract"):
        _validate_official_scoring_contract(
            forged,
            domain=domain,
            results_bytes=results_bytes,
        )
