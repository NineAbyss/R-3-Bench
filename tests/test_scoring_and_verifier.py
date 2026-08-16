from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from r3bench.coding.scoring import score_coding_saved_outputs
from r3bench.coding.verifier import CodingVerifierResult
from r3bench.commands.verifier_check import check_readiness
from r3bench.common.profile_registry import load_model_profiles
from r3bench.common.provider import MockProvider
from r3bench.common.scorer_registry import (
    load_scorer_profiles,
    scorer_profile_contract,
    scorer_profile_contract_sha256,
)
from r3bench.common.scoring_dispatch import (
    ProductionScoringRuntime,
    _production_backends,
    score_saved_outputs_cli,
)
from r3bench.common.scoring_dispatch import main as score_main


@pytest.mark.parametrize(
    ("domain", "data", "predictions"),
    [
        (
            "coding",
            "examples/data/coding.jsonl",
            "examples/inputs/scoring/coding_saved_outputs.jsonl",
        ),
        (
            "math",
            "examples/data/math",
            "examples/inputs/scoring/math_saved_outputs.jsonl",
        ),
        (
            "abstract_reasoning",
            "examples/data/abstract_reasoning.jsonl",
            "examples/inputs/scoring/abstract_reasoning_saved_outputs.jsonl",
        ),
    ],
)
def test_three_domain_mock_scoring(
    tmp_path: Path,
    resources: Path,
    domain: str,
    data: str,
    predictions: str,
) -> None:
    output = tmp_path / domain
    assert (
        score_main(
            [
                "--domain",
                domain,
                "--data",
                str(resources / data),
                "--predictions",
                str(resources / predictions),
                "--output-dir",
                str(output),
                "--scoring-mode",
                "mock",
                "--relaxed",
            ]
        )
        == 0
    )
    summary = json.loads((output / "scoring_summary.json").read_text(encoding="utf-8"))
    assert summary["input_count"] == 1
    assert summary["generation_provider_called"] is False


def test_toy_verifier_readiness_uses_no_300_id_manifest(resources: Path) -> None:
    result = check_readiness(
        data_source=resources / "examples/data/coding.jsonl",
        config_path=resources / "configs/verifiers/lightcpverifier.toy.yaml",
    )
    assert result["status"] == "not_configured"
    assert result["public_problem_count"] == 6
    assert result["upstream_id_lookup_contract"] is True
    assert result["external_verifier_started"] is False
    assert result["service_reachable"] is False


def test_coding_dry_run_never_calls_verifier(tmp_path: Path, resources: Path) -> None:
    output = tmp_path / "coding_dry"
    assert (
        score_main(
            [
                "--domain",
                "coding",
                "--data",
                str(resources / "examples/data/coding.jsonl"),
                "--predictions",
                str(resources / "examples/inputs/scoring/coding_saved_outputs.jsonl"),
                "--output-dir",
                str(output),
                "--scoring-mode",
                "dry-run",
                "--scorer-profile",
                str(resources / "configs/scorer_profiles.yaml"),
                "--scorer-key",
                "coding_lightcpverifier_external",
                "--relaxed",
            ]
        )
        == 0
    )
    summary = json.loads((output / "scoring_summary.json").read_text(encoding="utf-8"))
    assert summary["external_service_call_count"] == 0
    assert summary["external_service_called"] is False


def test_coding_production_records_resolved_scorer_contract(
    tmp_path: Path,
    resources: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "lightcpverifier.local.yaml"
    config.write_text(
        "\n".join(
            (
                "verifier_type: lightcpverifier",
                "mode: service",
                "service_url: http://127.0.0.1:8000",
                "asset_root_env: LIGHTCPVERIFIER_ASSET_ROOT",
                "timeout_seconds: 120",
                "max_retries: 1",
                "status: configured",
                "requires_owner_approval: false",
                "",
            )
        ),
        encoding="utf-8",
    )
    assets = tmp_path / "verifier-assets"
    assets.mkdir()
    monkeypatch.setenv("LIGHTCPVERIFIER_ASSET_ROOT", str(assets))
    scorer = load_scorer_profiles()["coding_lightcpverifier_external"]

    def executor(
        upstream_id: str,
        candidate: str,
        runtime_config: object,
    ) -> CodingVerifierResult:
        assert upstream_id == "toy-code-1"
        assert "main" in candidate
        return CodingVerifierResult(
            upstream_id=upstream_id,
            accepted=True,
            verdict="accepted",
        )

    output = tmp_path / "coding-production"
    rows, summary = score_coding_saved_outputs(
        predictions_path=resources
        / "examples/inputs/scoring/coding_saved_outputs.jsonl",
        data_source=resources / "examples/data/coding.jsonl",
        output_dir=output,
        mode="production",
        verifier_config_path=config,
        strict=False,
        production_executor=executor,
        scorer_profile=scorer,
    )
    assert rows[0]["evaluator"] == scorer.profile_id
    assert summary["scorer_profile"] == scorer.profile_id
    assert summary["scorer_contract"] == scorer_profile_contract(scorer)
    assert summary["scorer_contract_sha256"] == scorer_profile_contract_sha256(scorer)
    assert (
        summary["results_sha256"]
        == hashlib.sha256((output / "judge_results.jsonl").read_bytes()).hexdigest()
    )


def test_math_flash_judge_is_independent_of_benchmark_model_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_create_provider(
        provider_profile: object,
        model_profile: object,
        **kwargs: object,
    ) -> MockProvider:
        captured["model"] = model_profile
        captured["transport"] = kwargs["transport_config"]
        return MockProvider("unused")

    monkeypatch.setattr(
        "r3bench.common.scoring_dispatch.create_provider_adapter",
        fake_create_provider,
    )
    scorer = load_scorer_profiles()["math_equivalence_judge"]
    _production_backends(
        "math",
        scorer,
        ProductionScoringRuntime(
            math_model_profiles=str(tmp_path / "must-not-be-read.yaml")
        ),
    )
    model = captured["model"]
    assert getattr(model, "model_key") == "deepseek-v4-flash"
    assert getattr(model, "public_model_id") == "deepseek-v4-flash"
    assert "deepseek-v4-flash" not in load_model_profiles()


@pytest.mark.parametrize(
    ("domain", "data", "problem_id", "sentinel"),
    [
        ("math", "examples/data/math", "toy-math-1", "No answer"),
        (
            "abstract_reasoning",
            "examples/data/abstract_reasoning.jsonl",
            "toy-ar-1",
            "MISSING",
        ),
    ],
)
def test_already_parsed_answer_sentinel_is_not_sent_to_judge(
    tmp_path: Path,
    resources: Path,
    domain: str,
    data: str,
    problem_id: str,
    sentinel: str,
) -> None:
    predictions = tmp_path / f"{domain}.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "domain": domain,
                "problem_id": problem_id,
                "parsed_answer": sentinel,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / f"{domain}-scores"
    assert (
        score_main(
            [
                "--domain",
                domain,
                "--data",
                str(resources / data),
                "--predictions",
                str(predictions),
                "--output-dir",
                str(output),
                "--scoring-mode",
                "mock",
                "--relaxed",
            ]
        )
        == 0
    )
    result = json.loads((output / "judge_results.jsonl").read_text(encoding="utf-8"))
    summary = json.loads((output / "scoring_summary.json").read_text(encoding="utf-8"))
    assert result["judge_status"] == "not_judged"
    assert result["parsed_output"] is None
    assert summary["judged_count"] == 0


def test_scoring_preserves_generation_stage_and_request_binding(
    tmp_path: Path, resources: Path
) -> None:
    predictions = tmp_path / "bound-math.jsonl"
    provenance = {
        "run_id": "run-one",
        "request_id": "stage-two",
        "stage": "stage2",
        "stage1_request_id": "stage-one",
        "stage2_request_id": "stage-two",
        "source_setting": "tool_free",
    }
    predictions.write_text(
        json.dumps(
            {
                "domain": "math",
                "problem_id": "toy-math-1",
                "parsed_answer": "1",
                **provenance,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "bound-math-scores"
    assert (
        score_main(
            [
                "--domain",
                "math",
                "--data",
                str(resources / "examples/data/math"),
                "--predictions",
                str(predictions),
                "--output-dir",
                str(output),
                "--scoring-mode",
                "mock",
                "--relaxed",
            ]
        )
        == 0
    )
    result = json.loads((output / "judge_results.jsonl").read_text(encoding="utf-8"))
    assert {field: result[field] for field in provenance} == provenance


def test_common_scoring_parses_and_hashes_one_input_snapshot(
    tmp_path: Path,
    resources: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = tmp_path / "math-snapshot.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "domain": "math",
                "problem_id": "toy-math-1",
                "parsed_answer": "1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    original_bytes = predictions.read_bytes()

    class MutatingMathJudge:
        def judge(self, problem: object, candidate: str):
            predictions.write_text(
                json.dumps(
                    {
                        "domain": "math",
                        "problem_id": "toy-math-1",
                        "parsed_answer": "forged-after-read",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            from r3bench.math.judge import MathJudgeResult

            return MathJudgeResult(
                problem_id=getattr(problem, "problem_id"),
                correct=candidate == "1",
                verdict="correct",
            )

    backend = MutatingMathJudge()
    monkeypatch.setattr(
        "r3bench.common.scoring_dispatch._production_backends",
        lambda *args, **kwargs: (backend, backend, backend),
    )
    rows, summary = score_saved_outputs_cli(
        domain="math",
        input_path=predictions,
        output_dir=tmp_path / "math-snapshot-score",
        data_source=str(resources / "examples/data/math"),
        split="test",
        strict=False,
        scoring_mode="production",
        profile_path=str(resources / "configs/scorer_profiles.yaml"),
        scorer_key="math_equivalence_judge",
    )
    assert rows[0]["parsed_output"] == "1"
    assert rows[0]["correct"] is True
    assert summary["input_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert predictions.read_bytes() != original_bytes


def test_coding_scoring_parses_and_hashes_one_input_snapshot(
    tmp_path: Path,
    resources: Path,
) -> None:
    predictions = tmp_path / "coding-snapshot.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "domain": "coding",
                "problem_id": "toy-code-1",
                "parsed_answer": "int main() { return 0; }",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    original_bytes = predictions.read_bytes()

    class MutatingCodingVerifier:
        def verify(self, problem: object, candidate: str) -> CodingVerifierResult:
            predictions.write_text(
                json.dumps(
                    {
                        "domain": "coding",
                        "problem_id": "toy-code-1",
                        "parsed_answer": "forged-after-read",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return CodingVerifierResult(
                upstream_id="toy-code-1",
                accepted="return 0" in candidate,
                verdict="accepted",
            )

    rows, summary = score_coding_saved_outputs(
        predictions_path=predictions,
        data_source=resources / "examples/data/coding.jsonl",
        output_dir=tmp_path / "coding-snapshot-score",
        mode="mock",
        strict=False,
        mock_verifier=MutatingCodingVerifier(),
    )
    assert rows[0]["accepted"] is True
    assert summary["input_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert predictions.read_bytes() != original_bytes


@pytest.mark.parametrize(
    ("domain", "data", "problem_ids", "scorer_key", "method"),
    [
        (
            "math",
            "examples/data/math",
            ("toy-math-1", "toy-math-2"),
            "math_equivalence_judge",
            "judge",
        ),
        (
            "abstract_reasoning",
            "examples/data/abstract_reasoning.jsonl",
            ("toy-ar-1", "toy-ar-2"),
            "abstract_reasoning_reasoning_gym",
            "score",
        ),
    ],
)
def test_production_downstream_failure_is_zero_and_next_problem_continues(
    tmp_path: Path,
    resources: Path,
    monkeypatch: pytest.MonkeyPatch,
    domain: str,
    data: str,
    problem_ids: tuple[str, str],
    scorer_key: str,
    method: str,
) -> None:
    predictions = tmp_path / f"{domain}-two.jsonl"
    predictions.write_text(
        "".join(
            json.dumps(
                {
                    "domain": domain,
                    "problem_id": problem_id,
                    "parsed_answer": "1" if index == 1 else "2",
                }
            )
            + "\n"
            for index, problem_id in enumerate(problem_ids, start=1)
        ),
        encoding="utf-8",
    )

    class FailFirst:
        calls = 0

        def _result(self, problem: object):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("synthetic malformed downstream response")
            if method == "judge":
                from r3bench.math.judge import MathJudgeResult

                return MathJudgeResult(
                    problem_id=getattr(problem, "problem_id"),
                    correct=True,
                    verdict="correct",
                )
            from r3bench.abstract_reasoning.scorer import ARScorerResult

            return ARScorerResult(
                problem_id=getattr(problem, "problem_id"),
                correct=True,
                score=1.0,
                verdict="correct",
            )

        def judge(self, problem: object, candidate: str):
            return self._result(problem)

        def score(self, problem: object, candidate: str):
            return self._result(problem)

    backend = FailFirst()
    monkeypatch.setattr(
        "r3bench.common.scoring_dispatch._production_backends",
        lambda *args, **kwargs: (backend, backend, backend),
    )
    rows, _ = score_saved_outputs_cli(
        domain=domain,
        input_path=predictions,
        output_dir=tmp_path / f"{domain}-scores",
        data_source=str(resources / data),
        split="test",
        strict=False,
        scoring_mode="production",
        profile_path=str(resources / "configs/scorer_profiles.yaml"),
        scorer_key=scorer_key,
    )
    assert rows[0]["judge_status"] == "judge_error"
    assert rows[0]["score"] == 0.0
    assert rows[0]["error_type"] == "ValueError"
    assert rows[1]["judge_status"] == "judged"
    assert rows[1]["score"] == 1.0
