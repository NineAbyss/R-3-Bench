from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from r3bench.cli import main as cli_main
from r3bench.commands.run_evaluation import _two_stage_profile


def test_mock_single_and_contest_are_bounded(
    tmp_path: Path, resources: Path
) -> None:
    single_output = tmp_path / "single"
    assert (
        cli_main(
            [
                "run",
                "--setting",
                "tool_free",
                "--domain",
                "coding",
                "--mode",
                "single_problem",
                "--model",
                "local-mock",
                "--data",
                "examples/data/coding.jsonl",
                "--output-dir",
                str(single_output),
                "--output-token-budget",
                "2048",
                "--provider",
                "mock",
                "--toy",
                "--limit-problems",
                "1",
            ]
        )
        == 0
    )
    contest_output = tmp_path / "contest"
    assert (
        cli_main(
            [
                "run",
                "--setting",
                "tool_free",
                "--domain",
                "coding",
                "--mode",
                "contest",
                "--model",
                "local-mock",
                "--data",
                "examples/data/coding.jsonl",
                "--output-dir",
                str(contest_output),
                "--output-token-budget",
                "4096",
                "--provider",
                "mock",
                "--toy",
                "--limit-suites",
                "1",
            ]
        )
        == 0
    )
    assert json.loads(
        (single_output / "run_summary.json").read_text(encoding="utf-8")
    )["attempt_count"] == 1
    assert json.loads(
        (contest_output / "run_summary.json").read_text(encoding="utf-8")
    )["problem_count"] == 6


@pytest.mark.parametrize(
    ("model", "run"),
    [
        ("qwen3.7-max", "qwen_coding"),
        ("deepseek-chat", "deepseek_coding"),
    ],
)
def test_real_provider_preview_is_network_free(
    tmp_path: Path,
    resources: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    run: str,
) -> None:
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    output = tmp_path / model
    code = cli_main(
        [
            "run",
            "--setting",
            "tool_free",
            "--domain",
            "coding",
            "--mode",
            "single_problem",
            "--model",
            model,
            "--data",
            "examples/data/coding.jsonl",
            "--output-dir",
            str(output),
            "--output-token-budget",
            "2048",
            "--provider",
            "real",
            "--model-profiles",
            str(resources / "configs/model_profiles.yaml"),
            "--evaluator-profiles",
            str(resources / "configs/evaluator_profiles.yaml"),
            "--run-profiles",
            str(resources / "configs/run_profiles.yaml"),
            "--run-profile",
            run,
            "--dry-run",
            "--toy",
            "--limit-problems",
            "1",
        ]
    )
    assert code == 0
    preview = json.loads(
        (output / "run_summary.json").read_text(encoding="utf-8")
    )
    assert preview["network_called"] is False
    assert preview["credential_read"] is False
    request_preview = json.loads(
        (output / "request_preview.json").read_text(encoding="utf-8")
    )
    assert request_preview["status"] == "dry_run"
    assert request_preview["request_count"] == 1
    if model == "deepseek-chat":
        payload = request_preview["requests"][0]["payload"]
        assert "reasoning_effort" not in payload
        assert payload["enable_thinking"] is False


def test_cli_metadata_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_main(["profiles", "validate", "--json"]) == 0
    profiles = json.loads(capsys.readouterr().out)
    assert len(profiles["supported_models"]) in {4, 8}
    assert len(profiles["reference_benchmark_models"]) == 4
    assert profiles["reference_cell_count"] == 96
    assert cli_main(["doctor"]) == 0
    doctor = json.loads(capsys.readouterr().out)
    assert doctor["status"] == "ready"


def test_public_cli_has_one_evaluation_and_analysis_surface(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli_main(["--help"]) == 0
    help_text = capsys.readouterr().out
    assert "run" in help_text
    assert "analysis" in help_text
    for removed in ("evaluate", "oracle", "smoke", "release"):
        assert removed not in help_text


def test_two_stage_response_curve_uses_regular_model_profile(
    resources: Path,
) -> None:
    profile = _two_stage_profile(
        Namespace(
            model="qwen3.7-max",
            domain="coding",
            mode="response_curve",
            two_stage_profile=None,
            two_stage_profiles=str(resources / "configs/two_stage_profiles.yaml"),
        )
    )
    assert profile.profile_id == "qwen_coding_two_stage"
