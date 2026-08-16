from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from r3bench.agentic.external_backend import (
    AgenticExecutionProfile,
    ExternalAgenticBackendError,
    load_external_backend_config,
    resolve_agentic_execution_profile,
    run_external_agentic_backend,
    validate_external_backend_handoff,
)
from r3bench.agentic.scoring_handoff import write_agentic_saved_outputs
from r3bench.agentic.task_export import export_agentic_tasks
from r3bench.commands.agentic_backend import main as agentic_backend_main
from r3bench.commands.analysis import _validate_official_provenance
from r3bench.commands.run_evaluation import main as run_evaluation_main
from r3bench.common.budget import resolve_official_budget_profile
from r3bench.common.scorer_registry import (
    load_scorer_profiles,
    scorer_profile_contract,
    scorer_profile_contract_sha256,
)
from r3bench.oracle.response_curve_schema import OracleSchemaError


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _valid_handoff(
    tmp_path: Path,
    resources: Path,
    *,
    model_key: str = "deepseek-chat",
    budget: int = 1,
    domain: str = "math",
) -> SimpleNamespace:
    task = export_agentic_tasks(
        domain=domain,
        data_source=(
            resources / "examples/data/coding.jsonl"
            if domain == "coding"
            else resources / "examples/data/math/problems.jsonl"
        ),
        output_dir=tmp_path / "tasks",
        budget=budget,
        strict_data=False,
    )[0]
    task_config = json.loads(
        (task.task_dir / "task_config.json").read_text(encoding="utf-8")
    )
    profile = resolve_agentic_execution_profile(
        model_key, resources / "configs/model_profiles.yaml"
    )
    scratch = tmp_path / "handoff"
    (scratch / "artifacts").mkdir(parents=True)
    summary = {
        "task_id": task.task_id,
        "suite_id": task_config["suite_id"],
        "domain": domain,
        "model_key": profile.model_key,
        "public_model_id": profile.public_model_id,
        "execution_profile": profile.to_dict(),
        "backend": "harbor",
        "environment": "docker",
        "agent": "terminus-2",
        "paper_equivalent_runtime": True,
        "os_command_execution_available": True,
        "os_commands_executed": 0,
        "container_runtime_called": True,
        "compilation_and_tests_available": True,
        "sandbox_limits_enforced": True,
        "sandbox_limits": task_config["sandbox_limits"],
        "model_api_called": True,
        "llm_call_count": 1,
        "correctness_feedback_exposed": False,
        "raw_trajectory_saved": True,
        "trajectory_complete": True,
        "trajectory_format": "ATIF",
    }
    llm_step: dict[str, Any] = {
        "step_id": 2,
        "source": "agent",
        "message": "done",
        "llm_call_count": 1,
        "model_name": profile.public_model_id,
    }
    if profile.reasoning_effort is not None:
        llm_step["reasoning_effort"] = profile.reasoning_effort
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "agent": {
            "name": "terminus-2",
            "version": "synthetic-v1",
            "model_name": profile.public_model_id,
            "execution_profile": profile.to_dict(),
        },
        "steps": [
            {"step_id": 1, "source": "user", "message": "task"},
            llm_step,
        ],
    }
    expected_artifacts = json.loads(
        (task.task_dir / "expected_artifacts.json").read_text(encoding="utf-8")
    )["artifacts"]
    manifest = {
        "schema_version": "1.0",
        "grade_after_episode": True,
        "correctness_feedback_exposed": False,
        "artifacts": [
            {
                "container_path": row["container_path"],
                "problem_label": row.get("problem_label"),
                "required": row["required"],
                "exists": False,
                "size_bytes": 0,
                "sha256": None,
                "artifact_relative_path": None,
            }
            for row in expected_artifacts
        ],
    }
    action = {
        "budget": budget,
        "used": 0,
        "policy": "compute_tools",
        "actions": [],
    }
    value = SimpleNamespace(
        task=task,
        task_config=task_config,
        profile=profile,
        scratch=scratch,
        summary=summary,
        trajectory=trajectory,
        manifest=manifest,
        action=action,
        expected_artifacts=expected_artifacts,
    )
    _persist_handoff(value)
    return value


def _persist_handoff(value: SimpleNamespace) -> None:
    _write_json(value.scratch / "backend_summary.json", value.summary)
    _write_json(value.scratch / "trajectory.json", value.trajectory)
    _write_json(value.scratch / "final_artifacts_manifest.json", value.manifest)
    _write_json(value.scratch / "public_action_log.json", value.action)


@pytest.mark.parametrize(
    ("model_key", "expected"),
    [
        (
            "qwen3.7-max",
            {
                "thinking_enabled": True,
                "reasoning_effort": {"state": "value", "value": "high"},
                "temperature": {"state": "value", "value": 0.2},
                "top_p": {"state": "value", "value": 0.8},
            },
        ),
        (
            "deepseek-chat",
            {
                "thinking_enabled": False,
                "reasoning_effort": {"state": "omitted"},
                "temperature": {"state": "value", "value": 0.0},
                "top_p": {"state": "value", "value": 1.0},
            },
        ),
        (
            "deepseek-reasoner",
            {
                "thinking_enabled": True,
                "reasoning_effort": {"state": "value", "value": "high"},
                "temperature": {"state": "omitted"},
                "top_p": {"state": "omitted"},
            },
        ),
        (
            "deepseek-v4-pro",
            {
                "thinking_enabled": True,
                "reasoning_effort": {"state": "value", "value": "high"},
                "temperature": {"state": "omitted"},
                "top_p": {"state": "omitted"},
            },
        ),
    ],
)
def test_active_models_have_canonical_agentic_execution_profiles(
    resources: Path,
    model_key: str,
    expected: dict[str, Any],
) -> None:
    profile = resolve_agentic_execution_profile(
        model_key, resources / "configs/model_profiles.yaml"
    )
    serialized = profile.to_dict()
    assert serialized == {
        "schema_version": "1.0",
        "model_key": model_key,
        "public_model_id": model_key,
        **expected,
    }
    assert json.loads(profile.to_json()) == serialized
    with pytest.raises(FrozenInstanceError):
        profile.model_key = "different"  # type: ignore[misc]


def test_agentic_backend_command_passes_resolved_execution_profile(
    tmp_path: Path,
    resources: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(
        "r3bench.commands.agentic_backend.run_external_agentic_backend",
        fake_run,
    )
    assert (
        agentic_backend_main(
            [
                "run",
                "--config",
                str(resources / "configs/agentic/external_backend.example.yaml"),
                "--task-dir",
                str(tmp_path / "task"),
                "--output-dir",
                str(tmp_path / "output"),
                "--model",
                "deepseek-reasoner",
                "--model-profiles",
                str(resources / "configs/model_profiles.yaml"),
                "--allow-real-api",
                "--allow-agentic-backend",
            ]
        )
        == 0
    )
    profile = captured["execution_profile"]
    assert isinstance(profile, AgenticExecutionProfile)
    assert profile.model_key == "deepseek-reasoner"
    assert "model_key" not in captured


def test_unified_agentic_run_passes_resolved_execution_profile(
    tmp_path: Path,
    resources: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "r3bench.commands.run_evaluation.load_external_backend_config",
        lambda path: object(),
    )

    def fake_run(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        output = Path(str(kwargs["output_dir"]))
        output.mkdir(parents=True)
        _write_json(output / "backend_summary.json", {})
        return {"status": "completed"}

    monkeypatch.setattr(
        "r3bench.commands.run_evaluation.run_external_agentic_backend",
        fake_run,
    )
    monkeypatch.setattr(
        "r3bench.commands.run_evaluation.write_agentic_saved_outputs",
        lambda *args, **kwargs: None,
    )
    assert (
        run_evaluation_main(
            [
                "--setting",
                "agentic",
                "--domain",
                "math",
                "--mode",
                "single_problem",
                "--model",
                "deepseek-v4-pro",
                "--data",
                "examples/data/math/problems.jsonl",
                "--output-dir",
                str(tmp_path / "evaluation"),
                "--counted-action-budget",
                "0",
                "--provider",
                "real",
                "--agentic-backend-config",
                "unused-by-test.yaml",
                "--model-profiles",
                str(resources / "configs/model_profiles.yaml"),
                "--allow-real-api",
                "--allow-agentic-backend",
                "--toy",
                "--limit-problems",
                "1",
                "--repeats",
                "1",
            ]
        )
        == 0
    )
    profile = captured["execution_profile"]
    assert isinstance(profile, AgenticExecutionProfile)
    assert profile.model_key == "deepseek-v4-pro"
    assert profile.temperature is None
    assert "model_key" not in captured


def test_external_adapter_argv_carries_canonical_execution_profile(
    tmp_path: Path,
    resources: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = export_agentic_tasks(
        domain="math",
        data_source=resources / "examples/data/math/problems.jsonl",
        output_dir=tmp_path / "tasks",
        budget=0,
        strict_data=False,
    )[0]
    template = load_external_backend_config(
        resources / "configs/agentic/external_backend.example.yaml"
    )
    config = replace(
        template,
        status="configured",
        executable="r3bench-harbor-adapter",
        credential_env="R3BENCH_TEST_API_KEY",
    )
    profile = resolve_agentic_execution_profile(
        "deepseek-reasoner", resources / "configs/model_profiles.yaml"
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "r3bench.agentic.external_backend.check_external_backend_readiness",
        lambda *args, **kwargs: {"status": "ready"},
    )

    def fake_process(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        scratch = Path(command[command.index("--output-dir") + 1])
        (scratch / "artifacts").mkdir()
        for name in (
            "backend_summary.json",
            "public_action_log.json",
            "final_artifacts_manifest.json",
            "trajectory.json",
        ):
            _write_json(scratch / name, {})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("r3bench.agentic.external_backend.subprocess.run", fake_process)

    def fake_validate(*args: object, **kwargs: object) -> dict[str, object]:
        captured["validated_profile"] = kwargs.get("expected_execution_profile")
        return {
            "trajectory_format": "ATIF",
            "trajectory_complete": True,
            "trajectory_sha256": "0" * 64,
        }

    monkeypatch.setattr(
        "r3bench.agentic.external_backend.validate_external_backend_handoff",
        fake_validate,
    )
    run_external_agentic_backend(
        task_dir=task.task_dir,
        output_dir=tmp_path / "output",
        execution_profile=profile,
        config=config,
        allow_real_api=True,
        allow_agentic_backend=True,
    )
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--model-key") + 1] == profile.model_key
    assert command[command.index("--model") + 1] == profile.public_model_id
    assert json.loads(command[command.index("--execution-profile-json") + 1]) == (
        profile.to_dict()
    )
    assert captured["validated_profile"] == profile
    assert {
        child.name for child in (tmp_path / "output" / "task_binding").iterdir()
    } == {
        "budget_config.json",
        "expected_artifacts.json",
        "instruction.md",
        "public_problem_manifest.json",
        "task_config.json",
    }


def test_handoff_rejects_summary_or_atif_execution_profile_mismatch(
    tmp_path: Path,
    resources: Path,
) -> None:
    task = export_agentic_tasks(
        domain="math",
        data_source=resources / "examples/data/math/problems.jsonl",
        output_dir=tmp_path / "tasks",
        budget=1,
        strict_data=False,
    )[0]
    task_config = json.loads(
        (task.task_dir / "task_config.json").read_text(encoding="utf-8")
    )
    profile = resolve_agentic_execution_profile(
        "deepseek-reasoner", resources / "configs/model_profiles.yaml"
    )
    scratch = tmp_path / "handoff"
    (scratch / "artifacts").mkdir(parents=True)
    summary = {
        "task_id": task.task_id,
        "suite_id": task_config["suite_id"],
        "domain": "math",
        "model_key": profile.model_key,
        "public_model_id": profile.public_model_id,
        "execution_profile": profile.to_dict(),
        "backend": "harbor",
        "environment": "docker",
        "agent": "terminus-2",
        "paper_equivalent_runtime": True,
        "os_command_execution_available": True,
        "os_commands_executed": 0,
        "container_runtime_called": True,
        "compilation_and_tests_available": True,
        "sandbox_limits_enforced": True,
        "sandbox_limits": task_config["sandbox_limits"],
        "model_api_called": True,
        "llm_call_count": 1,
        "correctness_feedback_exposed": False,
        "raw_trajectory_saved": True,
        "trajectory_complete": True,
        "trajectory_format": "ATIF",
    }
    _write_json(scratch / "backend_summary.json", summary)
    _write_json(
        scratch / "public_action_log.json",
        {"budget": 1, "used": 0, "policy": "compute_tools", "actions": []},
    )
    expected_artifacts = json.loads(
        (task.task_dir / "expected_artifacts.json").read_text(encoding="utf-8")
    )["artifacts"]
    _write_json(
        scratch / "final_artifacts_manifest.json",
        {
            "schema_version": "1.0",
            "grade_after_episode": True,
            "correctness_feedback_exposed": False,
            "artifacts": [
                {
                    "container_path": row["container_path"],
                    "problem_label": row.get("problem_label"),
                    "required": row["required"],
                    "exists": False,
                    "size_bytes": 0,
                    "sha256": None,
                    "artifact_relative_path": None,
                }
                for row in expected_artifacts
            ],
        },
    )
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "agent": {
            "name": "terminus-2",
            "version": "synthetic-v1",
            "model_name": profile.public_model_id,
            "execution_profile": profile.to_dict(),
        },
        "steps": [
            {"step_id": 1, "source": "user", "message": "task"},
            {
                "step_id": 2,
                "source": "agent",
                "message": "done",
                "llm_call_count": 1,
                "model_name": profile.public_model_id,
                "reasoning_effort": "high",
            },
        ],
    }
    _write_json(scratch / "trajectory.json", trajectory)
    validated = validate_external_backend_handoff(
        scratch,
        task.task_dir,
        expected_execution_profile=profile,
    )
    assert (
        validated["trajectory_sha256"]
        == hashlib.sha256((scratch / "trajectory.json").read_bytes()).hexdigest()
    )

    summary["execution_profile"]["temperature"] = {
        "state": "value",
        "value": 0.7,
    }
    _write_json(scratch / "backend_summary.json", summary)
    with pytest.raises(ExternalAgenticBackendError, match="summary execution profile"):
        validate_external_backend_handoff(
            scratch,
            task.task_dir,
            expected_execution_profile=profile,
        )

    summary["execution_profile"] = profile.to_dict()
    _write_json(scratch / "backend_summary.json", summary)
    trajectory["agent"]["execution_profile"]["top_p"] = {
        "state": "value",
        "value": 0.5,
    }
    _write_json(scratch / "trajectory.json", trajectory)
    with pytest.raises(ExternalAgenticBackendError, match="ATIF execution profile"):
        validate_external_backend_handoff(
            scratch,
            task.task_dir,
            expected_execution_profile=profile,
        )


def test_handoff_requires_real_llm_steps_and_exact_suite_binding(
    tmp_path: Path,
    resources: Path,
) -> None:
    value = _valid_handoff(tmp_path, resources, model_key="deepseek-reasoner")
    validate_external_backend_handoff(
        value.scratch,
        value.task.task_dir,
        expected_execution_profile=value.profile,
    )

    value.summary["suite_id"] = "wrong-suite"
    _persist_handoff(value)
    with pytest.raises(ExternalAgenticBackendError, match="suite binding"):
        validate_external_backend_handoff(
            value.scratch,
            value.task.task_dir,
            expected_execution_profile=value.profile,
        )
    value.summary["suite_id"] = value.task_config["suite_id"]

    value.summary["model_api_called"] = False
    _persist_handoff(value)
    with pytest.raises(ExternalAgenticBackendError, match="does not attest"):
        validate_external_backend_handoff(
            value.scratch,
            value.task.task_dir,
            expected_execution_profile=value.profile,
        )
    value.summary["model_api_called"] = True

    value.trajectory["steps"][1]["model_name"] = "wrong-model"
    _persist_handoff(value)
    with pytest.raises(ExternalAgenticBackendError, match="LLM step"):
        validate_external_backend_handoff(
            value.scratch,
            value.task.task_dir,
            expected_execution_profile=value.profile,
        )
    value.trajectory["steps"][1]["model_name"] = value.profile.public_model_id

    value.trajectory["steps"][1]["reasoning_effort"] = "low"
    _persist_handoff(value)
    with pytest.raises(ExternalAgenticBackendError, match="LLM step"):
        validate_external_backend_handoff(
            value.scratch,
            value.task.task_dir,
            expected_execution_profile=value.profile,
        )
    value.trajectory["steps"][1]["reasoning_effort"] = "high"

    value.trajectory["steps"][1]["llm_call_count"] = 0
    value.trajectory["steps"][1].pop("model_name")
    value.trajectory["steps"][1].pop("reasoning_effort")
    value.summary["llm_call_count"] = 0
    _persist_handoff(value)
    with pytest.raises(ExternalAgenticBackendError, match="no recorded model API call"):
        validate_external_backend_handoff(
            value.scratch,
            value.task.task_dir,
            expected_execution_profile=value.profile,
        )


def test_public_actions_follow_atif_linear_order(
    tmp_path: Path,
    resources: Path,
) -> None:
    value = _valid_handoff(tmp_path, resources)
    calls = [
        {
            "tool_call_id": "status-one",
            "function_name": "contest_status",
            "arguments": {},
        },
        {
            "tool_call_id": "budget-two",
            "function_name": "remaining_budget",
            "arguments": {},
        },
    ]
    value.trajectory["steps"][1]["tool_calls"] = calls
    value.action["actions"] = [
        {
            "sequence": index,
            "source_step_id": 2,
            "tool_call_id": call["tool_call_id"],
            "function_name": call["function_name"],
            "command": call["function_name"],
            "classified_as": "free_status_action",
            "action_class": "free_status_action",
            "allowed": True,
            "counted": False,
            "executed": True,
            "budget_consumed": 0,
            "budget_before": 1,
            "budget_after": 1,
            "reason": "free_action_accepted",
            "active_problem_id": None,
            "attributed_problem_id": None,
        }
        for index, call in enumerate(calls, start=1)
    ]
    value.summary["os_commands_executed"] = 2
    _persist_handoff(value)
    validate_external_backend_handoff(
        value.scratch,
        value.task.task_dir,
        expected_execution_profile=value.profile,
    )

    value.action["actions"].reverse()
    for sequence, row in enumerate(value.action["actions"], start=1):
        row["sequence"] = sequence
    _persist_handoff(value)
    with pytest.raises(ExternalAgenticBackendError, match="tool-call order"):
        validate_external_backend_handoff(
            value.scratch,
            value.task.task_dir,
            expected_execution_profile=value.profile,
        )


def test_action_replay_attests_scope_and_budget_blocking(
    tmp_path: Path,
    resources: Path,
) -> None:
    value = _valid_handoff(tmp_path, resources, budget=0)
    active_problem_id = value.task_config["problem_labels"]["A"]
    calls = [
        {
            "tool_call_id": "unfocused",
            "function_name": "shell",
            "arguments": {"command": "python analyze.py"},
        },
        {
            "tool_call_id": "focus",
            "function_name": "focus_problem",
            "arguments": {"problem_id": "A"},
        },
        {
            "tool_call_id": "cross-problem",
            "function_name": "shell",
            "arguments": {"command": "python analyze.py /logs/problem_B/input.txt"},
        },
        {
            "tool_call_id": "over-budget",
            "function_name": "shell",
            "arguments": {"command": "python analyze.py"},
        },
    ]
    value.trajectory["steps"][1]["tool_calls"] = calls
    value.action.update(
        {
            "used": 0,
            "remaining": 0,
            "blocked_attempts": 3,
            "action_attempts": 4,
            "actions": [
                {
                    "sequence": 1,
                    "source_step_id": 2,
                    "tool_call_id": "unfocused",
                    "function_name": "shell",
                    "command": "python analyze.py",
                    "classified_as": "counted_tool_action",
                    "action_class": "blocked_action",
                    "allowed": False,
                    "counted": True,
                    "executed": False,
                    "budget_consumed": 0,
                    "budget_before": 0,
                    "budget_after": 0,
                    "reason": "counted_action_requires_active_focus",
                    "active_problem_id": None,
                    "attributed_problem_id": None,
                },
                {
                    "sequence": 2,
                    "source_step_id": 2,
                    "tool_call_id": "focus",
                    "function_name": "focus_problem",
                    "command": "focus_problem A",
                    "classified_as": "free_bookkeeping_action",
                    "action_class": "free_bookkeeping_action",
                    "allowed": True,
                    "counted": False,
                    "executed": True,
                    "budget_consumed": 0,
                    "budget_before": 0,
                    "budget_after": 0,
                    "reason": "free_action_accepted",
                    "active_problem_id": active_problem_id,
                    "attributed_problem_id": None,
                },
                {
                    "sequence": 3,
                    "source_step_id": 2,
                    "tool_call_id": "cross-problem",
                    "function_name": "shell",
                    "command": "python analyze.py /logs/problem_B/input.txt",
                    "classified_as": "counted_tool_action",
                    "action_class": "blocked_action",
                    "allowed": False,
                    "counted": True,
                    "executed": False,
                    "budget_consumed": 0,
                    "budget_before": 0,
                    "budget_after": 0,
                    "reason": "cross_problem_access_blocked:B",
                    "active_problem_id": active_problem_id,
                    "attributed_problem_id": None,
                },
                {
                    "sequence": 4,
                    "source_step_id": 2,
                    "tool_call_id": "over-budget",
                    "function_name": "shell",
                    "command": "python analyze.py",
                    "classified_as": "counted_tool_action",
                    "action_class": "blocked_action",
                    "allowed": False,
                    "counted": True,
                    "executed": False,
                    "budget_consumed": 0,
                    "budget_before": 0,
                    "budget_after": 0,
                    "reason": "counted_action_budget_exhausted",
                    "active_problem_id": active_problem_id,
                    "attributed_problem_id": None,
                },
            ],
        }
    )
    value.summary["os_commands_executed"] = 1
    _persist_handoff(value)
    validate_external_backend_handoff(
        value.scratch,
        value.task.task_dir,
        expected_execution_profile=value.profile,
    )

    value.action["actions"][2]["action_class"] = "counted_tool_action"
    _persist_handoff(value)
    with pytest.raises(ExternalAgenticBackendError, match="problem scope, or budget"):
        validate_external_backend_handoff(
            value.scratch,
            value.task.task_dir,
            expected_execution_profile=value.profile,
        )


def test_handoff_rejects_symlinked_artifact_root_when_all_artifacts_absent(
    tmp_path: Path,
    resources: Path,
) -> None:
    value = _valid_handoff(tmp_path, resources)
    artifact_root = value.scratch / "artifacts"
    outside = tmp_path / "outside-artifacts"
    artifact_root.rename(outside)
    artifact_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ExternalAgenticBackendError, match="artifact directory"):
        validate_external_backend_handoff(
            value.scratch,
            value.task.task_dir,
            expected_execution_profile=value.profile,
        )


def test_present_artifact_requires_write_evidence_and_rejects_symlink(
    tmp_path: Path,
    resources: Path,
) -> None:
    value = _valid_handoff(tmp_path, resources)
    content = "<answer>1</answer>\n"
    relative = "artifacts/logs/artifacts/answer.txt"
    artifact = value.scratch / relative
    artifact.parent.mkdir(parents=True)
    artifact.write_text(content, encoding="utf-8")
    row = value.manifest["artifacts"][0]
    row.update(
        {
            "exists": True,
            "size_bytes": len(content.encode("utf-8")),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "artifact_relative_path": relative,
        }
    )
    command = "printf '<answer>1</answer>\\n' > /logs/artifacts/answer.txt"
    value.trajectory["steps"][1]["tool_calls"] = [
        {
            "tool_call_id": "write-shell",
            "function_name": "shell",
            "arguments": {"command": command},
        }
    ]
    value.action["actions"] = [
        {
            "sequence": 1,
            "source_step_id": 2,
            "tool_call_id": "write-shell",
            "function_name": "shell",
            "command": command,
            "classified_as": "free_finalization_action",
            "action_class": "free_finalization_action",
            "allowed": True,
            "counted": False,
            "executed": True,
            "budget_consumed": 0,
            "budget_before": 1,
            "budget_after": 1,
            "reason": "free_action_accepted",
            "active_problem_id": None,
            "attributed_problem_id": None,
        }
    ]
    value.summary["os_commands_executed"] = 1
    _persist_handoff(value)
    validate_external_backend_handoff(
        value.scratch,
        value.task.task_dir,
        expected_execution_profile=value.profile,
    )

    mismatched_command = (
        "printf '<answer>different</answer>\\n' > /logs/artifacts/answer.txt"
    )
    value.trajectory["steps"][1]["tool_calls"][0]["arguments"]["command"] = (
        mismatched_command
    )
    value.action["actions"][0]["command"] = mismatched_command
    _persist_handoff(value)
    with pytest.raises(ExternalAgenticBackendError, match="artifact digest"):
        validate_external_backend_handoff(
            value.scratch,
            value.task.task_dir,
            expected_execution_profile=value.profile,
        )

    unverifiable_command = "write_final_artifact /logs/artifacts/answer.txt"
    value.trajectory["steps"][1]["tool_calls"][0]["arguments"]["command"] = (
        unverifiable_command
    )
    value.action["actions"][0]["command"] = unverifiable_command
    _persist_handoff(value)
    with pytest.raises(ExternalAgenticBackendError, match="verifiable content"):
        validate_external_backend_handoff(
            value.scratch,
            value.task.task_dir,
            expected_execution_profile=value.profile,
        )

    value.trajectory["steps"][1].pop("tool_calls")
    value.action["actions"] = []
    value.summary["os_commands_executed"] = 0
    _persist_handoff(value)
    with pytest.raises(ExternalAgenticBackendError, match="no executed write event"):
        validate_external_backend_handoff(
            value.scratch,
            value.task.task_dir,
            expected_execution_profile=value.profile,
        )

    outside = tmp_path / "outside-answer.txt"
    outside.write_text(content, encoding="utf-8")
    artifact.unlink()
    artifact.symlink_to(outside)
    _persist_handoff(value)
    with pytest.raises(ExternalAgenticBackendError, match="symlink"):
        validate_external_backend_handoff(
            value.scratch,
            value.task.task_dir,
            expected_execution_profile=value.profile,
        )


@pytest.mark.parametrize(
    ("command", "content"),
    [
        (
            "echo '<answer>echo</answer>' > /logs/artifacts/answer.txt",
            "<answer>echo</answer>\n",
        ),
        (
            "cat > /logs/artifacts/answer.txt <<'R3BENCH_EOF'\n"
            "<answer>heredoc</answer>\n"
            "R3BENCH_EOF",
            "<answer>heredoc</answer>\n",
        ),
    ],
)
def test_shell_final_write_evidence_supports_exact_literal_writers(
    tmp_path: Path,
    resources: Path,
    command: str,
    content: str,
) -> None:
    value = _valid_handoff(tmp_path, resources)
    relative = "artifacts/logs/artifacts/answer.txt"
    artifact = value.scratch / relative
    artifact.parent.mkdir(parents=True)
    artifact.write_text(content, encoding="utf-8")
    value.manifest["artifacts"][0].update(
        {
            "exists": True,
            "size_bytes": len(content.encode("utf-8")),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "artifact_relative_path": relative,
        }
    )
    value.trajectory["steps"][1]["tool_calls"] = [
        {
            "tool_call_id": "write-shell",
            "function_name": "shell",
            "arguments": {"command": command},
        }
    ]
    value.action["actions"] = [
        {
            "sequence": 1,
            "source_step_id": 2,
            "tool_call_id": "write-shell",
            "function_name": "shell",
            "command": command,
            "classified_as": "free_finalization_action",
            "action_class": "free_finalization_action",
            "allowed": True,
            "counted": False,
            "executed": True,
            "budget_consumed": 0,
            "budget_before": 1,
            "budget_after": 1,
            "reason": "free_action_accepted",
            "active_problem_id": None,
            "attributed_problem_id": None,
        }
    ]
    value.summary["os_commands_executed"] = 1
    _persist_handoff(value)
    validate_external_backend_handoff(
        value.scratch,
        value.task.task_dir,
        expected_execution_profile=value.profile,
    )


def test_artifact_manifest_rejects_public_binding_and_relative_aliases(
    tmp_path: Path,
    resources: Path,
) -> None:
    value = _valid_handoff(tmp_path, resources, domain="coding")
    value.manifest["artifacts"][0]["problem_label"] = "B"
    _persist_handoff(value)
    with pytest.raises(ExternalAgenticBackendError, match="public binding"):
        validate_external_backend_handoff(
            value.scratch,
            value.task.task_dir,
            expected_execution_profile=value.profile,
        )

    value.manifest["artifacts"][0]["problem_label"] = "A"
    for index in (0, 1):
        expected = value.expected_artifacts[index]
        relative = f"artifacts/{expected['sandbox_relative_path']}"
        path = value.scratch / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"// solution {index}\n"
        path.write_text(content, encoding="utf-8")
        value.manifest["artifacts"][index].update(
            {
                "exists": True,
                "size_bytes": len(content.encode("utf-8")),
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "artifact_relative_path": relative,
            }
        )
    value.manifest["artifacts"][1]["artifact_relative_path"] = value.manifest[
        "artifacts"
    ][0]["artifact_relative_path"]
    _persist_handoff(value)
    with pytest.raises(ExternalAgenticBackendError, match="metadata is inconsistent"):
        validate_external_backend_handoff(
            value.scratch,
            value.task.task_dir,
            expected_execution_profile=value.profile,
        )


def test_official_analysis_rechecks_agentic_execution_attestation(
    tmp_path: Path,
    resources: Path,
) -> None:
    profile = resolve_agentic_execution_profile("deepseek-chat")
    budget = resolve_official_budget_profile(
        "agentic_math_deepseek_chat_budgeted_rho_0p2",
        setting="agentic",
        domain="math",
        model_key="deepseek-chat",
    )
    value = _valid_handoff(
        tmp_path,
        resources,
        budget=int(budget.budget_value or 0),
    )
    handoff = validate_external_backend_handoff(
        value.scratch,
        value.task.task_dir,
        expected_execution_profile=value.profile,
    )
    value.summary.update(handoff)
    value.summary.update(
        {
            "execution_id": "execution-one",
            "budget_resolution": {
                "condition_kind": "official_profile",
                "profile_id": budget.profile_id,
                "unit": "counted_actions",
                "value": budget.budget_value,
            },
        }
    )
    _persist_handoff(value)
    run_dir = value.scratch
    task_binding = run_dir / "task_binding"
    task_binding.mkdir()
    for source in value.task.task_dir.iterdir():
        shutil.copy2(source, task_binding / source.name)
    write_agentic_saved_outputs(run_dir, run_dir / "saved_outputs.jsonl")

    scoring_dir = tmp_path / "scoring"
    scoring_dir.mkdir()
    saved_path = run_dir / "saved_outputs.jsonl"
    saved_rows = [
        json.loads(line)
        for line in saved_path.read_text(encoding="utf-8").splitlines()
    ]
    scorer = load_scorer_profiles()["math_equivalence_judge"]
    results_path = scoring_dir / "judge_results.jsonl"
    results_path.write_text(
        "".join(
            json.dumps(
                {
                    **row,
                    "parse_status": "missing",
                    "judge_status": "not_judged",
                    "correct": False,
                    "score": 0.0,
                    "scoring_mode": "production",
                    "evaluator": scorer.profile_id,
                }
            )
            + "\n"
            for row in saved_rows
        ),
        encoding="utf-8",
    )
    scoring_summary = {
        "status": "complete",
        "domain": "math",
        "scoring_mode": "production",
        "scorer_profile": scorer.profile_id,
        "scorer_contract": scorer_profile_contract(scorer),
        "scorer_contract_sha256": scorer_profile_contract_sha256(scorer),
        "input_sha256": hashlib.sha256(saved_path.read_bytes()).hexdigest(),
        "results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
    }
    scoring_summary_path = scoring_dir / "scoring_summary.json"
    _write_json(scoring_summary_path, scoring_summary)
    args = SimpleNamespace(
        setting="agentic",
        domain="math",
        model="deepseek-chat",
        run_dir=str(run_dir),
        scoring_dir=str(scoring_dir),
    )
    _validate_official_provenance(
        args, budget, expected_budget=int(budget.budget_value or 0)
    )

    trajectory_path = run_dir / "trajectory.json"
    trajectory_bytes = trajectory_path.read_bytes()
    trajectory_path.unlink()
    with pytest.raises(OracleSchemaError, match="fully revalidated"):
        _validate_official_provenance(
            args, budget, expected_budget=int(budget.budget_value or 0)
        )
    trajectory_path.write_bytes(trajectory_bytes)

    value.summary["trajectory_sha256"] = "Z" * 64
    _write_json(run_dir / "backend_summary.json", value.summary)
    with pytest.raises(OracleSchemaError, match="trajectory digest"):
        _validate_official_provenance(
            args, budget, expected_budget=int(budget.budget_value or 0)
        )
    value.summary["trajectory_sha256"] = handoff["trajectory_sha256"]
    _write_json(run_dir / "backend_summary.json", value.summary)

    action_path = run_dir / "public_action_log.json"
    action_bytes = action_path.read_bytes()
    action = json.loads(action_bytes)
    action["used"] = 1
    _write_json(action_path, action)
    with pytest.raises(OracleSchemaError, match="fully revalidated"):
        _validate_official_provenance(
            args, budget, expected_budget=int(budget.budget_value or 0)
        )
    action_path.write_bytes(action_bytes)

    manifest_path = run_dir / "final_artifacts_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest["artifacts"][0]["problem_label"] = "A"
    _write_json(manifest_path, manifest)
    with pytest.raises(OracleSchemaError, match="fully revalidated"):
        _validate_official_provenance(
            args, budget, expected_budget=int(budget.budget_value or 0)
        )
    manifest_path.write_bytes(manifest_bytes)

    extra_artifact = run_dir / "artifacts" / "unmanifested.txt"
    extra_artifact.write_text("not declared\n", encoding="utf-8")
    with pytest.raises(OracleSchemaError, match="fully revalidated"):
        _validate_official_provenance(
            args, budget, expected_budget=int(budget.budget_value or 0)
        )
    extra_artifact.unlink()

    instruction_path = task_binding / "instruction.md"
    instruction_bytes = instruction_path.read_bytes()
    instruction_path.write_bytes(instruction_bytes + b"tampered\n")
    with pytest.raises(OracleSchemaError, match="fully revalidated"):
        _validate_official_provenance(
            args, budget, expected_budget=int(budget.budget_value or 0)
        )
    instruction_path.write_bytes(instruction_bytes)

    saved_bytes = saved_path.read_bytes()
    results_bytes = results_path.read_bytes()
    scoring_summary_bytes = scoring_summary_path.read_bytes()
    forged_saved = [dict(row) for row in saved_rows]
    forged_saved[0]["parsed_answer"] = "forged"
    saved_path.write_text(
        "".join(json.dumps(row) + "\n" for row in forged_saved),
        encoding="utf-8",
    )
    forged_results = [
        json.loads(line)
        for line in results_bytes.decode("utf-8").splitlines()
    ]
    forged_results[0]["parsed_answer"] = "forged"
    results_path.write_text(
        "".join(json.dumps(row) + "\n" for row in forged_results),
        encoding="utf-8",
    )
    scoring_summary["input_sha256"] = hashlib.sha256(
        saved_path.read_bytes()
    ).hexdigest()
    scoring_summary["results_sha256"] = hashlib.sha256(
        results_path.read_bytes()
    ).hexdigest()
    _write_json(scoring_summary_path, scoring_summary)
    with pytest.raises(OracleSchemaError, match="validated final artifacts"):
        _validate_official_provenance(
            args, budget, expected_budget=int(budget.budget_value or 0)
        )
    saved_path.write_bytes(saved_bytes)
    results_path.write_bytes(results_bytes)
    scoring_summary_path.write_bytes(scoring_summary_bytes)

    value.summary["execution_profile"]["temperature"] = {"state": "omitted"}
    _write_json(run_dir / "backend_summary.json", value.summary)
    with pytest.raises(OracleSchemaError, match="Harbor/Terminus-2 provenance"):
        _validate_official_provenance(
            args, budget, expected_budget=int(budget.budget_value or 0)
        )
