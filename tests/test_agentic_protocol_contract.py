from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from r3bench.agentic.external_backend import (
    ExternalAgenticBackendError,
    check_external_backend_readiness,
    load_external_backend_config,
    resolve_agentic_execution_profile,
    run_external_agentic_backend,
    validate_external_backend_handoff,
)
from r3bench.agentic.protocol_contract import PAPER_SANDBOX_LIMITS
from r3bench.agentic.scoring_handoff import extract_agentic_answer_sections
from r3bench.agentic.task_export import (
    AgenticTaskExportError,
    export_agentic_response_curve_tasks,
    export_agentic_tasks,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_formal_coding_export_uses_compute_tools_and_free_writes(
    tmp_path: Path, resources: Path
) -> None:
    task = export_agentic_tasks(
        domain="coding",
        data_source=resources / "examples/data/coding.jsonl",
        output_dir=tmp_path / "tasks",
        budget=3,
        strict_data=False,
    )[0]
    task_config = json.loads(
        (task.task_dir / "task_config.json").read_text(encoding="utf-8")
    )
    budget = json.loads(
        (task.task_dir / "budget_config.json").read_text(encoding="utf-8")
    )
    assert task_config["runtime"] == "harbor_terminus2_paper_v1"
    assert task_config["offline_test_runtime"] == "offline_mock_replay_v1"
    assert task_config["sandbox_limits"] == PAPER_SANDBOX_LIMITS["coding"]
    assert task_config["action_policy"] == "compute_tools"
    assert budget["policy"] == "compute_tools"
    assert budget["final_artifact_write_counted"] is False
    assert "pure_file_write" in budget["free_categories"]
    instruction = (task.task_dir / "instruction.md").read_text(encoding="utf-8")
    assert "pure file-write" in instruction
    assert "Scratch-file construction" not in instruction

    math_task = export_agentic_tasks(
        domain="math",
        data_source=resources / "examples/data/math/problems.jsonl",
        output_dir=tmp_path / "math-tasks",
        budget=3,
        strict_data=False,
    )[0]
    math_instruction = (math_task.task_dir / "instruction.md").read_text(
        encoding="utf-8"
    )
    assert "Pure direct-text" in math_instruction
    assert "Scratch-file construction" not in math_instruction
    with pytest.raises(AgenticTaskExportError, match="every domain"):
        export_agentic_tasks(
            domain="coding",
            data_source=resources / "examples/data/coding.jsonl",
            output_dir=tmp_path / "invalid",
            budget=3,
            strict_data=False,
            action_policy="all_nonfree",
        )


def test_response_curve_preserves_repeated_caps_and_repeat_identity(
    tmp_path: Path, resources: Path
) -> None:
    tasks = export_agentic_response_curve_tasks(
        domain="coding",
        data_source=resources / "examples/data/coding.jsonl",
        output_dir=tmp_path / "curve",
        budgets=(0, 1, 1, 2, 4, 8),
        repeat_ids=(1, 2),
        limit_problems=1,
        confirm_full_curve=True,
        strict_data=False,
    )
    assert len(tasks) == 12
    assert len({task.task_id for task in tasks}) == 12
    assert [(task.budget_level, task.counted_action_budget) for task in tasks[::2]] == [
        (1, 0),
        (2, 1),
        (3, 1),
        (4, 2),
        (5, 4),
        (6, 8),
    ]
    assert {task.repeat_id for task in tasks} == {1, 2}
    for task in tasks:
        config = json.loads(
            (task.task_dir / "budget_config.json").read_text(encoding="utf-8")
        )
        assert config["budget_level"] == task.budget_level
        assert config["repeat_id"] == task.repeat_id

    default_repeats = export_agentic_response_curve_tasks(
        domain="coding",
        data_source=resources / "examples/data/coding.jsonl",
        output_dir=tmp_path / "default-repeats",
        budgets=(0,),
        limit_problems=1,
        strict_data=False,
    )
    assert [task.repeat_id for task in default_repeats] == [1, 2, 3, 4, 5]


def test_math_and_ar_agentic_answer_contracts_are_domain_specific() -> None:
    math = """## Problem A
Work. \\boxed{42}
## Problem B
<answer>not a Math box</answer>
## Problem C
\\boxed{first} and \\boxed{second}
"""
    assert extract_agentic_answer_sections("math", math, ("A", "B", "C")) == {
        "A": "42"
    }
    assert extract_agentic_answer_sections("math", r"\boxed{7}", ("A",)) == {
        "A": "7"
    }

    abstract_reasoning = """## Problem A
<answer>blue square</answer>
## Problem B
<answer B>old labeled format</answer B>
## Problem C
\\boxed{not an AR tag}
"""
    assert extract_agentic_answer_sections(
        "abstract_reasoning", abstract_reasoning, ("A", "B", "C")
    ) == {"A": "blue square"}
    assert extract_agentic_answer_sections(
        "abstract_reasoning", "<answer>circle</answer>", ("A",)
    ) == {"A": "circle"}


def test_harbor_handoff_requires_limits_and_complete_atif_trajectory(
    tmp_path: Path, resources: Path
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
    execution_profile = resolve_agentic_execution_profile("deepseek-chat")
    scratch = tmp_path / "handoff"
    (scratch / "artifacts").mkdir(parents=True)
    _write_json(
        scratch / "backend_summary.json",
        {
            "task_id": task.task_id,
            "suite_id": task_config["suite_id"],
            "domain": "math",
            "model_key": "deepseek-chat",
            "public_model_id": execution_profile.public_model_id,
            "execution_profile": execution_profile.to_dict(),
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
        },
    )
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
            ]
        },
    )
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "agent": {
            "name": "terminus-2",
            "version": "synthetic-v1",
            "model_name": "deepseek-chat",
            "execution_profile": execution_profile.to_dict(),
        },
        "steps": [
            {"step_id": 1, "source": "user", "message": "task"},
            {
                "step_id": 2,
                "source": "agent",
                "message": "done",
                "llm_call_count": 1,
                "model_name": "deepseek-chat",
            },
        ],
    }
    _write_json(scratch / "trajectory.json", trajectory)
    validated = validate_external_backend_handoff(
        scratch, task.task_dir, expected_model_key="deepseek-chat"
    )
    assert validated == {
        "trajectory_format": "ATIF",
        "trajectory_complete": True,
        "trajectory_sha256": hashlib.sha256(
            (scratch / "trajectory.json").read_bytes()
        ).hexdigest(),
    }

    trajectory["agent"]["model_name"] = "wrong-model"
    _write_json(scratch / "trajectory.json", trajectory)
    with pytest.raises(ExternalAgenticBackendError, match="requested model"):
        validate_external_backend_handoff(
            scratch, task.task_dir, expected_model_key="deepseek-chat"
        )
    trajectory["agent"]["model_name"] = "deepseek-chat"

    del trajectory["agent"]["version"]
    _write_json(scratch / "trajectory.json", trajectory)
    with pytest.raises(ExternalAgenticBackendError, match="agent version"):
        validate_external_backend_handoff(
            scratch, task.task_dir, expected_model_key="deepseek-chat"
        )
    trajectory["agent"]["version"] = "synthetic-v1"

    del trajectory["steps"][1]["message"]
    _write_json(scratch / "trajectory.json", trajectory)
    with pytest.raises(ExternalAgenticBackendError, match="step message"):
        validate_external_backend_handoff(
            scratch, task.task_dir, expected_model_key="deepseek-chat"
        )
    trajectory["steps"][1]["message"] = "done"

    trajectory["steps"][1]["tool_calls"] = [
        {
            "tool_call_id": "call-1",
            "function_name": "shell",
            "arguments": {"command": "g++ solution.cpp"},
        }
    ]
    _write_json(scratch / "trajectory.json", trajectory)
    _write_json(
        scratch / "public_action_log.json",
        {
            "budget": 1,
            "used": 0,
            "policy": "compute_tools",
            "actions": [
                {
                    "sequence": 1,
                    "source_step_id": 2,
                    "tool_call_id": "call-1",
                    "function_name": "shell",
                    "command": "g++ solution.cpp",
                    "action_class": "free_environment_action",
                    "counted": False,
                    "executed": False,
                    "budget_consumed": 0,
                }
            ],
        },
    )
    with pytest.raises(ExternalAgenticBackendError, match="compute_tools"):
        validate_external_backend_handoff(
            scratch, task.task_dir, expected_model_key="deepseek-chat"
        )

    summary = json.loads(
        (scratch / "backend_summary.json").read_text(encoding="utf-8")
    )
    summary["os_commands_executed"] = 1
    _write_json(scratch / "backend_summary.json", summary)
    _write_json(
        scratch / "public_action_log.json",
        {
            "budget": 1,
            "used": 1,
            "policy": "compute_tools",
            "actions": [
                {
                    "sequence": 1,
                    "source_step_id": 2,
                    "tool_call_id": "call-1",
                    "function_name": "shell",
                    "command": "g++ solution.cpp",
                    "action_class": "counted_tool_action",
                    "counted": True,
                    "allowed": True,
                    "executed": True,
                    "budget_consumed": 1,
                    "active_problem_id": None,
                    "attributed_problem_id": "toy-math-1",
                }
            ],
        },
    )
    with pytest.raises(ExternalAgenticBackendError, match="problem scope"):
        validate_external_backend_handoff(
            scratch, task.task_dir, expected_model_key="deepseek-chat"
        )

    trajectory["steps"][1]["tool_calls"] = [
        {
            "tool_call_id": "focus-1",
            "function_name": "focus_problem",
            "arguments": {"problem_id": "A"},
        },
        {
            "tool_call_id": "call-2",
            "function_name": "shell",
            "arguments": {"command": "python analyze.py /logs/problem_B/input.txt"},
        },
    ]
    _write_json(scratch / "trajectory.json", trajectory)
    summary["os_commands_executed"] = 2
    _write_json(scratch / "backend_summary.json", summary)
    _write_json(
        scratch / "public_action_log.json",
        {
            "budget": 1,
            "used": 1,
            "policy": "compute_tools",
            "actions": [
                {
                    "sequence": 1,
                    "source_step_id": 2,
                    "tool_call_id": "focus-1",
                    "function_name": "focus_problem",
                    "command": "focus_problem A",
                    "action_class": "free_bookkeeping_action",
                    "counted": False,
                    "allowed": True,
                    "executed": True,
                    "budget_consumed": 0,
                    "active_problem_id": "toy-math-1",
                    "attributed_problem_id": None,
                },
                {
                    "sequence": 2,
                    "source_step_id": 2,
                    "tool_call_id": "call-2",
                    "function_name": "shell",
                    "command": "python analyze.py /logs/problem_B/input.txt",
                    "action_class": "counted_tool_action",
                    "counted": True,
                    "allowed": True,
                    "executed": True,
                    "budget_consumed": 1,
                    "active_problem_id": "toy-math-1",
                    "attributed_problem_id": "toy-math-1",
                },
            ],
        },
    )
    with pytest.raises(ExternalAgenticBackendError, match="problem scope"):
        validate_external_backend_handoff(
            scratch, task.task_dir, expected_model_key="deepseek-chat"
        )

    trajectory["steps"][1]["tool_calls"] = [
        {
            "tool_call_id": "mystery-1",
            "function_name": "mystery_solver",
            "arguments": {"command": "focus_problem A"},
        }
    ]
    _write_json(scratch / "trajectory.json", trajectory)
    with pytest.raises(ExternalAgenticBackendError, match="unsupported native tool"):
        validate_external_backend_handoff(
            scratch, task.task_dir, expected_model_key="deepseek-chat"
        )

    trajectory["steps"][1].pop("tool_calls")
    _write_json(scratch / "trajectory.json", trajectory)
    summary["os_commands_executed"] = 0
    _write_json(scratch / "backend_summary.json", summary)
    _write_json(
        scratch / "public_action_log.json",
        {"budget": 1, "used": 0, "policy": "compute_tools", "actions": []},
    )

    trajectory["agent"] = {"name": "mock-agent"}
    _write_json(scratch / "trajectory.json", trajectory)
    with pytest.raises(ExternalAgenticBackendError, match="Terminus-2"):
        validate_external_backend_handoff(
            scratch, task.task_dir, expected_model_key="deepseek-chat"
        )


def test_external_backend_example_encodes_table_14_limits(resources: Path) -> None:
    config = load_external_backend_config(
        resources / "configs/agentic/external_backend.example.yaml"
    )
    assert config.timeout_seconds == 7200
    assert dict(config.domain_sandbox_limits["coding"]) == {
        "agent_timeout_seconds": 7200,
        "build_timeout_seconds": 600,
        "memory_mb": 2048,
        "storage_mb": 10240,
    }
    assert dict(config.domain_sandbox_limits["math"]) == {
        "agent_timeout_seconds": 7200,
        "verifier_timeout_seconds": 7200,
        "cpu_count": 1,
        "memory_mb": 2048,
        "storage_mb": 10240,
    }


def test_harbor_capability_probe_requires_real_runtime_features(
    resources: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = load_external_backend_config(
        resources / "configs/agentic/external_backend.example.yaml"
    )
    config = replace(
        template,
        status="configured",
        executable="r3bench-harbor-adapter",
        credential_env="R3BENCH_TEST_API_KEY",
    )
    monkeypatch.setenv("R3BENCH_TEST_API_KEY", "synthetic-test-value")
    monkeypatch.setattr(
        "r3bench.agentic.external_backend.shutil.which",
        lambda command: f"/test-bin/{command}",
    )
    capabilities = {
        "r3bench_agentic_protocol": "2.0",
        "backend": "harbor",
        "environment": "docker",
        "agent": "terminus-2",
        "action_policy": "compute_tools",
        "os_command_execution_available": True,
        "supports_compilation_and_tests": True,
        "writes_complete_trajectory": True,
        "trajectory_format": "ATIF",
        "enforces_domain_sandbox_limits": True,
    }
    monkeypatch.setattr(
        "r3bench.agentic.external_backend.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(capabilities)
        ),
    )
    assert check_external_backend_readiness(config, probe=True)["status"] == "ready"

    capabilities["enforces_domain_sandbox_limits"] = False
    assert (
        check_external_backend_readiness(config, probe=True)["status"]
        == "protocol_incompatible"
    )


def test_external_backend_uses_and_rechecks_a_task_snapshot(
    tmp_path: Path, resources: Path, monkeypatch: pytest.MonkeyPatch
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
    monkeypatch.setattr(
        "r3bench.agentic.external_backend.check_external_backend_readiness",
        lambda *args, **kwargs: {"status": "ready"},
    )

    def mutate_snapshot(command: list[str], **kwargs: object) -> SimpleNamespace:
        runtime_task = Path(command[command.index("--task-dir") + 1])
        assert runtime_task.resolve() != task.task_dir.resolve()
        (runtime_task / "instruction.md").write_text("mutated\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "r3bench.agentic.external_backend.subprocess.run", mutate_snapshot
    )
    with pytest.raises(ExternalAgenticBackendError, match="recorded fingerprint"):
        run_external_agentic_backend(
            task_dir=task.task_dir,
            output_dir=tmp_path / "output",
            model_key="deepseek-chat",
            config=config,
            allow_real_api=True,
            allow_agentic_backend=True,
        )
