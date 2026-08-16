from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from r3bench.agentic.scoring_handoff import collect_agentic_saved_outputs


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("domain", "container_path"),
    [
        ("math", "/logs/artifacts/answer.txt"),
        ("abstract_reasoning", "/logs/artifacts/answer.txt"),
        ("coding", "/app/solution_A.cpp"),
    ],
)
def test_invalid_utf8_final_output_becomes_a_parse_failure(
    tmp_path: Path, domain: str, container_path: str
) -> None:
    artifact = tmp_path / "artifacts" / (
        "solution_A.cpp" if domain == "coding" else "answer.txt"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"\xff\xfe\x00")
    _write_json(
        tmp_path / "backend_summary.json",
        {
            "domain": domain,
            "execution_id": "execution-one",
            "task_id": "task-one",
            "model_key": "deepseek-chat",
        },
    )
    _write_json(
        tmp_path / "task_binding" / "public_problem_manifest.json",
        {
            "problems": [
                {"problem_id": "problem-one", "problem_label": "A"}
            ]
        },
    )
    _write_json(
        tmp_path / "final_artifacts_manifest.json",
        {
            "artifacts": [
                {
                    "container_path": container_path,
                    "problem_label": "A" if domain == "coding" else None,
                    "exists": True,
                    "artifact_relative_path": f"artifacts/{artifact.name}",
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                }
            ]
        },
    )
    rows = collect_agentic_saved_outputs(tmp_path)
    assert len(rows) == 1
    assert rows[0]["parsed_answer"] is None
    assert rows[0]["execution_id"] == "execution-one"
