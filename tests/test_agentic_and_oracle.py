from __future__ import annotations

import json
from pathlib import Path

from r3bench.agentic.action_accounting import (
    ActionClass,
    apply_budget_decision,
    policy_from_name,
)
from r3bench.agentic.budget import ActionBudget
from r3bench.agentic.scope import AgenticScopeState
from r3bench.agentic.task_export import (
    export_agentic_response_curve_tasks,
    export_agentic_tasks,
)
from r3bench.commands.analysis import main as analysis_main
from r3bench.commands.agentic_dryrun import main as dryrun_main
from r3bench.oracle.knapsack import (
    KnapsackItem,
    solve_knapsack,
    solve_multiple_choice_knapsack,
)
from r3bench.oracle.protocol_v3 import run_condition_analysis


def test_agentic_policies_separate_free_and_counted_actions() -> None:
    policy = policy_from_name("compute_tools")
    assert policy.classify_action("focus_problem A") == ActionClass.FREE_BOOKKEEPING
    assert policy.classify_action("python solve.py") == ActionClass.COUNTED
    assert (
        policy.classify_action("printf draft > notes.txt")
        == ActionClass.FREE_FILE_WRITE
    )
    assert (
        policy.classify_action("write_final_artifact /app/solution_A.cpp")
        == ActionClass.FREE_FINALIZATION
    )
    assert (
        policy.classify_action("submit_solution solution_A.cpp") == ActionClass.BLOCKED
    )


def test_compute_tools_counts_build_test_debug_and_unknown_commands() -> None:
    policy = policy_from_name("compute_tools")
    counted_commands = (
        "make",
        "cmake --build build",
        "ninja -C build",
        "cargo test",
        "pytest -q",
        "gdb ./solution_A",
        "./solution_A",
        "/logs/problem_A/solution",
        "custom-local-solver --run instance.txt",
    )
    for command in counted_commands:
        assert policy.classify_action(command) == ActionClass.COUNTED, command

    budget = ActionBudget(1)
    accepted = apply_budget_decision("custom-local-solver input.txt", budget, policy)
    assert accepted.counted is True
    assert accepted.budget_consumed == 1
    assert budget.remaining == 0


def test_compute_tools_keeps_appendix_g_passive_file_operations_free() -> None:
    policy = policy_from_name("compute_tools")
    free_reads = (
        "cat task.txt",
        "cat /logs/problem_A/source.cpp",
        "head -n 20 task.txt",
        "tail -n 20 /logs/problem_A/source.cpp",
    )
    free_mutations = (
        "cp source.cpp backup.cpp",
        "mv draft.cpp staged.cpp",
        "touch notes.txt",
        "mkdir scratch",
        "rm scratch.tmp",
        "rmdir scratch",
        "chmod 644 source.cpp",
        "truncate -s 0 notes.txt",
    )
    for command in free_reads:
        assert policy.classify_action(command) == ActionClass.FREE_ENVIRONMENT
    for command in free_mutations:
        assert policy.classify_action(command) == ActionClass.FREE_FILE_WRITE

    program_write = """cat <<'PY' > solve.py
print(sum(range(10)))
PY"""
    assert policy.classify_action(program_write) == ActionClass.FREE_FILE_WRITE
    assert policy.classify_action("python solve.py") == ActionClass.COUNTED

    combined_write_and_run = """cat <<'PY' > solve.py
print(sum(range(10)))
PY
python solve.py"""
    assert policy.classify_action(combined_write_and_run) == ActionClass.COUNTED
    assert policy.classify_action("cd /app") == ActionClass.FREE_ENVIRONMENT
    assert (
        policy.classify_action("export TMPDIR=/tmp/r3bench")
        == ActionClass.FREE_ENVIRONMENT
    )
    assert (
        policy.classify_action("mkdir /tmp/stage && cp /app/a /tmp/stage/a")
        == ActionClass.FREE_FILE_WRITE
    )
    assert (
        policy.classify_action("mkdir /tmp/stage && custom-solver input.txt")
        == ActionClass.COUNTED
    )


def test_compute_tools_keeps_explicit_writes_free_and_unknown_actions_scoped() -> None:
    policy = policy_from_name("compute_tools")
    assert (
        policy.classify_action("printf draft > notes.txt")
        == ActionClass.FREE_FILE_WRITE
    )
    assert (
        policy.classify_action("printf solution > /app/solution_A.cpp")
        == ActionClass.FREE_FINALIZATION
    )

    scope = AgenticScopeState(
        valid_problem_ids=frozenset({"problem-a", "problem-b"}),
        problem_labels={"A": "problem-a", "B": "problem-b"},
    )
    command = "custom-local-solver input.txt"
    without_focus = scope.authorize_action(policy.classify_action(command), command)
    assert without_focus.allowed is False
    assert without_focus.reason == "counted_action_requires_active_focus"

    scope.focus_problem("A")
    focused = scope.authorize_action(policy.classify_action(command), command)
    assert focused.allowed is True
    assert focused.attributed_problem_id == "problem-a"

    cross_problem = "g++ /logs/problem_B/solution_B.cpp"
    blocked = scope.authorize_action(
        policy.classify_action(cross_problem), cross_problem
    )
    assert blocked.allowed is False
    assert blocked.reason == "cross_problem_access_blocked:B"

    cross_problem_scratch = "cat /logs/problem_B/input.txt"
    blocked_scratch = scope.authorize_action(
        policy.classify_action(cross_problem_scratch), cross_problem_scratch
    )
    assert blocked_scratch.allowed is True

    counted_scratch = "python analyze.py /logs/problem_B/input.txt"
    blocked_counted_scratch = scope.authorize_action(
        policy.classify_action(counted_scratch), counted_scratch
    )
    assert blocked_counted_scratch.allowed is False
    assert blocked_counted_scratch.reason == "cross_problem_access_blocked:B"

    for dynamic_path in (
        "/logs/problem_?/input.txt",
        "/logs/problem_[AB]/input.txt",
        "/logs/problem_${LABEL}/input.txt",
        "/app/solution_?.cpp",
    ):
        command = f"python analyze.py {dynamic_path}"
        dynamic = scope.authorize_action(policy.classify_action(command), command)
        assert dynamic.allowed is False
        assert dynamic.reason == "cross_problem_access_blocked:dynamic"

    shared_answer = "python verify.py /logs/artifacts/answer.txt"
    shared = scope.authorize_action(
        policy.classify_action(shared_answer), shared_answer
    )
    assert shared.allowed is False
    assert shared.reason == "cross_problem_access_blocked:shared"


def test_agentic_export_and_dryrun_execute_no_commands(
    tmp_path: Path, resources: Path
) -> None:
    exported = export_agentic_tasks(
        domain="coding",
        data_source=resources / "examples/data/coding.jsonl",
        output_dir=tmp_path / "tasks",
        budget=3,
        limit_suites=1,
        strict_data=False,
    )
    assert len(exported) == 1
    output = tmp_path / "dryrun"
    assert (
        dryrun_main(
            [
                "--task-dir",
                str(tmp_path / "tasks"),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads((output / "dryrun_summary.json").read_text(encoding="utf-8"))
    assert summary["commands_executed_by_os"] is False
    assert summary["network_called"] is False
    assert summary["correctness_feedback_exposed"] is False


def test_agentic_curve_preserves_duplicate_caps_by_level_and_repeat(
    tmp_path: Path, resources: Path
) -> None:
    tasks = export_agentic_response_curve_tasks(
        domain="coding",
        data_source=resources / "examples/data/coding.jsonl",
        output_dir=tmp_path / "curve-tasks",
        budgets=(0, 1, 1, 2, 4, 8),
        repeat_ids=range(1, 6),
        limit_problems=1,
        confirm_full_curve=True,
        strict_data=False,
    )
    assert len(tasks) == 30
    assert len({task.task_dir for task in tasks}) == 30
    assert [
        (task.budget_level, task.counted_action_budget, task.repeat_id)
        for task in tasks[:15:5]
    ] == [(1, 0, 1), (2, 1, 1), (3, 1, 1)]


def test_knapsack_is_deterministic() -> None:
    result = solve_knapsack(
        [
            KnapsackItem("b", cost=2, value=1),
            KnapsackItem("a", cost=2, value=1),
            KnapsackItem("c", cost=3, value=2),
        ],
        budget=4,
    )
    assert result.selected_keys == ("c",)
    assert result.total_value == 2
    assert result.total_cost == 3


def test_multiple_choice_knapsack_uses_stable_float_ties() -> None:
    result = solve_multiple_choice_knapsack(
        (
            (
                KnapsackItem("a-exact", cost=1, value=0.3),
                KnapsackItem("z-split-1", cost=0, value=0.1),
            ),
            (
                KnapsackItem("a-zero", cost=0, value=0.0),
                KnapsackItem("z-split-2", cost=1, value=0.2),
            ),
        ),
        budget=1,
    )
    assert result.selected_keys == ("a-exact", "a-zero")
    assert result.combination_count == 4


def test_analysis_compare_pipeline(tmp_path: Path, resources: Path) -> None:
    output = tmp_path / "oracle"
    assert (
        analysis_main(
            [
                "compare",
                "--response-curve",
                str(resources / "examples/inputs/analysis/response_curve_points.jsonl"),
                "--contest-results",
                str(resources / "examples/inputs/analysis/contest_results.jsonl"),
                "--budgets",
                str(resources / "examples/inputs/analysis/budgets.json"),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    gap = json.loads((output / "gap_summary.json").read_text(encoding="utf-8"))
    assert len(gap["summaries"]) == 1
    assert (output / "oracle_items.jsonl").is_file()


def test_zero_budget_analysis_requires_executed_and_scored_run(
    tmp_path: Path, resources: Path
) -> None:
    assert (
        analysis_main(
            [
                "build-response-curve",
                "--setting",
                "tool_free",
                "--domain",
                "math",
                "--model",
                "local-mock",
                "--data",
                str(resources / "examples/data/math/problems.jsonl"),
                "--budget",
                "0",
                "--output",
                str(tmp_path / "zero.jsonl"),
                "--relaxed",
            ]
        )
        == 2
    )
    assert not (tmp_path / "zero.jsonl").exists()


def test_official_analysis_rejects_missing_repeat_and_level_metadata(
    tmp_path: Path, resources: Path
) -> None:
    run_dir = tmp_path / "legacy-run"
    scoring_dir = tmp_path / "legacy-scoring"
    run_dir.mkdir()
    scoring_dir.mkdir()
    (run_dir / "run_summary.json").write_text("{}\n", encoding="utf-8")
    common = [
        "--setting",
        "tool_free",
        "--domain",
        "math",
        "--model",
        "formal-model",
        "--run-dir",
        str(run_dir),
        "--scoring-dir",
        str(scoring_dir),
        "--condition-kind",
        "official_profile",
        "--budget-profile",
        "paper-profile",
    ]
    assert (
        analysis_main(
            [
                "build-response-curve",
                *common,
                "--data",
                str(resources / "examples/data/math/problems.jsonl"),
                "--budget",
                "1",
                "--output",
                str(tmp_path / "curve.jsonl"),
                "--relaxed",
            ]
        )
        == 2
    )
    assert (
        analysis_main(
            [
                "build-contest-results",
                *common,
                "--contest-budget",
                "12",
                "--condition-id",
                "paper-contest",
                "--response-curve-grid",
                "0,1,1,2,4,8",
                "--output",
                str(tmp_path / "contest.jsonl"),
            ]
        )
        == 2
    )


def test_formal_oracle_uses_six_levels_five_repeats_and_configured_cost(
    tmp_path: Path,
) -> None:
    budgets = (0, 1, 1, 2, 4, 8)
    successes = (0, 5, 1, 0, 0, 0)
    curve_rows = []
    contest_rows = []
    for problem_index in range(1, 7):
        problem_id = f"formal-{problem_index}"
        label = "ABCDEF"[problem_index - 1]
        for budget_level, (budget, success_count) in enumerate(
            zip(budgets, successes, strict=True), start=1
        ):
            for repeat_id in range(1, 6):
                curve_rows.append(
                    {
                        "schema_version": "3.0",
                        "condition_id": "formal_curve",
                        "condition_kind": "custom",
                        "domain": "math",
                        "model_key": "formal-model",
                        "setting": "tool_free",
                        "budget_unit": "output_tokens",
                        "mode": "single_problem",
                        "problem_id": problem_id,
                        "suite_id": "formal-suite",
                        "problem_index": problem_index,
                        "problem_label": label,
                        "budget": budget,
                        "observed_cost": 0,
                        "reward": int(repeat_id <= success_count),
                        "parse_status": "parsed",
                        "judge_status": "judged",
                        "source_run_id": (
                            f"curve-{problem_id}-level-{budget_level}-"
                            f"repeat-{repeat_id}"
                        ),
                        "repeat_id": repeat_id,
                        "budget_level": budget_level,
                    }
                )
        for repeat_id in range(1, 6):
            contest_rows.append(
                {
                    "schema_version": "3.0",
                    "domain": "math",
                    "model_key": "formal-model",
                    "setting": "tool_free",
                    "budget_unit": "output_tokens",
                    "mode": "contest",
                    "condition_id": "formal_contest",
                    "condition_kind": "custom",
                    "contest_budget": 12,
                    "problem_id": problem_id,
                    "suite_id": "formal-suite",
                    "problem_index": problem_index,
                    "problem_label": label,
                    "reward": 0,
                    "parse_status": "parsed",
                    "judge_status": "judged",
                    "source_run_id": f"contest-repeat-{repeat_id}",
                    "budget_profile": None,
                    "rho": None,
                    "repeat_id": repeat_id,
                }
            )

    curve_path = tmp_path / "curve.jsonl"
    curve_path.write_text(
        "".join(json.dumps(row) + "\n" for row in curve_rows), encoding="utf-8"
    )
    contest_path = tmp_path / "contest.jsonl"
    contest_path.write_text(
        "".join(json.dumps(row) + "\n" for row in contest_rows), encoding="utf-8"
    )
    budget_path = tmp_path / "budgets.json"
    budget_path.write_text(
        json.dumps(
            {
                "schema_version": "3.0",
                "budgets": [
                    {
                        "domain": "math",
                        "model_key": "formal-model",
                        "setting": "tool_free",
                        "budget_unit": "output_tokens",
                        "condition_id": "formal_contest",
                        "condition_kind": "custom",
                        "contest_budget": 12,
                        "response_curve_grid": list(budgets),
                        "budget_profile": None,
                        "rho": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "formal-output"
    result = run_condition_analysis(
        response_curve_path=curve_path,
        contest_results_path=contest_path,
        budgets_path=budget_path,
        output_dir=output,
    )
    assert result["oracle_protocol"] == "six_level_five_repeat_mckp"
    assert result["response_curve_point_count"] == 180
    assert result["oracle_budget_option_count"] == 36

    equal = json.loads((output / "equal_replay.json").read_text(encoding="utf-8"))
    assert equal["results"][0]["equal_score"] == 0.0
    assert {row["budget_level"] for row in equal["results"][0]["problem_results"]} == {
        4
    }
    oracle = json.loads((output / "oracle_results.json").read_text(encoding="utf-8"))
    assert oracle["results"][0]["oracle_score"] == 6.0
    assert oracle["results"][0]["total_selected_cost"] == 6
    assert oracle["results"][0]["combination_count"] == 6**6
    assert {
        row["budget_level"] for row in oracle["results"][0]["problem_selections"]
    } == {2}
    gap = json.loads((output / "gap_summary.json").read_text(encoding="utf-8"))
    summary = gap["summaries"][0]
    assert summary["suite_count"] == 1
    assert summary["repeat_count"] == 5
    assert summary["contest_run_count"] == 5
    assert summary["contest_score"] == 0.0
    assert summary["oracle_score"] == 6.0
