from __future__ import annotations

import json
from pathlib import Path

import pytest

from r3bench.oracle.protocol_v3 import (
    load_response_curve_points_compatible,
    run_condition_analysis,
)
from r3bench.oracle.response_curve_schema import OracleSchemaError


MODEL = "deepseek-chat"
DOMAIN = "math"
SETTING = "tool_free"
BUDGET_UNIT = "output_tokens"
SUITE_ID = "condition-binding-suite"
CURVE_PROFILE = "tool_free_math_deepseek_chat_single_problem_response_curve"
CONTEST_PROFILE = "tool_free_math_deepseek_chat_budgeted_rho_0p2"
CURVE_GRID = (0, 966, 1931, 3862, 7724, 15448)
CONTEST_BUDGET = 3862


def _curve_rows(
    *,
    condition_id: str = CURVE_PROFILE,
    condition_kind: str = "official_profile",
    grid: tuple[int, ...] = CURVE_GRID,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for problem_index, label in enumerate("ABCDEF", start=1):
        problem_id = f"condition-binding-{label}"
        for budget_level, budget in enumerate(grid, start=1):
            for repeat_id in range(1, 6):
                rows.append(
                    {
                        "schema_version": "3.0",
                        "condition_id": condition_id,
                        "condition_kind": condition_kind,
                        "domain": DOMAIN,
                        "model_key": MODEL,
                        "setting": SETTING,
                        "budget_unit": BUDGET_UNIT,
                        "mode": "single_problem",
                        "problem_id": problem_id,
                        "suite_id": SUITE_ID,
                        "problem_index": problem_index,
                        "problem_label": label,
                        "budget": budget,
                        "observed_cost": 0,
                        "reward": 0,
                        "parse_status": "parsed",
                        "judge_status": "judged",
                        "source_run_id": (
                            f"curve-{label}-level-{budget_level}-repeat-{repeat_id}"
                        ),
                        "repeat_id": repeat_id,
                        "budget_level": budget_level,
                    }
                )
    return rows


def _contest_rows(
    *,
    condition_id: str = CONTEST_PROFILE,
    condition_kind: str = "official_profile",
    budget_profile: str | None = CONTEST_PROFILE,
    rho: float | None = 0.2,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for repeat_id in range(1, 6):
        for problem_index, label in enumerate("ABCDEF", start=1):
            rows.append(
                {
                    "schema_version": "3.0",
                    "domain": DOMAIN,
                    "model_key": MODEL,
                    "setting": SETTING,
                    "budget_unit": BUDGET_UNIT,
                    "mode": "contest",
                    "condition_id": condition_id,
                    "condition_kind": condition_kind,
                    "contest_budget": CONTEST_BUDGET,
                    "problem_id": f"condition-binding-{label}",
                    "suite_id": SUITE_ID,
                    "problem_index": problem_index,
                    "problem_label": label,
                    "reward": 0,
                    "parse_status": "parsed",
                    "judge_status": "judged",
                    "source_run_id": f"contest-repeat-{repeat_id}",
                    "budget_profile": budget_profile,
                    "rho": rho,
                    "repeat_id": repeat_id,
                }
            )
    return rows


def _budget_document(
    *,
    condition_id: str = CONTEST_PROFILE,
    condition_kind: str = "official_profile",
    budget_profile: str | None = CONTEST_PROFILE,
    rho: float | None = 0.2,
) -> dict[str, object]:
    return {
        "schema_version": "3.0",
        "budgets": [
            {
                "domain": DOMAIN,
                "model_key": MODEL,
                "setting": SETTING,
                "budget_unit": BUDGET_UNIT,
                "condition_id": condition_id,
                "condition_kind": condition_kind,
                "contest_budget": CONTEST_BUDGET,
                "response_curve_grid": list(CURVE_GRID),
                "budget_profile": budget_profile,
                "rho": rho,
            }
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_inputs(
    tmp_path: Path,
    *,
    curves: list[dict[str, object]],
    contests: list[dict[str, object]],
    budgets: dict[str, object],
) -> tuple[Path, Path, Path]:
    curve_path = tmp_path / "curve.jsonl"
    contest_path = tmp_path / "contest.jsonl"
    budget_path = tmp_path / "budgets.json"
    _write_jsonl(curve_path, curves)
    _write_jsonl(contest_path, contests)
    budget_path.write_text(json.dumps(budgets), encoding="utf-8")
    return curve_path, contest_path, budget_path


def _run(
    tmp_path: Path,
    *,
    curves: list[dict[str, object]],
    contests: list[dict[str, object]] | None = None,
    budgets: dict[str, object] | None = None,
) -> dict[str, object]:
    curve_path, contest_path, budget_path = _write_inputs(
        tmp_path,
        curves=curves,
        contests=contests or _contest_rows(),
        budgets=budgets or _budget_document(),
    )
    return run_condition_analysis(
        response_curve_path=curve_path,
        contest_results_path=contest_path,
        budgets_path=budget_path,
        output_dir=tmp_path / "output",
    )


def test_v3_curve_loader_preserves_condition_metadata(tmp_path: Path) -> None:
    path = tmp_path / "one-point.jsonl"
    _write_jsonl(path, [_curve_rows()[0]])

    point = load_response_curve_points_compatible(path)[0]

    assert point.condition_id == CURVE_PROFILE
    assert point.condition_kind == "official_profile"


def test_matching_official_curve_profile_runs_formal_analysis(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path, curves=_curve_rows())

    assert result["status"] == "complete"
    assert result["oracle_protocol"] == "six_level_five_repeat_mckp"
    assert result["response_curve_point_count"] == 180


def test_custom_curve_cannot_masquerade_as_official_analysis(
    tmp_path: Path,
) -> None:
    with pytest.raises(OracleSchemaError, match="every response-curve row"):
        _run(
            tmp_path,
            curves=_curve_rows(
                condition_id="custom_curve",
                condition_kind="custom",
            ),
        )


def test_official_curve_rejects_wrong_profile_condition(tmp_path: Path) -> None:
    with pytest.raises(OracleSchemaError, match="single-problem budget profile"):
        _run(tmp_path, curves=_curve_rows(condition_id=CONTEST_PROFILE))


def test_official_curve_rejects_observed_grid_drift(tmp_path: Path) -> None:
    drifted = CURVE_GRID[:-1] + (CURVE_GRID[-1] + 1,)
    with pytest.raises(OracleSchemaError, match="profile grid"):
        _run(tmp_path, curves=_curve_rows(grid=drifted))


def test_official_curve_cannot_mix_with_custom_contest_and_budget(
    tmp_path: Path,
) -> None:
    with pytest.raises(OracleSchemaError, match="custom contest"):
        _run(
            tmp_path,
            curves=_curve_rows(),
            contests=_contest_rows(
                condition_id="custom_contest",
                condition_kind="custom",
                budget_profile=None,
                rho=None,
            ),
            budgets=_budget_document(
                condition_id="custom_contest",
                condition_kind="custom",
                budget_profile=None,
                rho=None,
            ),
        )


def test_custom_five_repeat_analysis_remains_supported(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        curves=_curve_rows(
            condition_id="custom_curve",
            condition_kind="custom",
        ),
        contests=_contest_rows(
            condition_id="custom_contest",
            condition_kind="custom",
            budget_profile=None,
            rho=None,
        ),
        budgets=_budget_document(
            condition_id="custom_contest",
            condition_kind="custom",
            budget_profile=None,
            rho=None,
        ),
    )

    assert result["status"] == "complete"
    assert result["oracle_protocol"] == "six_level_five_repeat_mckp"
