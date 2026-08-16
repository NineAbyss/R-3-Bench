from __future__ import annotations

import pytest

from r3bench.oracle.from_agentic_outputs import (
    _reward as agentic_reward,
)
from r3bench.oracle.from_agentic_outputs import (
    _statuses as agentic_statuses,
)
from r3bench.oracle.from_nl_outputs import _reward as nl_reward
from r3bench.oracle.from_nl_outputs import _statuses as nl_statuses
from r3bench.oracle.response_curve_schema import OracleSchemaError


@pytest.mark.parametrize("domain", ["coding", "math", "abstract_reasoning"])
@pytest.mark.parametrize("parse_status", ["missing", "parse_error"])
def test_production_parse_failures_are_valid_zero_rewards(
    domain: str, parse_status: str
) -> None:
    row = {
        "domain": domain,
        "parse_status": parse_status,
        "judge_status": "not_judged",
        "correct": False,
        "score": 0.0,
    }
    assert nl_statuses(row, allow_unjudged=False) == (
        parse_status,
        "not_judged",
    )
    assert agentic_statuses(row, allow_unjudged=False) == (
        parse_status,
        "not_judged",
    )
    assert nl_reward(row, allow_unjudged=False) == 0
    assert agentic_reward(row, allow_unjudged=False) == 0


def test_unresolved_parsed_outputs_remain_rejected() -> None:
    row = {
        "parse_status": "parsed",
        "judge_status": "not_judged",
        "correct": False,
        "score": 0.0,
    }
    for validator in (nl_statuses, agentic_statuses):
        with pytest.raises(OracleSchemaError):
            validator(row, allow_unjudged=False)
    for reward in (nl_reward, agentic_reward):
        with pytest.raises(OracleSchemaError):
            reward(row, allow_unjudged=False)


def test_downstream_judge_failures_are_retained_as_zero() -> None:
    row = {
        "parse_status": "parsed",
        "judge_status": "judge_error",
        "correct": False,
        "score": 0.0,
    }
    assert nl_statuses(row, allow_unjudged=False) == ("parsed", "judge_error")
    assert agentic_statuses(row, allow_unjudged=False) == (
        "parsed",
        "judge_error",
    )
    assert nl_reward(row, allow_unjudged=False) == 0
    assert agentic_reward(row, allow_unjudged=False) == 0
