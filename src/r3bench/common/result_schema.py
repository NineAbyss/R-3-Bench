"""Public, provider-neutral result records for offline NL evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias

from r3bench.common.provider import UsageInfo
from r3bench.common.schema import DOMAINS, Domain
from r3bench.common.settings import (
    BudgetUnit,
    EvaluationSetting,
    expected_budget_unit,
)


Mode: TypeAlias = Literal["single_problem", "contest"]
Visibility: TypeAlias = Literal["hidden", "labeled"]
Stage: TypeAlias = Literal["one_stage", "stage1", "stage2"]
ParseStatus: TypeAlias = Literal["parsed", "missing", "parse_error"]
JudgeStatus: TypeAlias = Literal["judged", "not_judged", "judge_error"]
StageInputKind: TypeAlias = Literal["public_prompt", "stage1_output"]

_MODES = frozenset({"single_problem", "contest"})
_VISIBILITIES = frozenset({"hidden", "labeled"})
_STAGES = frozenset({"one_stage", "stage1", "stage2"})
_PARSE_STATUSES = frozenset({"parsed", "missing", "parse_error"})
_JUDGE_STATUSES = frozenset({"judged", "not_judged", "judge_error"})
_STAGE_INPUT_KINDS = frozenset({"public_prompt", "stage1_output"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:home|mnt)/|/tmp/rbench(?:/|\b)")
_CREDENTIAL = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{12,}\b|\bhf_[A-Za-z0-9]{12,}\b|"
    r"\bolp_[A-Za-z0-9]{12,}\b|Bearer\s+[A-Za-z0-9._~+/-]{12,})"
)


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_nonempty(value: object, field: str) -> None:
    if value is not None:
        _nonempty(value, field)


def _public_text(value: str, field: str) -> None:
    if _PRIVATE_PATH.search(value):
        raise ValueError(f"{field} contains a private machine path")
    if _CREDENTIAL.search(value):
        raise ValueError(f"{field} contains a credential-like value")


def _common(
    *,
    run_id: str,
    domain: Domain,
    mode: Mode,
    visibility: Visibility,
    stage: Stage,
    split: str,
    suite_id: str | None,
    problem_id: str | None,
    problem_label: str | None,
    created_at: str,
) -> None:
    _nonempty(run_id, "run_id")
    if domain not in DOMAINS:
        raise ValueError(f"unsupported domain: {domain!r}")
    if mode not in _MODES:
        raise ValueError(f"unsupported mode: {mode!r}")
    if visibility not in _VISIBILITIES:
        raise ValueError(f"unsupported visibility: {visibility!r}")
    if stage not in _STAGES:
        raise ValueError(f"unsupported stage: {stage!r}")
    _nonempty(split, "split")
    _optional_nonempty(suite_id, "suite_id")
    _optional_nonempty(problem_id, "problem_id")
    if problem_label is not None and problem_label not in tuple("ABCDEF"):
        raise ValueError("problem_label must be A through F or null")
    _nonempty(created_at, "created_at")


@dataclass(frozen=True, slots=True)
class RunMetadata:
    run_id: str
    domain: Domain
    mode: Mode
    visibility: Visibility
    stage: Stage
    split: str
    model_name: str
    provider_name: str
    prompt_template: str
    judge_profile: str
    created_at: str

    def __post_init__(self) -> None:
        _common(
            run_id=self.run_id,
            domain=self.domain,
            mode=self.mode,
            visibility=self.visibility,
            stage=self.stage,
            split=self.split,
            suite_id=None,
            problem_id=None,
            problem_label=None,
            created_at=self.created_at,
        )
        for field_name in (
            "model_name",
            "provider_name",
            "prompt_template",
            "judge_profile",
        ):
            _nonempty(getattr(self, field_name), field_name)


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    run_id: str
    request_id: str
    domain: Domain
    mode: Mode
    visibility: Visibility
    stage: Stage
    split: str
    suite_id: str | None
    problem_id: str | None
    problem_label: str | None
    model_name: str
    provider_name: str
    prompt_template: str
    prompt_sha256: str
    prompt_text: str
    parent_request_id: str | None
    stage1_request_id: str | None
    stage2_request_id: str | None
    stage_input_kind: StageInputKind
    stage_input_sha256: str
    response_text: str
    usage: UsageInfo
    finish_reason: str | None
    error_type: str | None
    error_message: str | None
    created_at: str

    def __post_init__(self) -> None:
        _common(
            run_id=self.run_id,
            domain=self.domain,
            mode=self.mode,
            visibility=self.visibility,
            stage=self.stage,
            split=self.split,
            suite_id=self.suite_id,
            problem_id=self.problem_id,
            problem_label=self.problem_label,
            created_at=self.created_at,
        )
        for field_name in (
            "request_id",
            "model_name",
            "provider_name",
            "prompt_template",
            "prompt_sha256",
            "prompt_text",
        ):
            _nonempty(getattr(self, field_name), field_name)
        if not _SHA256.fullmatch(self.prompt_sha256):
            raise ValueError("prompt_sha256 must contain 64 lowercase hexadecimal characters")
        for field_name in (
            "parent_request_id",
            "stage1_request_id",
            "stage2_request_id",
        ):
            _optional_nonempty(getattr(self, field_name), field_name)
        if self.stage_input_kind not in _STAGE_INPUT_KINDS:
            raise ValueError(f"unsupported stage_input_kind: {self.stage_input_kind!r}")
        if not _SHA256.fullmatch(self.stage_input_sha256):
            raise ValueError(
                "stage_input_sha256 must contain 64 lowercase hexadecimal characters"
            )
        if self.stage == "one_stage":
            if any(
                value is not None
                for value in (
                    self.parent_request_id,
                    self.stage1_request_id,
                    self.stage2_request_id,
                )
            ):
                raise ValueError("one-stage attempts cannot contain stage request links")
            if self.stage_input_kind != "public_prompt":
                raise ValueError("one-stage attempts must use public_prompt input")
        elif self.stage == "stage1":
            if self.parent_request_id is not None or self.stage2_request_id is not None:
                raise ValueError("Stage 1 attempts cannot link to a parent or Stage 2")
            if self.stage1_request_id != self.request_id:
                raise ValueError("Stage 1 attempt must identify itself as stage1_request_id")
            if self.stage_input_kind != "public_prompt":
                raise ValueError("Stage 1 attempts must use public_prompt input")
        else:
            if self.parent_request_id != self.stage1_request_id:
                raise ValueError("Stage 2 parent_request_id must equal stage1_request_id")
            if self.stage2_request_id != self.request_id:
                raise ValueError("Stage 2 attempt must identify itself as stage2_request_id")
            if self.stage_input_kind != "stage1_output":
                raise ValueError("Stage 2 attempts must use stage1_output input")
        if not isinstance(self.response_text, str):
            raise ValueError("response_text must be a string")
        for field_name in (
            "prompt_template",
            "prompt_text",
            "response_text",
            "error_message",
        ):
            value = getattr(self, field_name)
            if isinstance(value, str):
                _public_text(value, field_name)
        if not isinstance(self.usage, UsageInfo):
            raise ValueError("usage must be UsageInfo")
        _optional_nonempty(self.finish_reason, "finish_reason")
        _optional_nonempty(self.error_type, "error_type")
        _optional_nonempty(self.error_message, "error_message")
        if self.error_message is not None:
            _public_text(self.error_message, "error_message")


@dataclass(frozen=True, slots=True)
class ParsedAnswerRecord:
    run_id: str
    request_id: str
    domain: Domain
    mode: Mode
    visibility: Visibility
    stage: Stage
    split: str
    suite_id: str
    problem_id: str
    problem_label: str | None
    stage1_request_id: str | None
    stage2_request_id: str | None
    parsed_answer: str | None
    parse_status: ParseStatus
    error_type: str | None
    error_message: str | None
    created_at: str

    def __post_init__(self) -> None:
        _common(
            run_id=self.run_id,
            domain=self.domain,
            mode=self.mode,
            visibility=self.visibility,
            stage=self.stage,
            split=self.split,
            suite_id=self.suite_id,
            problem_id=self.problem_id,
            problem_label=self.problem_label,
            created_at=self.created_at,
        )
        _nonempty(self.request_id, "request_id")
        _optional_nonempty(self.stage1_request_id, "stage1_request_id")
        _optional_nonempty(self.stage2_request_id, "stage2_request_id")
        if self.stage == "stage2":
            if self.stage1_request_id is None or self.stage2_request_id != self.request_id:
                raise ValueError("Stage 2 parsed records require linked Stage 1 and Stage 2 IDs")
        elif self.stage1_request_id is not None or self.stage2_request_id is not None:
            raise ValueError("one-stage parsed records cannot contain stage request links")
        if self.parse_status not in _PARSE_STATUSES:
            raise ValueError(f"unsupported parse_status: {self.parse_status!r}")
        if self.parsed_answer is not None and not isinstance(self.parsed_answer, str):
            raise ValueError("parsed_answer must be a string or null")
        if self.parse_status == "parsed" and not (self.parsed_answer or "").strip():
            raise ValueError("parsed records must contain a non-empty answer")
        if self.parsed_answer is not None:
            _public_text(self.parsed_answer, "parsed_answer")
        _optional_nonempty(self.error_type, "error_type")
        _optional_nonempty(self.error_message, "error_message")
        if self.error_message is not None:
            _public_text(self.error_message, "error_message")


@dataclass(frozen=True, slots=True)
class JudgeResultRecord:
    run_id: str
    request_id: str
    domain: Domain
    mode: Mode
    visibility: Visibility
    stage: Stage
    split: str
    suite_id: str
    problem_id: str
    problem_label: str | None
    stage1_request_id: str | None
    stage2_request_id: str | None
    judge_status: JudgeStatus
    verdict: str
    score: float
    error_type: str | None
    error_message: str | None
    created_at: str

    def __post_init__(self) -> None:
        _common(
            run_id=self.run_id,
            domain=self.domain,
            mode=self.mode,
            visibility=self.visibility,
            stage=self.stage,
            split=self.split,
            suite_id=self.suite_id,
            problem_id=self.problem_id,
            problem_label=self.problem_label,
            created_at=self.created_at,
        )
        _nonempty(self.request_id, "request_id")
        _optional_nonempty(self.stage1_request_id, "stage1_request_id")
        _optional_nonempty(self.stage2_request_id, "stage2_request_id")
        if self.stage == "stage2":
            if self.stage1_request_id is None or self.stage2_request_id != self.request_id:
                raise ValueError("Stage 2 judge records require linked Stage 1 and Stage 2 IDs")
        elif self.stage1_request_id is not None or self.stage2_request_id is not None:
            raise ValueError("one-stage judge records cannot contain stage request links")
        if self.judge_status not in _JUDGE_STATUSES:
            raise ValueError(f"unsupported judge_status: {self.judge_status!r}")
        _nonempty(self.verdict, "verdict")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("score must be numeric")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("score must be between 0 and 1")
        _optional_nonempty(self.error_type, "error_type")
        _optional_nonempty(self.error_message, "error_message")
        if self.error_message is not None:
            _public_text(self.error_message, "error_message")


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    domain: Domain
    mode: Mode
    visibility: Visibility
    stage: Stage
    split: str
    model_name: str
    provider_name: str
    attempt_count: int
    problem_count: int
    parsed_count: int
    judged_count: int
    correct_count: int
    total_score: float
    error_count: int
    created_at: str

    def __post_init__(self) -> None:
        _common(
            run_id=self.run_id,
            domain=self.domain,
            mode=self.mode,
            visibility=self.visibility,
            stage=self.stage,
            split=self.split,
            suite_id=None,
            problem_id=None,
            problem_label=None,
            created_at=self.created_at,
        )
        _nonempty(self.model_name, "model_name")
        _nonempty(self.provider_name, "provider_name")
        for field_name in (
            "attempt_count",
            "problem_count",
            "parsed_count",
            "judged_count",
            "correct_count",
            "error_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if not 0.0 <= float(self.total_score) <= float(self.problem_count):
            raise ValueError("total_score must be between zero and problem_count")


@dataclass(frozen=True, slots=True)
class UnifiedEvaluationSummary:
    """Common envelope around setting-specific generation and scoring outputs."""

    schema_version: str
    cell_id: str
    setting: EvaluationSetting
    domain: Domain
    mode: Literal["single_problem", "contest", "response_curve"]
    visibility: Visibility
    model_key: str
    budget_unit: BudgetUnit
    budget_value: int | None
    generation_status: str
    scoring_status: str
    output_dir: str
    created_at: str

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported unified result schema_version")
        _nonempty(self.cell_id, "cell_id")
        if self.domain not in DOMAINS:
            raise ValueError(f"unsupported domain: {self.domain!r}")
        if self.mode not in {"single_problem", "contest", "response_curve"}:
            raise ValueError(f"unsupported unified mode: {self.mode!r}")
        if self.visibility not in _VISIBILITIES:
            raise ValueError(f"unsupported visibility: {self.visibility!r}")
        _nonempty(self.model_key, "model_key")
        if self.budget_unit != expected_budget_unit(self.setting):
            raise ValueError(
                f"{self.setting} requires budget_unit="
                f"{expected_budget_unit(self.setting)!r}"
            )
        if self.budget_value is not None and (
            isinstance(self.budget_value, bool)
            or not isinstance(self.budget_value, int)
            or self.budget_value < 0
        ):
            raise ValueError("budget_value must be a non-negative integer or null")
        _nonempty(self.generation_status, "generation_status")
        _nonempty(self.scoring_status, "scoring_status")
        _nonempty(self.output_dir, "output_dir")
        _public_text(self.output_dir, "output_dir")
        _nonempty(self.created_at, "created_at")


def to_public_dict(value: Any) -> Any:
    """Convert immutable result objects to strict JSON-compatible values."""

    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_public_dict(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_public_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_public_dict(item) for item in value]
    if isinstance(value, MappingProxyType):
        return {str(key): to_public_dict(item) for key, item in value.items()}
    return value


__all__ = [
    "AttemptRecord",
    "JudgeResultRecord",
    "ParsedAnswerRecord",
    "RunMetadata",
    "RunSummary",
    "UnifiedEvaluationSummary",
    "to_public_dict",
]
