"""Offline pure-NL orchestration over public R3Bench interfaces."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal

from r3bench.abstract_reasoning.parser import extract_ar_answer
from r3bench.abstract_reasoning.scorer import MockARScorer
from r3bench.coding.parser import extract_contest_cpp_sections, extract_cpp_code
from r3bench.coding.verifier import MockCodingVerifier
from r3bench.common.experiment import ExperimentConfig, PromptConfig
from r3bench.common.loader import load_contest_suites, load_single_problems
from r3bench.common.presentation import (
    PresentationOrderRecord,
    present_contest_suite,
)
from r3bench.common.prompt import (
    PromptTemplate,
    render_contest_prompt,
    render_single_prompt,
    render_two_stage_prompt,
)
from r3bench.common.provider import (
    Message,
    MockProvider,
    ModelRequest,
    ModelResponse,
    Provider,
    ReplayProvider,
    TransientStageHandoff,
    UsageInfo,
)
from r3bench.common.result_schema import (
    AttemptRecord,
    JudgeResultRecord,
    ParsedAnswerRecord,
    RunMetadata,
    RunSummary,
    to_public_dict,
)
from r3bench.common.schema import ContestSuite, ProblemRecord
from r3bench.math.judge import MockMathJudge
from r3bench.math.parser import extract_math_answer
from r3bench.providers.base import ProviderAdapter
from r3bench.providers.errors import (
    ProviderAuthError,
    ProviderConfigError,
    ProviderError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderRetryError,
)
from r3bench.common.two_stage_profile import (
    TwoStageProtocol,
    default_two_stage_protocol,
)
from r3bench.resource_paths import resolve_path


_NL_SECTION = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?problem\s+([1-6A-F])\s*:?\s*$"
)
_STAGE1_TRACE_MARKER = "[[R3BENCH_STAGE1_TRACE]]"
_STAGE1_REASONING_MARKER = "[[R3BENCH_STAGE1_REASONING]]"
_STAGE1_VISIBLE_MARKER = "[[R3BENCH_STAGE1_VISIBLE_OUTPUT]]"


class OfflineRunnerError(ValueError):
    """Raised when a public offline run violates the protocol."""


@dataclass(frozen=True, slots=True)
class NLRunArtifacts:
    metadata: RunMetadata
    attempts: tuple[AttemptRecord, ...]
    parsed_answers: tuple[ParsedAnswerRecord, ...]
    judge_results: tuple[JudgeResultRecord, ...]
    summary: RunSummary
    presentation_orders: tuple[PresentationOrderRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedNLRequest:
    """One rendered public item and its provider-neutral model request."""

    item: ProblemRecord | ContestSuite
    prompt_text: str
    request: ModelRequest
    presentation: PresentationOrderRecord | None = None


@dataclass(frozen=True, slots=True)
class PreparedTwoStageRequest:
    item: ProblemRecord | ContestSuite
    stage1_prompt: str
    stage1_request: ModelRequest
    stage2_prompt: str
    stage2_request: ModelRequest
    stage1_output_sha256: str
    presentation: PresentationOrderRecord | None = None


@dataclass(frozen=True, slots=True)
class CompletionOutcome:
    """A provider completion or a sanitized evaluator-local failure."""

    response: ModelResponse | None
    error_type: str | None = None
    error_message: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_public_path(path: str) -> Path:
    candidate = resolve_path(path)
    if candidate.exists():
        return candidate
    raise OfflineRunnerError(f"public input does not exist: {path}")


def _data_source(config: ExperimentConfig) -> str:
    """Return an explicit local/HF source without treating it as a prompt path."""

    if config.data_source.startswith("hf://"):
        return config.data_source
    return str(resolve_path(config.data_source))


def _validate_config(config: ExperimentConfig, expected_mode: str) -> None:
    if config.mode != expected_mode:
        raise OfflineRunnerError(
            f"runner requires mode={expected_mode!r}, found {config.mode!r}"
        )
    if config.setting != "tool_free":
        raise OfflineRunnerError("the Tool-Free runner requires setting='tool_free'")
    if not config.judge_profile_name.startswith("mock_"):
        raise OfflineRunnerError(
            "the Tool-Free runner supports only explicit mock judge/verifier profiles"
        )


def _validate_provider(provider: object) -> Provider:
    if not isinstance(provider, (MockProvider, ReplayProvider, ProviderAdapter)):
        raise OfflineRunnerError(
            "provider must be MockProvider or ReplayProvider or a validated "
            "ProviderAdapter"
        )
    return provider


def _run_id(config: ExperimentConfig) -> str:
    stable = {
        "name": config.name,
        "domain": config.domain,
        "mode": config.mode,
        "visibility": config.visibility,
        "stage": config.stage,
        "split": config.split,
        "model": config.model_name,
        "provider": config.provider.name,
        "max_tokens": config.max_tokens,
        "template": config.prompt_template_path,
        "system_template": config.system_prompt_template_path,
        "judge": config.judge_profile_name,
    }
    if config.presentation.order != "canonical":
        stable["presentation"] = config.presentation.to_dict()
    digest = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "-", config.name).strip("-")
    return f"{safe_name}-{digest}"


def _request_id(run_id: str, stage: str, item_id: str) -> str:
    return f"{run_id}:{stage}:{item_id}"


def _metadata(config: ExperimentConfig, run_id: str, created_at: str) -> RunMetadata:
    return RunMetadata(
        run_id=run_id,
        domain=config.domain,
        mode=config.mode,
        visibility=config.visibility,
        stage=config.stage,
        split=config.split,
        model_name=config.model_name,
        provider_name=config.provider.name,
        prompt_template=config.prompt_template_path,
        judge_profile=config.judge_profile_name,
        created_at=created_at,
    )


def _model_request(
    config: ExperimentConfig,
    *,
    request_id: str,
    item_id: str,
    prompt_text: str | None = None,
    messages: tuple[Message, ...] = (),
) -> ModelRequest:
    return ModelRequest(
        request_id=request_id,
        model=config.model_name,
        max_tokens=config.max_tokens,
        prompt_text=prompt_text,
        messages=messages,
        temperature=config.budget.temperature,
        top_p=config.budget.top_p,
        metadata={
            "domain": config.domain,
            "mode": config.mode,
            "stage": config.stage,
            "item_id": item_id,
        },
    )


def _system_prompt(config: ExperimentConfig) -> str | None:
    path = config.system_prompt_template_path
    if path is None:
        return None
    try:
        text = _resolve_public_path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise OfflineRunnerError(f"cannot read public system prompt: {path}") from exc
    if not text:
        raise OfflineRunnerError(f"public system prompt is empty: {path}")
    return text


def _prompt_messages(
    config: ExperimentConfig,
    prompt_text: str,
) -> tuple[Message, ...]:
    system = _system_prompt(config)
    if system is None:
        return ()
    return (
        Message(role="system", content=system),
        Message(role="user", content=prompt_text),
    )


def _validate_limit(value: int | None, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OfflineRunnerError(f"{field} must be a positive integer")
    return value


def _present_contest_suites(
    config: ExperimentConfig,
    suites: tuple[ContestSuite, ...],
) -> tuple[tuple[ContestSuite, PresentationOrderRecord], ...]:
    """Apply the explicit presentation layer using the full-suite index."""

    return tuple(
        present_contest_suite(
            suite,
            order=config.presentation.order,
            seed=config.presentation.seed,
            suite_index=suite_index,
            seed_suite_id_template=config.presentation.seed_suite_id_template,
        )
        for suite_index, suite in enumerate(suites)
    )


def _filter_items(
    items: tuple[ProblemRecord | ContestSuite, ...],
    item_ids: Iterable[str] | None,
) -> tuple[ProblemRecord | ContestSuite, ...]:
    if item_ids is None:
        return items
    requested = tuple(item_ids)
    if not requested:
        return ()
    if len(requested) != len(set(requested)):
        raise OfflineRunnerError("item_ids cannot contain duplicates")
    by_id = {
        item.problem_id if isinstance(item, ProblemRecord) else item.suite_id: item
        for item in items
    }
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise OfflineRunnerError(f"unknown requested item IDs: {missing[:10]}")
    requested_set = set(requested)
    return tuple(
        item
        for item in items
        if (item.problem_id if isinstance(item, ProblemRecord) else item.suite_id)
        in requested_set
    )


def list_nl_item_ids(
    config: ExperimentConfig,
    *,
    limit: int | None = None,
) -> tuple[str, ...]:
    """List canonical run units without rendering prompts or calling providers."""

    checked_limit = _validate_limit(limit, "limit")
    if config.mode == "single_problem":
        items: tuple[ProblemRecord | ContestSuite, ...] = load_single_problems(
            config.domain,
            config.split,
            _data_source(config),
            strict=config.strict_data,
        )
        result = tuple(item.problem_id for item in items if isinstance(item, ProblemRecord))
    else:
        suites = load_contest_suites(
            config.domain,
            config.split,
            _data_source(config),
            strict=config.strict_data,
        )
        result = tuple(suite.suite_id for suite in suites)
    return result[:checked_limit] if checked_limit is not None else result


def prepare_single_problem_requests(
    config: ExperimentConfig,
    *,
    limit: int | None = None,
    item_ids: Iterable[str] | None = None,
) -> tuple[PreparedNLRequest, ...]:
    """Load, render, and build requests through the shared single path."""

    _validate_config(config, "single_problem")
    if config.stage != "one_stage":
        raise OfflineRunnerError("single-stage runner requires stage='one_stage'")
    checked_limit = _validate_limit(limit, "limit")
    problems = load_single_problems(
        config.domain,
        config.split,
        _data_source(config),
        strict=config.strict_data,
    )
    problems = tuple(
        item
        for item in _filter_items(tuple(problems), item_ids)
        if isinstance(item, ProblemRecord)
    )
    if checked_limit is not None:
        problems = problems[:checked_limit]
    template = PromptTemplate.from_file(_resolve_public_path(config.prompt_template_path))
    run_id = _run_id(config)
    prepared: list[PreparedNLRequest] = []
    for problem in problems:
        prompt = render_single_prompt(
            problem,
            template,
            config.visibility,
            budget_tokens=config.max_tokens,
        )
        request = _model_request(
            config,
            request_id=_request_id(run_id, config.stage, problem.problem_id),
            prompt_text=prompt if config.system_prompt_template_path is None else None,
            messages=_prompt_messages(config, prompt),
            item_id=problem.problem_id,
        )
        prepared.append(PreparedNLRequest(problem, prompt, request))
    return tuple(prepared)


def prepare_contest_requests(
    config: ExperimentConfig,
    *,
    limit_suites: int | None = None,
    item_ids: Iterable[str] | None = None,
) -> tuple[PreparedNLRequest, ...]:
    """Load, render, and build requests through the shared contest path."""

    _validate_config(config, "contest")
    if config.stage != "one_stage":
        raise OfflineRunnerError("single-stage runner requires stage='one_stage'")
    checked_limit = _validate_limit(limit_suites, "limit_suites")
    suites = load_contest_suites(
        config.domain,
        config.split,
        _data_source(config),
        strict=config.strict_data,
    )
    presented_suites = _present_contest_suites(config, suites)
    requested = (
        None if item_ids is None else set(item_ids)
    )
    if requested is not None:
        if not requested:
            raise OfflineRunnerError("item_ids cannot be empty")
        known = {suite.suite_id for suite, _ in presented_suites}
        missing = sorted(requested - known)
        if missing:
            raise OfflineRunnerError(f"unknown requested item IDs: {missing[:10]}")
        presented_suites = tuple(
            pair for pair in presented_suites if pair[0].suite_id in requested
        )
    if checked_limit is not None:
        presented_suites = presented_suites[:checked_limit]
    template = PromptTemplate.from_file(_resolve_public_path(config.prompt_template_path))
    run_id = _run_id(config)
    prepared: list[PreparedNLRequest] = []
    for suite, presentation in presented_suites:
        prompt = render_contest_prompt(
            suite,
            template,
            config.visibility,
            budget_tokens=config.max_tokens,
        )
        request = _model_request(
            config,
            request_id=_request_id(run_id, config.stage, suite.suite_id),
            prompt_text=prompt if config.system_prompt_template_path is None else None,
            messages=_prompt_messages(config, prompt),
            item_id=suite.suite_id,
        )
        prepared.append(
            PreparedNLRequest(
                suite,
                prompt,
                request,
                presentation=presentation,
            )
        )
    return tuple(prepared)


def prepare_two_stage_requests(
    config_stage1: ExperimentConfig,
    config_stage2: ExperimentConfig,
    *,
    limit: int | None = None,
    stage1_output: str = "[DRY RUN STAGE 1 OUTPUT]",
    protocol: TwoStageProtocol | None = None,
    item_ids: Iterable[str] | None = None,
) -> tuple[PreparedTwoStageRequest, ...]:
    """Prepare deterministic Stage 1/2 requests without provider execution."""

    if config_stage1.stage != "stage1" or config_stage2.stage != "stage2":
        raise OfflineRunnerError("two-stage preparation requires stage1 and stage2")
    checked_limit = _validate_limit(limit, "limit")
    presentations: dict[str, PresentationOrderRecord] = {}
    if config_stage1.mode == "single_problem":
        items: tuple[ProblemRecord | ContestSuite, ...] = load_single_problems(
            config_stage1.domain,
            config_stage1.split,
            _data_source(config_stage1),
            strict=config_stage1.strict_data,
        )
    else:
        canonical_suites = load_contest_suites(
            config_stage1.domain,
            config_stage1.split,
            _data_source(config_stage1),
            strict=config_stage1.strict_data,
        )
        presented_suites = _present_contest_suites(
            config_stage1, canonical_suites
        )
        items = tuple(suite for suite, _ in presented_suites)
        presentations = {
            suite.suite_id: presentation
            for suite, presentation in presented_suites
        }
    items = _filter_items(items, item_ids)
    if presentations:
        selected = {
            item.problem_id if isinstance(item, ProblemRecord) else item.suite_id
            for item in items
        }
        presentations = {
            item_id: row
            for item_id, row in presentations.items()
            if item_id in selected
        }
    if checked_limit is not None:
        items = items[:checked_limit]
    template1 = PromptTemplate.from_file(
        _resolve_public_path(config_stage1.prompt_template_path)
    )
    template2 = PromptTemplate.from_file(
        _resolve_public_path(config_stage2.prompt_template_path)
    )
    run_id = _two_stage_run_id(config_stage1, config_stage2)
    checked_protocol = protocol or default_two_stage_protocol(config_stage1.domain)
    _validate_two_stage_protocol(config_stage1.domain, checked_protocol)
    handoff = TransientStageHandoff(visible_output=stage1_output)
    stage1_hash = _stage_handoff_sha256(handoff, checked_protocol)
    result: list[PreparedTwoStageRequest] = []
    for item in items:
        item_id = item.problem_id if isinstance(item, ProblemRecord) else item.suite_id
        prompt1 = render_two_stage_prompt(
            item,
            "stage1",
            template1,
            config_stage1.visibility,
            budget_tokens=config_stage1.max_tokens,
        )
        request1 = _model_request(
            config_stage1,
            request_id=_request_id(run_id, "stage1", item_id),
            prompt_text=(
                prompt1
                if config_stage1.system_prompt_template_path is None
                else None
            ),
            messages=_prompt_messages(config_stage1, prompt1),
            item_id=item_id,
        )
        prompt2 = render_two_stage_prompt(
            item,
            "stage2",
            template2,
            config_stage2.visibility,
            budget_tokens=config_stage2.max_tokens,
        )
        request2 = _model_request(
            config_stage2,
            request_id=_request_id(run_id, "stage2", item_id),
            messages=_stage2_messages(
                _assemble_stage2_prompt(prompt2, handoff, checked_protocol),
                system_prompt=_system_prompt(config_stage2),
            ),
            item_id=item_id,
        )
        result.append(
            PreparedTwoStageRequest(
                item=item,
                stage1_prompt=prompt1,
                stage1_request=request1,
                stage2_prompt=prompt2,
                stage2_request=request2,
                stage1_output_sha256=stage1_hash,
                presentation=presentations.get(item_id),
            )
        )
    return tuple(result)


def _attempt(
    config: ExperimentConfig,
    *,
    run_id: str,
    request: ModelRequest,
    prompt_text: str,
    created_at: str,
    suite_id: str | None,
    problem_id: str | None,
    problem_label: str | None,
    response: ModelResponse | None,
    parent_request_id: str | None,
    stage1_request_id: str | None,
    stage2_request_id: str | None,
    stage_input_kind: Literal["public_prompt", "stage1_output"],
    stage_input_sha256: str,
    error_type: str | None = None,
    error_message: str | None = None,
) -> AttemptRecord:
    return AttemptRecord(
        run_id=run_id,
        request_id=request.request_id,
        domain=config.domain,
        mode=config.mode,
        visibility=config.visibility,
        stage=config.stage,
        split=config.split,
        suite_id=suite_id,
        problem_id=problem_id,
        problem_label=problem_label,
        model_name=config.model_name,
        provider_name=config.provider.name,
        prompt_template=config.prompt_template_path,
        prompt_sha256=request.input_sha256,
        prompt_text=prompt_text,
        parent_request_id=parent_request_id,
        stage1_request_id=stage1_request_id,
        stage2_request_id=stage2_request_id,
        stage_input_kind=stage_input_kind,
        stage_input_sha256=stage_input_sha256,
        response_text=response.response_text if response is not None else "",
        usage=response.usage if response is not None else UsageInfo(),
        finish_reason=response.finish_reason if response is not None else None,
        error_type=(
            None
            if response is not None
            else (error_type or "provider_error")
        ),
        error_message=(
            None
            if response is not None
            else (error_message or "provider completion failed")
        ),
        created_at=created_at,
    )


def _caused_by_timeout(error: BaseException) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, TimeoutError):
            return True
        visited.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _provider_failure(error: BaseException) -> CompletionOutcome:
    if _caused_by_timeout(error):
        return CompletionOutcome(
            None,
            "provider_timeout",
            "provider request timed out within the configured retry policy",
        )
    if isinstance(error, ProviderAuthError):
        return CompletionOutcome(
            None, "provider_auth_error", "provider authentication failed"
        )
    if isinstance(error, ProviderConfigError):
        return CompletionOutcome(
            None, "provider_config_error", "provider configuration is invalid"
        )
    if isinstance(error, ProviderRetryError):
        return CompletionOutcome(
            None, "provider_retry_exhausted", "provider retries were exhausted"
        )
    if isinstance(error, ProviderResponseError):
        return CompletionOutcome(
            None, "provider_response_error", "provider response could not be parsed"
        )
    if isinstance(error, ProviderRequestError):
        return CompletionOutcome(
            None, "provider_request_error", "provider request failed"
        )
    if isinstance(error, KeyError):
        return CompletionOutcome(
            None, "provider_replay_missing", "provider replay entry is missing"
        )
    return CompletionOutcome(
        None, "provider_protocol_error", "provider completion violated the protocol"
    )


def _complete(provider: Provider, request: ModelRequest) -> CompletionOutcome:
    try:
        return CompletionOutcome(provider.complete(request))
    except (KeyError, ProviderError, ValueError) as exc:
        return _provider_failure(exc)


def _stage_handoff(
    provider: Provider,
    request: ModelRequest,
    response: ModelResponse | None,
) -> TransientStageHandoff:
    if response is None:
        return TransientStageHandoff()
    consumer = getattr(provider, "consume_stage_handoff", None)
    if callable(consumer):
        value = consumer(request.request_id, response.response_text)
        if isinstance(value, TransientStageHandoff):
            return value
        if isinstance(value, str):
            return TransientStageHandoff(
                reasoning_content=value,
                visible_output=response.response_text,
            )
    return TransientStageHandoff(visible_output=response.response_text)


def _single_parse(problem: ProblemRecord, text: str) -> str | None:
    if problem.domain == "coding":
        return extract_cpp_code(text)
    if problem.domain == "math":
        return extract_math_answer(text)
    return extract_ar_answer(text)


def _contest_sections(domain: str, text: str) -> dict[str, str]:
    if domain == "coding":
        return extract_contest_cpp_sections(text)
    if domain in {"math", "abstract_reasoning"}:
        matches = list(_NL_SECTION.finditer(text))
        sections: dict[str, str] = {}
        duplicates: set[str] = set()
        for index, match in enumerate(matches):
            raw_label = match.group(1).upper()
            label = (
                chr(ord("A") + int(raw_label) - 1)
                if raw_label.isdigit()
                else raw_label
            )
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            if label in sections:
                duplicates.add(label)
            sections[label] = text[match.end() : end].strip()
        for label in duplicates:
            sections.pop(label, None)
        return sections

    return {}


def _contest_parse(problem: ProblemRecord, sections: dict[str, str]) -> str | None:
    label = problem.problem_label
    if label is None or label not in sections:
        return None
    section = sections[label]
    if problem.domain == "math":
        return extract_math_answer(section, allow_final_answer_fallback=True)
    if problem.domain == "abstract_reasoning":
        return extract_ar_answer(section, allow_final_answer_fallback=True)
    return section


def _parsed_record(
    config: ExperimentConfig,
    *,
    run_id: str,
    request_id: str,
    problem: ProblemRecord,
    parsed: str | None,
    provider_failed: bool,
    created_at: str,
    stage1_request_id: str | None = None,
    stage2_request_id: str | None = None,
) -> ParsedAnswerRecord:
    if provider_failed:
        status: Literal["parsed", "missing", "parse_error"] = "missing"
        error_type = "provider_error"
        error_message = "no provider response was available"
    elif parsed is None:
        status = "missing"
        error_type = "parse_error"
        error_message = "no scoreable answer could be extracted"
    else:
        status = "parsed"
        error_type = None
        error_message = None
    return ParsedAnswerRecord(
        run_id=run_id,
        request_id=request_id,
        domain=config.domain,
        mode=config.mode,
        visibility=config.visibility,
        stage=config.stage,
        split=config.split,
        suite_id=problem.suite_id,
        problem_id=problem.problem_id,
        problem_label=problem.problem_label,
        stage1_request_id=stage1_request_id,
        stage2_request_id=stage2_request_id,
        parsed_answer=parsed,
        parse_status=status,
        error_type=error_type,
        error_message=error_message,
        created_at=created_at,
    )


def _judge_record(
    config: ExperimentConfig,
    *,
    run_id: str,
    request_id: str,
    problem: ProblemRecord,
    parsed: str | None,
    created_at: str,
    stage1_request_id: str | None = None,
    stage2_request_id: str | None = None,
) -> JudgeResultRecord:
    common = dict(
        run_id=run_id,
        request_id=request_id,
        domain=config.domain,
        mode=config.mode,
        visibility=config.visibility,
        stage=config.stage,
        split=config.split,
        suite_id=problem.suite_id,
        problem_id=problem.problem_id,
        problem_label=problem.problem_label,
        stage1_request_id=stage1_request_id,
        stage2_request_id=stage2_request_id,
        created_at=created_at,
    )
    if parsed is None:
        return JudgeResultRecord(
            **common,
            judge_status="not_judged",
            verdict="missing",
            score=0.0,
            error_type="parse_error",
            error_message="no parsed answer was available",
        )
    try:
        if problem.domain == "coding":
            upstream_id = problem.domain_payload.get("upstream_id")
            if not isinstance(upstream_id, str) or not upstream_id.strip():
                raise ValueError("missing upstream_id")
            result = MockCodingVerifier().verify(upstream_id, parsed)
            verdict = result.verdict
            score = 1.0 if result.accepted else 0.0
        elif problem.domain == "math":
            result = MockMathJudge().judge(problem, parsed)
            verdict = result.verdict
            score = 1.0 if result.correct else 0.0
        else:
            result = MockARScorer().score(problem, parsed)
            verdict = result.verdict
            score = result.score
        return JudgeResultRecord(
            **common,
            judge_status="judged",
            verdict=verdict,
            score=score,
            error_type=None,
            error_message=None,
        )
    except (TypeError, ValueError):
        return JudgeResultRecord(
            **common,
            judge_status="judge_error",
            verdict="judge_error",
            score=0.0,
            error_type="judge_error",
            error_message="offline mock evaluator rejected the parsed answer",
        )


def _not_judged_record(
    config: ExperimentConfig,
    *,
    run_id: str,
    request_id: str,
    problem: ProblemRecord,
    parsed: str | None,
    created_at: str,
    stage1_request_id: str | None = None,
    stage2_request_id: str | None = None,
) -> JudgeResultRecord:
    return JudgeResultRecord(
        run_id=run_id,
        request_id=request_id,
        domain=config.domain,
        mode=config.mode,
        visibility=config.visibility,
        stage=config.stage,
        split=config.split,
        suite_id=problem.suite_id,
        problem_id=problem.problem_id,
        problem_label=problem.problem_label,
        stage1_request_id=stage1_request_id,
        stage2_request_id=stage2_request_id,
        judge_status="not_judged",
        verdict="pending_offline_scoring" if parsed is not None else "missing",
        score=0.0,
        error_type=None if parsed is not None else "parse_error",
        error_message=(
            None if parsed is not None else "no parsed answer was available"
        ),
        created_at=created_at,
    )


def _result_record(
    config: ExperimentConfig,
    *,
    judge_mode: Literal["mock", "none"],
    run_id: str,
    request_id: str,
    problem: ProblemRecord,
    parsed: str | None,
    created_at: str,
    stage1_request_id: str | None = None,
    stage2_request_id: str | None = None,
) -> JudgeResultRecord:
    if judge_mode == "none":
        return _not_judged_record(
            config,
            run_id=run_id,
            request_id=request_id,
            problem=problem,
            parsed=parsed,
            created_at=created_at,
            stage1_request_id=stage1_request_id,
            stage2_request_id=stage2_request_id,
        )
    return _judge_record(
        config,
        run_id=run_id,
        request_id=request_id,
        problem=problem,
        parsed=parsed,
        created_at=created_at,
        stage1_request_id=stage1_request_id,
        stage2_request_id=stage2_request_id,
    )


def _summary(
    config: ExperimentConfig,
    *,
    run_id: str,
    attempts: Iterable[AttemptRecord],
    parsed: Iterable[ParsedAnswerRecord],
    judged: Iterable[JudgeResultRecord],
    created_at: str,
) -> RunSummary:
    attempts_tuple = tuple(attempts)
    parsed_tuple = tuple(parsed)
    judged_tuple = tuple(judged)
    return RunSummary(
        run_id=run_id,
        domain=config.domain,
        mode=config.mode,
        visibility=config.visibility,
        stage=config.stage,
        split=config.split,
        model_name=config.model_name,
        provider_name=config.provider.name,
        attempt_count=len(attempts_tuple),
        problem_count=len(parsed_tuple),
        parsed_count=sum(row.parse_status == "parsed" for row in parsed_tuple),
        judged_count=sum(row.judge_status == "judged" for row in judged_tuple),
        correct_count=sum(row.score >= 1.0 for row in judged_tuple),
        total_score=sum(row.score for row in judged_tuple),
        error_count=sum(row.error_type is not None for row in attempts_tuple)
        + sum(row.error_type is not None for row in parsed_tuple)
        + sum(row.judge_status == "judge_error" for row in judged_tuple),
        created_at=created_at,
    )


def run_single_problem_nl(
    config: ExperimentConfig,
    provider: Provider,
    *,
    limit: int | None = None,
    judge_mode: Literal["mock", "none"] = "mock",
    item_ids: Iterable[str] | None = None,
    on_item_complete: Callable[[str, NLRunArtifacts], None] | None = None,
) -> NLRunArtifacts:
    """Run configured single-problem rows through the shared provider path."""

    checked_provider = _validate_provider(provider)
    prepared_requests = prepare_single_problem_requests(
        config, limit=limit, item_ids=item_ids
    )
    created_at = _utc_now()
    run_id = _run_id(config)
    attempts: list[AttemptRecord] = []
    parsed_rows: list[ParsedAnswerRecord] = []
    judged_rows: list[JudgeResultRecord] = []

    for prepared in prepared_requests:
        if not isinstance(prepared.item, ProblemRecord):
            raise OfflineRunnerError("single request preparation returned a contest suite")
        problem = prepared.item
        prompt = prepared.prompt_text
        request = prepared.request
        request_id = request.request_id
        completion = _complete(checked_provider, request)
        response = completion.response
        attempt_start = len(attempts)
        parsed_start = len(parsed_rows)
        judged_start = len(judged_rows)
        attempts.append(
            _attempt(
                config,
                run_id=run_id,
                request=request,
                prompt_text=prompt,
                created_at=created_at,
                suite_id=problem.suite_id,
                problem_id=problem.problem_id,
                problem_label=None,
                response=response,
                parent_request_id=None,
                stage1_request_id=None,
                stage2_request_id=None,
                stage_input_kind="public_prompt",
                stage_input_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                error_type=completion.error_type,
                error_message=completion.error_message,
            )
        )
        parsed = _single_parse(problem, response.response_text) if response is not None else None
        parsed_rows.append(
            _parsed_record(
                config,
                run_id=run_id,
                request_id=request_id,
                problem=problem,
                parsed=parsed,
                provider_failed=response is None,
                created_at=created_at,
            )
        )
        judged_rows.append(
            _result_record(
                config,
                judge_mode=judge_mode,
                run_id=run_id,
                request_id=request_id,
                problem=problem,
                parsed=parsed,
                created_at=created_at,
            )
        )
        if on_item_complete is not None:
            partial_attempts = attempts[attempt_start:]
            partial_parsed = parsed_rows[parsed_start:]
            partial_judged = judged_rows[judged_start:]
            on_item_complete(
                problem.problem_id,
                NLRunArtifacts(
                    metadata=_metadata(config, run_id, created_at),
                    attempts=tuple(partial_attempts),
                    parsed_answers=tuple(partial_parsed),
                    judge_results=tuple(partial_judged),
                    summary=_summary(
                        config,
                        run_id=run_id,
                        attempts=partial_attempts,
                        parsed=partial_parsed,
                        judged=partial_judged,
                        created_at=created_at,
                    ),
                ),
            )

    summary = _summary(
        config,
        run_id=run_id,
        attempts=attempts,
        parsed=parsed_rows,
        judged=judged_rows,
        created_at=created_at,
    )
    return NLRunArtifacts(
        metadata=_metadata(config, run_id, created_at),
        attempts=tuple(attempts),
        parsed_answers=tuple(parsed_rows),
        judge_results=tuple(judged_rows),
        summary=summary,
    )


def run_contest_nl(
    config: ExperimentConfig,
    provider: Provider,
    *,
    limit_suites: int | None = None,
    judge_mode: Literal["mock", "none"] = "mock",
    item_ids: Iterable[str] | None = None,
    on_item_complete: Callable[[str, NLRunArtifacts], None] | None = None,
) -> NLRunArtifacts:
    """Run complete suites through the shared provider path."""

    checked_provider = _validate_provider(provider)
    prepared_requests = prepare_contest_requests(
        config, limit_suites=limit_suites, item_ids=item_ids
    )
    created_at = _utc_now()
    run_id = _run_id(config)
    attempts: list[AttemptRecord] = []
    parsed_rows: list[ParsedAnswerRecord] = []
    judged_rows: list[JudgeResultRecord] = []

    for prepared in prepared_requests:
        if not isinstance(prepared.item, ContestSuite):
            raise OfflineRunnerError("contest request preparation returned a problem")
        suite = prepared.item
        prompt = prepared.prompt_text
        request = prepared.request
        request_id = request.request_id
        completion = _complete(checked_provider, request)
        response = completion.response
        attempt_start = len(attempts)
        parsed_start = len(parsed_rows)
        judged_start = len(judged_rows)
        attempts.append(
            _attempt(
                config,
                run_id=run_id,
                request=request,
                prompt_text=prompt,
                created_at=created_at,
                suite_id=suite.suite_id,
                problem_id=None,
                problem_label=None,
                response=response,
                parent_request_id=None,
                stage1_request_id=None,
                stage2_request_id=None,
                stage_input_kind="public_prompt",
                stage_input_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                error_type=completion.error_type,
                error_message=completion.error_message,
            )
        )
        sections = _contest_sections(config.domain, response.response_text) if response else {}
        for problem in suite.problems:
            parsed = _contest_parse(problem, sections)
            parsed_rows.append(
                _parsed_record(
                    config,
                    run_id=run_id,
                    request_id=request_id,
                    problem=problem,
                    parsed=parsed,
                    provider_failed=response is None,
                    created_at=created_at,
                )
            )
            judged_rows.append(
                _result_record(
                    config,
                    judge_mode=judge_mode,
                    run_id=run_id,
                    request_id=request_id,
                    problem=problem,
                    parsed=parsed,
                    created_at=created_at,
                )
            )
        if on_item_complete is not None:
            partial_attempts = attempts[attempt_start:]
            partial_parsed = parsed_rows[parsed_start:]
            partial_judged = judged_rows[judged_start:]
            partial_presentation = (
                (prepared.presentation,)
                if prepared.presentation is not None
                and prepared.presentation.presentation_order != "canonical"
                else ()
            )
            on_item_complete(
                suite.suite_id,
                NLRunArtifacts(
                    metadata=_metadata(config, run_id, created_at),
                    attempts=tuple(partial_attempts),
                    parsed_answers=tuple(partial_parsed),
                    judge_results=tuple(partial_judged),
                    summary=_summary(
                        config,
                        run_id=run_id,
                        attempts=partial_attempts,
                        parsed=partial_parsed,
                        judged=partial_judged,
                        created_at=created_at,
                    ),
                    presentation_orders=partial_presentation,
                ),
            )

    summary = _summary(
        config,
        run_id=run_id,
        attempts=attempts,
        parsed=parsed_rows,
        judged=judged_rows,
        created_at=created_at,
    )
    return NLRunArtifacts(
        metadata=_metadata(config, run_id, created_at),
        attempts=tuple(attempts),
        parsed_answers=tuple(parsed_rows),
        judge_results=tuple(judged_rows),
        summary=summary,
        presentation_orders=tuple(
            row.presentation
            for row in prepared_requests
            if row.presentation is not None
            and row.presentation.presentation_order != "canonical"
        ),
    )


def _two_stage_run_id(
    config_stage1: ExperimentConfig, config_stage2: ExperimentConfig
) -> str:
    digest = hashlib.sha256(
        f"{_run_id(config_stage1)}\n{_run_id(config_stage2)}".encode("utf-8")
    ).hexdigest()[:12]
    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]+", "-", config_stage2.name
    ).strip("-")
    return f"{safe_name}-{digest}"


def _selected_handoff_payload(
    handoff: TransientStageHandoff,
    protocol: TwoStageProtocol,
) -> dict[str, str]:
    payload: dict[str, str] = {}
    for channel in protocol.handoff_channels:
        if channel == "reasoning_content":
            payload[channel] = handoff.reasoning_content
        else:
            payload[channel] = handoff.visible_output
    return payload


def _handoff_has_selected_content(
    handoff: TransientStageHandoff,
    protocol: TwoStageProtocol,
) -> bool:
    return any(
        value.strip()
        for value in _selected_handoff_payload(handoff, protocol).values()
    )


def _validate_two_stage_protocol(domain: str, protocol: TwoStageProtocol) -> None:
    if domain == "coding":
        if (
            protocol.prompt_assembly
            not in {
                "coding_reasoning_visible_trace",
                "coding_visible_output_only",
            }
            or protocol.include_original_problems
        ):
            raise OfflineRunnerError(
                "Coding Stage 2 must use trace-only formal assembly"
            )
        return
    if domain in {"math", "abstract_reasoning"}:
        if (
            protocol.prompt_assembly != "reasoning_visible_trace"
            or protocol.include_original_problems
        ):
            raise OfflineRunnerError(
                "Math/AR Stage 2 must use trace-only formal assembly"
            )
        return
    raise OfflineRunnerError(f"unsupported two-stage domain: {domain}")


def _stage_handoff_sha256(
    handoff: TransientStageHandoff,
    protocol: TwoStageProtocol,
) -> str:
    """Hash exactly the transient Stage 1 channels consumed by Stage 2."""

    canonical = json.dumps(
        _selected_handoff_payload(handoff, protocol),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _assemble_stage2_prompt(
    task_prompt: str,
    handoff: TransientStageHandoff,
    protocol: TwoStageProtocol,
) -> str:
    """Assemble the formal domain-specific Stage 2 user prompt in memory."""

    selected = _selected_handoff_payload(handoff, protocol)
    reasoning = selected.get("reasoning_content", "")
    visible = selected.get("visible_output", "")
    if protocol.prompt_assembly == "coding_visible_output_only":
        if protocol.include_original_problems:
            raise OfflineRunnerError(
                "Coding visible-only finalizer cannot receive original problems"
            )
        if _STAGE1_VISIBLE_MARKER not in task_prompt:
            raise OfflineRunnerError(
                "Coding visible-only Stage 2 template is missing output marker"
            )
        return task_prompt.replace(
            _STAGE1_VISIBLE_MARKER,
            visible if visible.strip() else "<EMPTY>",
        )

    if protocol.prompt_assembly == "coding_reasoning_visible_trace":
        if protocol.include_original_problems:
            raise OfflineRunnerError(
                "Coding trace-only finalizer cannot receive original problems"
            )
        if _STAGE1_TRACE_MARKER not in task_prompt:
            raise OfflineRunnerError("Coding Stage 2 template is missing trace marker")
        trace = (
            "[REASONING_CONTENT]\n"
            + (reasoning if reasoning.strip() else "<UNAVAILABLE>")
            + "\n\n[VISIBLE_OUTPUT]\n"
            + (visible if visible.strip() else "<EMPTY>")
        )
        return task_prompt.replace(_STAGE1_TRACE_MARKER, trace)

    if protocol.prompt_assembly != "reasoning_visible_trace":
        raise OfflineRunnerError("unsupported Stage 2 prompt assembly")
    if protocol.include_original_problems:
        raise OfflineRunnerError("Math/AR Stage 2 cannot receive original problems")
    missing = [
        marker
        for marker in (_STAGE1_REASONING_MARKER, _STAGE1_VISIBLE_MARKER)
        if marker not in task_prompt
    ]
    if missing:
        raise OfflineRunnerError("Stage 2 template is missing handoff markers")
    return (
        task_prompt.replace(
            _STAGE1_REASONING_MARKER,
            reasoning if reasoning.strip() else "<UNAVAILABLE>",
        )
        .replace(
            _STAGE1_VISIBLE_MARKER,
            visible if visible.strip() else "<EMPTY>",
        )
    )


def _stage2_messages(
    assembled_prompt: str,
    *,
    system_prompt: str | None = None,
) -> tuple[Message, ...]:
    """Mirror formal runners with an optional static system message."""

    messages: list[Message] = []
    if system_prompt is not None:
        messages.append(Message(role="system", content=system_prompt))
    messages.append(Message(role="user", content=assembled_prompt))
    return tuple(messages)


def run_two_stage_nl(
    config_stage1: ExperimentConfig,
    config_stage2: ExperimentConfig,
    provider_stage1: Provider,
    provider_stage2: Provider,
    *,
    limit: int | None = None,
    judge_mode: Literal["mock", "none"] = "mock",
    protocol: TwoStageProtocol | None = None,
    item_ids: Iterable[str] | None = None,
    on_item_complete: Callable[[str, NLRunArtifacts], None] | None = None,
) -> NLRunArtifacts:
    """Run an offline two-stage pipeline and score only Stage 2 outputs."""

    if config_stage1.stage != "stage1" or config_stage2.stage != "stage2":
        raise OfflineRunnerError("two-stage runner requires stage1 and stage2 configs")
    comparable = (
        "domain",
        "mode",
        "visibility",
        "split",
        "data_source",
        "presentation",
    )
    for field in comparable:
        if getattr(config_stage1, field) != getattr(config_stage2, field):
            raise OfflineRunnerError(f"two-stage configs disagree on {field}")
    _validate_config(config_stage1, config_stage1.mode)
    _validate_config(config_stage2, config_stage2.mode)
    checked_stage1 = _validate_provider(provider_stage1)
    checked_stage2 = _validate_provider(provider_stage2)
    checked_protocol = protocol or default_two_stage_protocol(config_stage1.domain)
    _validate_two_stage_protocol(config_stage1.domain, checked_protocol)
    created_at = _utc_now()
    run_id = _two_stage_run_id(config_stage1, config_stage2)
    template1 = PromptTemplate.from_file(
        _resolve_public_path(config_stage1.prompt_template_path)
    )
    template2 = PromptTemplate.from_file(
        _resolve_public_path(config_stage2.prompt_template_path)
    )
    attempts: list[AttemptRecord] = []
    parsed_rows: list[ParsedAnswerRecord] = []
    judged_rows: list[JudgeResultRecord] = []

    checked_limit = _validate_limit(limit, "limit")
    presentation_orders: tuple[PresentationOrderRecord, ...] = ()
    if config_stage1.mode == "single_problem":
        items: tuple[ProblemRecord | ContestSuite, ...] = load_single_problems(
            config_stage1.domain,
            config_stage1.split,
            _data_source(config_stage1),
            strict=config_stage1.strict_data,
        )
    else:
        canonical_suites = load_contest_suites(
            config_stage1.domain,
            config_stage1.split,
            _data_source(config_stage1),
            strict=config_stage1.strict_data,
        )
        presented_suites = _present_contest_suites(
            config_stage1, canonical_suites
        )
        items = tuple(suite for suite, _ in presented_suites)
        presentation_orders = tuple(
            record
            for _, record in presented_suites
            if record.presentation_order != "canonical"
        )
    items = _filter_items(items, item_ids)
    if presentation_orders:
        selected_suite_ids = {
            item.suite_id for item in items if isinstance(item, ContestSuite)
        }
        presentation_orders = tuple(
            row for row in presentation_orders if row.suite_id in selected_suite_ids
        )
    if checked_limit is not None:
        items = items[:checked_limit]
        presentation_orders = presentation_orders[:checked_limit]

    for item in items:
        attempt_start = len(attempts)
        parsed_start = len(parsed_rows)
        judged_start = len(judged_rows)
        item_id = item.problem_id if isinstance(item, ProblemRecord) else item.suite_id
        suite_id = item.suite_id
        problem_id = item.problem_id if isinstance(item, ProblemRecord) else None
        prompt1 = render_two_stage_prompt(
            item,
            "stage1",
            template1,
            config_stage1.visibility,
            budget_tokens=config_stage1.max_tokens,
        )
        request1_id = _request_id(run_id, "stage1", item_id)
        request1 = _model_request(
            config_stage1,
            request_id=request1_id,
            prompt_text=(
                prompt1
                if config_stage1.system_prompt_template_path is None
                else None
            ),
            messages=_prompt_messages(config_stage1, prompt1),
            item_id=item_id,
        )
        completion1 = _complete(
            checked_stage1,
            request1,
        )
        response1 = completion1.response
        attempts.append(
            _attempt(
                config_stage1,
                run_id=run_id,
                request=request1,
                prompt_text=prompt1,
                created_at=created_at,
                suite_id=suite_id,
                problem_id=problem_id,
                problem_label=None,
                response=response1,
                parent_request_id=None,
                stage1_request_id=request1_id,
                stage2_request_id=None,
                stage_input_kind="public_prompt",
                stage_input_sha256=hashlib.sha256(
                    prompt1.encode("utf-8")
                ).hexdigest(),
                error_type=completion1.error_type,
                error_message=completion1.error_message,
            )
        )

        base_prompt2 = render_two_stage_prompt(
            item,
            "stage2",
            template2,
            config_stage2.visibility,
            budget_tokens=config_stage2.max_tokens,
        )
        stage1_handoff = _stage_handoff(checked_stage1, request1, response1)
        stage1_output_sha256 = _stage_handoff_sha256(
            stage1_handoff,
            checked_protocol,
        )
        messages2 = _stage2_messages(
            _assemble_stage2_prompt(
                base_prompt2,
                stage1_handoff,
                checked_protocol,
            ),
            system_prompt=_system_prompt(config_stage2),
        )
        request2_id = _request_id(run_id, "stage2", item_id)
        request2 = _model_request(
            config_stage2,
            request_id=request2_id,
            messages=messages2,
            item_id=item_id,
        )
        completion2 = (
            _complete(checked_stage2, request2)
            if response1 is not None
            and _handoff_has_selected_content(stage1_handoff, checked_protocol)
            else CompletionOutcome(None)
        )
        response2 = completion2.response
        if response1 is None:
            stage2_error_type = "stage1_failed"
            stage2_error_message = (
                "Stage 2 was skipped because Stage 1 did not complete"
            )
        elif not _handoff_has_selected_content(stage1_handoff, checked_protocol):
            stage2_error_type = "stage1_output_missing"
            stage2_error_message = (
                "Stage 2 was skipped because Stage 1 produced no usable handoff"
            )
        else:
            stage2_error_type = completion2.error_type
            stage2_error_message = completion2.error_message
        attempts.append(
            _attempt(
                config_stage2,
                run_id=run_id,
                request=request2,
                prompt_text=base_prompt2,
                created_at=created_at,
                suite_id=suite_id,
                problem_id=problem_id,
                problem_label=None,
                response=response2,
                parent_request_id=request1_id,
                stage1_request_id=request1_id,
                stage2_request_id=request2_id,
                stage_input_kind="stage1_output",
                stage_input_sha256=stage1_output_sha256,
                error_type=stage2_error_type,
                error_message=stage2_error_message,
            )
        )

        problems = (item,) if isinstance(item, ProblemRecord) else item.problems
        sections = (
            _contest_sections(config_stage2.domain, response2.response_text)
            if response2 is not None and isinstance(item, ContestSuite)
            else {}
        )
        for problem in problems:
            parsed = (
                _single_parse(problem, response2.response_text)
                if response2 is not None and isinstance(item, ProblemRecord)
                else _contest_parse(problem, sections)
            )
            parsed_rows.append(
                _parsed_record(
                    config_stage2,
                    run_id=run_id,
                    request_id=request2_id,
                    problem=problem,
                    parsed=parsed,
                    provider_failed=response2 is None,
                    created_at=created_at,
                    stage1_request_id=request1_id,
                    stage2_request_id=request2_id,
                )
            )
            judged_rows.append(
                _result_record(
                    config_stage2,
                    judge_mode=judge_mode,
                    run_id=run_id,
                    request_id=request2_id,
                    problem=problem,
                    parsed=parsed,
                    created_at=created_at,
                    stage1_request_id=request1_id,
                    stage2_request_id=request2_id,
                )
            )
        if on_item_complete is not None:
            partial_attempts = attempts[attempt_start:]
            partial_parsed = parsed_rows[parsed_start:]
            partial_judged = judged_rows[judged_start:]
            partial_presentation = tuple(
                row for row in presentation_orders if row.suite_id == suite_id
            )
            on_item_complete(
                item_id,
                NLRunArtifacts(
                    metadata=_metadata(config_stage2, run_id, created_at),
                    attempts=tuple(partial_attempts),
                    parsed_answers=tuple(partial_parsed),
                    judge_results=tuple(partial_judged),
                    summary=_summary(
                        config_stage2,
                        run_id=run_id,
                        attempts=partial_attempts,
                        parsed=partial_parsed,
                        judged=partial_judged,
                        created_at=created_at,
                    ),
                    presentation_orders=partial_presentation,
                ),
            )

    summary = _summary(
        config_stage2,
        run_id=run_id,
        attempts=attempts,
        parsed=parsed_rows,
        judged=judged_rows,
        created_at=created_at,
    )
    return NLRunArtifacts(
        metadata=_metadata(config_stage2, run_id, created_at),
        attempts=tuple(attempts),
        parsed_answers=tuple(parsed_rows),
        judge_results=tuple(judged_rows),
        summary=summary,
        presentation_orders=presentation_orders,
    )


def write_run_artifacts(artifacts: NLRunArtifacts, output_dir: str | Path) -> None:
    """Write standardized evaluator result files as strict JSON/JSONL."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows = {
        "attempts.jsonl": artifacts.attempts,
        "parsed_answers.jsonl": artifacts.parsed_answers,
        "judge_results.jsonl": artifacts.judge_results,
    }
    if artifacts.presentation_orders:
        rows["presentation_orders.jsonl"] = artifacts.presentation_orders
    for name, records in rows.items():
        (target / name).write_text(
            "".join(
                json.dumps(to_public_dict(record), ensure_ascii=False, allow_nan=False)
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
    (target / "run_summary.json").write_text(
        json.dumps(
            to_public_dict(artifacts.summary),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def default_mock_response(config: ExperimentConfig) -> str:
    """Return synthetic output matching the public toy data conventions."""

    if config.domain == "coding":
        program = "#include <iostream>\nint main() { std::cout << 1 << '\\n'; }"
        if config.mode == "single_problem":
            return f"```cpp\n{program}\n```"
        return "\n\n".join(
            f"Solution {label}\n```cpp\n{program}\n```" for label in "ABCDEF"
        )
    if config.domain == "math":
        if config.mode == "single_problem":
            return r"\boxed{1}"
        return "\n\n".join(
            f"## Problem {index}\nFinal Answer: \\boxed{{{index}}}"
            for index in range(1, 7)
        )
    if config.mode == "single_problem":
        return "<answer>answer-1</answer>"
    return "\n\n".join(
        f"## Problem {index}\nFinal Answer: <answer>answer-{index}</answer>"
        for index in range(1, 7)
    )


def default_stage1_mock_response(config: ExperimentConfig) -> str:
    """Return synthetic Stage 1 material containing complete candidate answers."""

    return "Synthetic Stage 1 draft with complete candidates.\n\n" + default_mock_response(
        config
    )


def derive_two_stage_configs(
    base: ExperimentConfig,
) -> tuple[ExperimentConfig, ExperimentConfig]:
    """Derive public Stage 1/2 contest configs from a one-stage base config."""

    suffix = "contest" if base.mode == "contest" else "single"
    system1 = (
        f"prompts/{base.domain}/{suffix}_stage1_system.txt"
        if base.domain in {"math", "abstract_reasoning"}
        else None
    )
    system2 = (
        f"prompts/{base.domain}/{suffix}_stage2_system.txt"
        if base.domain in {"math", "abstract_reasoning"}
        else None
    )
    stage1 = replace(
        base,
        name=f"{base.name}_stage1",
        stage="stage1",
        prompt=PromptConfig(
            f"prompts/{base.domain}/{suffix}_stage1.txt",
            system1,
        ),
    )
    stage2 = replace(
        base,
        name=f"{base.name}_stage2",
        stage="stage2",
        prompt=PromptConfig(
            f"prompts/{base.domain}/{suffix}_stage2.txt",
            system2,
        ),
    )
    return stage1, stage2


def build_offline_provider(
    config: ExperimentConfig,
    provider_kind: Literal["mock", "replay"],
    replay_path: str | None = None,
    *,
    mock_response: str | None = None,
) -> MockProvider | ReplayProvider:
    if provider_kind == "mock":
        return MockProvider(
            default_mock_response(config) if mock_response is None else mock_response
        )
    if replay_path is None:
        raise OfflineRunnerError("--replay-file is required with provider='replay'")
    return ReplayProvider(str(_resolve_public_path(replay_path)))


__all__ = [
    "NLRunArtifacts",
    "OfflineRunnerError",
    "PreparedNLRequest",
    "PreparedTwoStageRequest",
    "build_offline_provider",
    "default_mock_response",
    "default_stage1_mock_response",
    "derive_two_stage_configs",
    "prepare_contest_requests",
    "prepare_single_problem_requests",
    "prepare_two_stage_requests",
    "run_contest_nl",
    "run_single_problem_nl",
    "run_two_stage_nl",
    "write_run_artifacts",
]
