"""Provider-neutral Mathematics judge interfaces and opt-in adapter boundary.

The production parser supports the Omni-MATH GPT-eval report used by the
formal runs. Transport remains an explicit post-generation dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Mapping, Protocol

from r3bench.common.provider import ModelRequest, Provider
from r3bench.common.schema import ProblemRecord


@dataclass(frozen=True, slots=True)
class MathJudgeConfig:
    """Offline mock-judge policy kept outside public problem data."""

    judge_model: str | None = None
    prompt_template: str | None = None
    normalize_whitespace: bool = True
    case_sensitive: bool = True


@dataclass(frozen=True, slots=True)
class MathEquivalenceJudgeConfig:
    """Runtime-only production judge configuration."""

    judge_model: str = "unresolved"
    prompt_template: str | None = None
    api_key_env: str | None = None
    response_format: str = "omnimath_markdown"


@dataclass(frozen=True, slots=True)
class MathJudgeResult:
    problem_id: str
    correct: bool
    verdict: str
    detail: str | None = None


class MathJudge(Protocol):
    def judge(self, problem: ProblemRecord, candidate_answer: str) -> MathJudgeResult:
        ...


class MathJudgeTransport(Protocol):
    """Minimal injected transport; it exposes no provider response metadata."""

    def generate(self, prompt: str, *, model: str, api_key: str) -> str:
        ...


class ProviderMathJudgeTransport:
    """Bind the Math judge to the shared provider interface.

    The supplied provider may be a real opt-in adapter, MockProvider, or
    ReplayProvider. The credential argument is never retained or serialized;
    real provider adapters read the same environment variable at execution.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        max_tokens: int = 16384,
        temperature: float = 0.0,
    ) -> None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self._provider = provider
        self._max_tokens = max_tokens
        self._temperature = float(temperature)

    def generate(self, prompt: str, *, model: str, api_key: str) -> str:
        if not api_key:
            raise RuntimeError("Math judge credential is not available")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:20]
        response = self._provider.complete(
            ModelRequest(
                request_id=f"math-judge-{digest}",
                model=model,
                prompt_text=prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                metadata={"purpose": "math_equivalence_judge"},
            )
        )
        return response.response_text


def _reference_answer(problem: ProblemRecord) -> str:
    if problem.domain != "math":
        raise ValueError("MathJudge requires a Math ProblemRecord")
    reference = problem.domain_payload.get("answer")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"Math problem {problem.problem_id!r} has no reference answer")
    return reference


def build_judge_prompt(
    problem: ProblemRecord,
    candidate_answer: str,
    reference_answer: str,
    *,
    template: str,
) -> str:
    """Render a fixed equivalence prompt without invoking a judge model."""

    if problem.domain != "math":
        raise ValueError("equivalence judge requires a Math ProblemRecord")
    if not isinstance(candidate_answer, str) or not candidate_answer.strip():
        raise ValueError("candidate answer must be non-empty")
    if not isinstance(reference_answer, str) or not reference_answer.strip():
        raise ValueError("reference answer must be non-empty")
    compact = ("{problem_statement}", "{reference_answer}", "{candidate_answer}")
    formal = ("{{Problem}}", "{{Reference Answer}}", "{{Solution}}")
    if not isinstance(template, str) or not (
        all(token in template for token in compact)
        or all(token in template for token in formal)
    ):
        raise ValueError("judge prompt template is missing required placeholders")
    return (
        template.replace("{{Problem}}", problem.problem_statement)
        .replace("{{Reference Answer}}", reference_answer)
        .replace("{{Solution}}", candidate_answer)
        .replace("{problem_statement}", problem.problem_statement)
        .replace("{reference_answer}", reference_answer)
        .replace("{candidate_answer}", candidate_answer)
    )


def parse_judge_response(response_text: str) -> tuple[bool, str | None]:
    """Parse JSON compatibility responses or the formal Omni-MATH report."""

    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("math judge returned an empty response")
    try:
        value = json.loads(response_text)
    except json.JSONDecodeError:
        value = None
    if value is not None:
        if not isinstance(value, Mapping) or not isinstance(value.get("equivalent"), bool):
            raise ValueError("math judge response must contain boolean 'equivalent'")
        if set(value) - {"equivalent", "reason"}:
            raise ValueError("math judge response contains unsupported fields")
        reason = value.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("math judge reason must be a string or null")
        return value["equivalent"], reason

    match = re.search(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?Equivalence Judg(?:e)?ment\s*"
        r"(?:\n|[:：])\s*[*_`-]*\s*(TRUE|FALSE)\b",
        response_text,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError(
            "math judge response is neither valid JSON nor an Omni-MATH "
            "Equivalence Judgement report"
        )
    justification_match = re.search(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?Justification\s*(?:\n|[:：])\s*"
        r"(.*?)(?:\n\s*===\s*report over\s*===|\Z)",
        response_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    reason = justification_match.group(1).strip() if justification_match else None
    return match.group(1).upper() == "TRUE", reason or None


class MockMathJudge:
    """Deterministic normalized-string judge for tests, not official scoring."""

    def __init__(self, config: MathJudgeConfig | None = None) -> None:
        self.config = config or MathJudgeConfig()

    def _normalize(self, value: str) -> str:
        normalized = value.strip()
        if self.config.normalize_whitespace:
            normalized = re.sub(r"\s+", " ", normalized)
        if not self.config.case_sensitive:
            normalized = normalized.casefold()
        return normalized

    def judge(self, problem: ProblemRecord, candidate_answer: str) -> MathJudgeResult:
        reference = _reference_answer(problem)
        if not isinstance(candidate_answer, str) or not candidate_answer.strip():
            return MathJudgeResult(
                problem_id=problem.problem_id,
                correct=False,
                verdict="incorrect",
                detail="mock judge received an empty candidate",
            )
        correct = self._normalize(reference) == self._normalize(candidate_answer)
        return MathJudgeResult(
            problem_id=problem.problem_id,
            correct=correct,
            verdict="correct" if correct else "incorrect",
            detail="mock normalized-string comparison; no judge API was called",
        )


class MathEquivalenceJudgeAdapter:
    """Opt-in model-judge adapter; no real transport is provided by default."""

    def __init__(
        self,
        config: MathEquivalenceJudgeConfig,
        *,
        transport: MathJudgeTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport

    def build_judge_prompt(
        self,
        problem: ProblemRecord,
        parsed_answer: str,
        reference_answer: str | None = None,
    ) -> str:
        reference = _reference_answer(problem) if reference_answer is None else reference_answer
        if not self.config.prompt_template:
            raise RuntimeError("Math equivalence judge prompt is not configured")
        return build_judge_prompt(
            problem,
            parsed_answer,
            reference,
            template=self.config.prompt_template,
        )

    @staticmethod
    def parse_judge_response(response_text: str) -> tuple[bool, str | None]:
        return parse_judge_response(response_text)

    def judge(self, problem: ProblemRecord, candidate_answer: str) -> MathJudgeResult:
        reference = _reference_answer(problem)
        if self.config.judge_model == "unresolved" or not self.config.judge_model:
            raise RuntimeError("Math equivalence judge model is unresolved")
        if not self.config.api_key_env:
            raise RuntimeError("Math equivalence judge api_key_env is not configured")
        if self.config.response_format not in {"omnimath_markdown", "json_compat"}:
            raise RuntimeError("Math equivalence judge response_format is unsupported")
        if self._transport is None:
            raise RuntimeError(
                "No Math judge transport is bound; real judge execution is opt-in only"
            )
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError("Math judge credential is not available")
        prompt = self.build_judge_prompt(problem, candidate_answer, reference)
        response = self._transport.generate(
            prompt, model=self.config.judge_model, api_key=api_key
        )
        equivalent, _ = self.parse_judge_response(response)
        return MathJudgeResult(
            problem_id=problem.problem_id,
            correct=equivalent,
            verdict="correct" if equivalent else "incorrect",
            detail="model equivalence judgment; provider metadata omitted",
        )


__all__ = [
    "MathEquivalenceJudgeAdapter",
    "MathEquivalenceJudgeConfig",
    "MathJudge",
    "MathJudgeConfig",
    "MathJudgeResult",
    "MathJudgeTransport",
    "MockMathJudge",
    "ProviderMathJudgeTransport",
    "build_judge_prompt",
    "parse_judge_response",
]
