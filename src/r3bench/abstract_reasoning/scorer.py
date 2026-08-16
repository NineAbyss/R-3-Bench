"""Abstract Reasoning scorer contracts and optional generator-aware adapter."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from r3bench.common.schema import ProblemRecord


@dataclass(frozen=True, slots=True)
class ARScorerConfig:
    reasoning_gym_version: str = "0.1.25"
    reasoning_gym_revision: str = "21e6d2a9a581b3e11aafe711abfd37402f8482d5"
    module_name: str = "reasoning_gym"
    case_sensitive: bool = True


@dataclass(frozen=True, slots=True)
class ReasoningGymScorerConfig:
    reasoning_gym_version: str = "0.1.25"
    reasoning_gym_revision: str | None = "21e6d2a9a581b3e11aafe711abfd37402f8482d5"
    module_name: str = "reasoning_gym"
    enforce_installed_version: bool = True


@dataclass(frozen=True, slots=True)
class ARScorerResult:
    problem_id: str
    correct: bool
    score: float
    verdict: str
    detail: str | None = None


class ARScorer(Protocol):
    def score(self, problem: ProblemRecord, candidate_answer: str) -> ARScorerResult:
        ...


class GeneratorAwareScoringCallable(Protocol):
    def __call__(
        self,
        *,
        generator: str,
        scorer_metadata: Mapping[str, Any],
        candidate_answer: str,
        reference_answer: str,
    ) -> float | bool:
        ...


def _reference_answer(problem: ProblemRecord) -> str:
    if problem.domain != "abstract_reasoning":
        raise ValueError("ARScorer requires an Abstract Reasoning ProblemRecord")
    reference = problem.domain_payload.get("answer")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError(f"AR problem {problem.problem_id!r} has no reference answer")
    return reference


def _scoring_context(problem: ProblemRecord) -> tuple[str, Mapping[str, Any]]:
    if problem.domain != "abstract_reasoning":
        raise ValueError("ReasoningGymScorerAdapter requires an AR ProblemRecord")
    generator = problem.domain_payload.get("generator")
    metadata = problem.domain_payload.get("scorer_metadata")
    if not isinstance(generator, str) or not generator.strip():
        raise ValueError("generator-aware AR scoring requires a public generator name")
    if not isinstance(metadata, Mapping) or not metadata:
        raise ValueError("generator-aware AR scoring requires public scorer_metadata")
    return generator, metadata


class MockARScorer:
    """Exact-match scorer for interface tests only, never official AR scoring."""

    def __init__(self, config: ARScorerConfig | None = None) -> None:
        self.config = config or ARScorerConfig()

    def score(self, problem: ProblemRecord, candidate_answer: str) -> ARScorerResult:
        reference = _reference_answer(problem).strip()
        candidate = candidate_answer.strip() if isinstance(candidate_answer, str) else ""
        if not self.config.case_sensitive:
            reference = reference.casefold()
            candidate = candidate.casefold()
        correct = bool(candidate) and candidate == reference
        return ARScorerResult(
            problem_id=problem.problem_id,
            correct=correct,
            score=1.0 if correct else 0.0,
            verdict="correct" if correct else "incorrect",
            detail="mock exact-match comparison; Reasoning Gym was not invoked",
        )


class ReasoningGymScorerAdapter:
    """Optional dependency boundary for generator-aware production scoring."""

    def __init__(
        self,
        config: ReasoningGymScorerConfig | None = None,
        *,
        scoring_callable: GeneratorAwareScoringCallable | None = None,
        reasoning_gym_module: object | None = None,
    ) -> None:
        self.config = config or ReasoningGymScorerConfig()
        self._scoring_callable = scoring_callable
        self._module = reasoning_gym_module
        self._dataset_cache: dict[str, object] = {}

    def _load_dependency(self) -> object:
        if self._module is not None:
            return self._module
        try:
            module = importlib.import_module(self.config.module_name)
        except ImportError as exc:
            raise RuntimeError(
                "ReasoningGymScorerAdapter requires the optional 'reasoning_gym' "
                f"package at version {self.config.reasoning_gym_version}. Install "
                "the pinned external dependency before official AR scoring."
            ) from exc
        if self.config.enforce_installed_version:
            try:
                installed = importlib.metadata.version("reasoning-gym")
            except importlib.metadata.PackageNotFoundError as exc:
                raise RuntimeError(
                    "Reasoning Gym is importable but its installed distribution version "
                    "cannot be verified"
                ) from exc
            if installed != self.config.reasoning_gym_version:
                raise RuntimeError(
                    "Reasoning Gym version mismatch: expected "
                    f"{self.config.reasoning_gym_version}, found {installed}"
                )
        self._module = module
        return module

    @staticmethod
    def _formal_context(
        problem: ProblemRecord,
        generator: str,
        scorer_metadata: Mapping[str, Any],
        reference: str,
    ) -> tuple[str, Mapping[str, Any], dict[str, Any]]:
        dataset_params = scorer_metadata.get("dataset_params") or {}
        generator_metadata = scorer_metadata.get("generator_metadata") or {}
        if not isinstance(dataset_params, Mapping):
            raise ValueError("AR scorer dataset_params must be an object")
        if not isinstance(generator_metadata, Mapping):
            raise ValueError("AR scorer generator_metadata must be an object")
        cache_key = json.dumps(
            [generator, dict(dataset_params)],
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        entry = {
            "question": problem.problem_statement,
            "answer": (
                None
                if problem.domain_payload.get("dynamic_scorer") is True
                else reference
            ),
            "metadata": dict(generator_metadata),
        }
        return cache_key, dataset_params, entry

    def _score_with_formal_bridge(
        self,
        module: object,
        problem: ProblemRecord,
        generator: str,
        scorer_metadata: Mapping[str, Any],
        candidate_answer: str,
        reference: str,
    ) -> float:
        create_dataset = getattr(module, "create_dataset", None)
        if not callable(create_dataset):
            raise RuntimeError("Reasoning Gym module has no callable create_dataset")
        cache_key, dataset_params, entry = self._formal_context(
            problem, generator, scorer_metadata, reference
        )
        if cache_key not in self._dataset_cache:
            self._dataset_cache[cache_key] = create_dataset(
                generator, **dict(dataset_params)
            )
        score_answer = getattr(self._dataset_cache[cache_key], "score_answer", None)
        if not callable(score_answer):
            raise RuntimeError("Reasoning Gym dataset has no callable score_answer")
        return float(score_answer(answer=candidate_answer, entry=entry))

    def score(self, problem: ProblemRecord, candidate_answer: str) -> ARScorerResult:
        reference = _reference_answer(problem)
        if not isinstance(candidate_answer, str) or not candidate_answer.strip():
            raise ValueError("candidate answer must be non-empty")
        generator, metadata = _scoring_context(problem)
        module = self._load_dependency()
        if self._scoring_callable is not None:
            raw_score = self._scoring_callable(
                generator=generator,
                scorer_metadata=metadata,
                candidate_answer=candidate_answer,
                reference_answer=reference,
            )
        else:
            raw_score = self._score_with_formal_bridge(
                module,
                problem,
                generator,
                metadata,
                candidate_answer,
                reference,
            )
        score = float(raw_score)
        if not 0.0 <= score <= 1.0:
            raise RuntimeError("Reasoning Gym scorer returned a score outside [0, 1]")
        correct = score == 1.0
        return ARScorerResult(
            problem_id=problem.problem_id,
            correct=correct,
            score=score,
            verdict="correct" if correct else "incorrect",
            detail="generator-aware Reasoning Gym score",
        )


__all__ = [
    "ARScorer",
    "ARScorerConfig",
    "ARScorerResult",
    "GeneratorAwareScoringCallable",
    "MockARScorer",
    "ReasoningGymScorerAdapter",
    "ReasoningGymScorerConfig",
]
