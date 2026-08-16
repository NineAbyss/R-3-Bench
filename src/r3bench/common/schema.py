"""Immutable evaluator-facing data objects."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias


Domain: TypeAlias = Literal["coding", "math", "abstract_reasoning"]
TaskType: TypeAlias = Literal["single_problem", "contest"]
Difficulty: TypeAlias = Literal["easy", "medium", "hard"]

DOMAINS = frozenset({"coding", "math", "abstract_reasoning"})
TASK_TYPES = frozenset({"single_problem", "contest"})
DIFFICULTIES = frozenset({"easy", "medium", "hard"})
CONTEST_LABELS = tuple("ABCDEF")


class SchemaError(ValueError):
    """Raised when an evaluator-facing record violates the data contract."""


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON-like values without changing scalar values."""

    if isinstance(value, Mapping):
        frozen = {str(key): _freeze(item) for key, item in value.items()}
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class ProblemRecord:
    """One normalized R3Bench problem in a single or contest view."""

    domain: Domain
    split: str
    task_type: TaskType
    problem_id: str
    suite_id: str
    problem_index: int
    problem_label: str | None
    problem_statement: str
    difficulty: Difficulty
    source: str
    metadata_public: Mapping[str, Any]
    domain_payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise SchemaError(f"unsupported domain: {self.domain!r}")
        _require_nonempty_string(self.split, "split")
        if self.task_type not in TASK_TYPES:
            raise SchemaError(f"unsupported task_type: {self.task_type!r}")
        _require_nonempty_string(self.problem_id, "problem_id")
        _require_nonempty_string(self.suite_id, "suite_id")
        if isinstance(self.problem_index, bool) or not isinstance(self.problem_index, int):
            raise SchemaError("problem_index must be an integer")
        if not 1 <= self.problem_index <= 6:
            raise SchemaError("problem_index must be between 1 and 6")
        _require_nonempty_string(self.problem_statement, "problem_statement")
        if self.difficulty not in DIFFICULTIES:
            raise SchemaError(f"unsupported difficulty: {self.difficulty!r}")
        _require_nonempty_string(self.source, "source")

        if self.task_type == "single_problem":
            if self.problem_label is not None:
                raise SchemaError("single-problem records must not have a visible label")
        elif self.problem_label not in CONTEST_LABELS:
            raise SchemaError("contest records must have a visible label from A through F")

        if not isinstance(self.metadata_public, Mapping):
            raise SchemaError("metadata_public must be a mapping")
        if not isinstance(self.domain_payload, Mapping):
            raise SchemaError("domain_payload must be a mapping")
        object.__setattr__(self, "metadata_public", _freeze(self.metadata_public))
        object.__setattr__(self, "domain_payload", _freeze(self.domain_payload))


@dataclass(frozen=True, slots=True)
class ContestSuite:
    """A normalized six-problem contest in presented order."""

    domain: Domain
    split: str
    suite_id: str
    problems: tuple[ProblemRecord, ...]

    def __post_init__(self) -> None:
        if self.domain not in DOMAINS:
            raise SchemaError(f"unsupported domain: {self.domain!r}")
        _require_nonempty_string(self.split, "split")
        _require_nonempty_string(self.suite_id, "suite_id")
        problems = tuple(self.problems)
        object.__setattr__(self, "problems", problems)
        if len(problems) != 6:
            raise SchemaError("a contest suite must contain exactly six problems")
        if len({problem.problem_id for problem in problems}) != 6:
            raise SchemaError("a contest suite must contain six distinct problem IDs")
        for problem in problems:
            if problem.domain != self.domain:
                raise SchemaError("problem domain does not match suite domain")
            if problem.split != self.split:
                raise SchemaError("problem split does not match suite split")
            if problem.suite_id != self.suite_id:
                raise SchemaError("problem suite_id does not match parent suite")
            if problem.task_type != "contest":
                raise SchemaError("ContestSuite can contain only contest-view records")
        if tuple(problem.problem_index for problem in problems) != (1, 2, 3, 4, 5, 6):
            raise SchemaError("contest problems must be ordered at indices 1 through 6")
        if tuple(problem.problem_label for problem in problems) != CONTEST_LABELS:
            raise SchemaError("contest labels must be A through F in order")
