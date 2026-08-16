"""Explicit, auditable contest presentation transforms for Pure-NL runs."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

from r3bench.common.schema import CONTEST_LABELS, ContestSuite


PresentationOrder: TypeAlias = Literal["canonical", "formal_seeded"]


class PresentationError(ValueError):
    """Raised when a contest presentation profile is invalid."""


@dataclass(frozen=True, slots=True)
class PresentationOrderRecord:
    """Public provenance for one suite's rendered problem order."""

    suite_id: str
    seed_suite_id: str
    suite_index: int
    presentation_order: PresentationOrder
    presentation_seed: int | None
    presented_problem_ids: tuple[str, ...]
    canonical_positions: tuple[int, ...]
    permutation_sha256: str

    def __post_init__(self) -> None:
        if not self.suite_id or not self.seed_suite_id:
            raise PresentationError("suite IDs must be non-empty")
        if (
            isinstance(self.suite_index, bool)
            or not isinstance(self.suite_index, int)
            or self.suite_index < 0
        ):
            raise PresentationError("suite_index must be a non-negative integer")
        if self.presentation_order not in {"canonical", "formal_seeded"}:
            raise PresentationError("unsupported presentation_order")
        if self.presentation_order == "canonical":
            if self.presentation_seed is not None:
                raise PresentationError("canonical presentation cannot have a seed")
        elif (
            isinstance(self.presentation_seed, bool)
            or not isinstance(self.presentation_seed, int)
            or self.presentation_seed < 0
        ):
            raise PresentationError(
                "formal_seeded presentation requires a non-negative integer seed"
            )
        if len(self.presented_problem_ids) != 6:
            raise PresentationError("presentation must contain six problem IDs")
        if len(set(self.presented_problem_ids)) != 6:
            raise PresentationError("presentation problem IDs must be distinct")
        if sorted(self.canonical_positions) != [1, 2, 3, 4, 5, 6]:
            raise PresentationError(
                "canonical_positions must be a permutation of 1 through 6"
            )
        expected_hash = _permutation_sha256(
            self.presented_problem_ids, self.canonical_positions
        )
        if self.permutation_sha256 != expected_hash:
            raise PresentationError("permutation_sha256 does not match the mapping")


def _permutation_sha256(
    problem_ids: tuple[str, ...], canonical_positions: tuple[int, ...]
) -> str:
    payload = {
        "presented_problem_ids": list(problem_ids),
        "canonical_positions": list(canonical_positions),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _seed_suite_id(
    suite: ContestSuite,
    *,
    suite_index: int,
    seed_suite_id_template: str | None,
) -> str:
    if seed_suite_id_template is None:
        return suite.suite_id
    try:
        rendered = seed_suite_id_template.format(
            suite_index=suite_index,
            suite_number=suite_index + 1,
        )
    except (IndexError, KeyError, ValueError) as exc:
        raise PresentationError(
            "seed_suite_id_template must use only suite_index or suite_number"
        ) from exc
    if not rendered:
        raise PresentationError("seed_suite_id_template rendered an empty suite ID")
    return rendered


def present_contest_suite(
    suite: ContestSuite,
    *,
    order: PresentationOrder,
    seed: int | None,
    suite_index: int,
    seed_suite_id_template: str | None = None,
) -> tuple[ContestSuite, PresentationOrderRecord]:
    """Return a presentation view without mutating the canonical loader output.

    ``formal_seeded`` exactly matches the historical runner's algorithm:
    ``random.Random(f"{seed}:{suite_id}:{suite_index}").shuffle(problems)``.
    """

    if order not in {"canonical", "formal_seeded"}:
        raise PresentationError(f"unsupported presentation order: {order!r}")
    if isinstance(suite_index, bool) or not isinstance(suite_index, int) or suite_index < 0:
        raise PresentationError("suite_index must be a non-negative integer")
    if order == "canonical":
        if seed is not None or seed_suite_id_template is not None:
            raise PresentationError(
                "canonical presentation cannot have a seed or seed suite-ID template"
            )
        seed_suite_id = suite.suite_id
        selected = list(suite.problems)
    else:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise PresentationError(
                "formal_seeded presentation requires a non-negative integer seed"
            )
        seed_suite_id = _seed_suite_id(
            suite,
            suite_index=suite_index,
            seed_suite_id_template=seed_suite_id_template,
        )
        selected = list(suite.problems)
        random.Random(f"{seed}:{seed_suite_id}:{suite_index}").shuffle(selected)

    canonical_positions = tuple(problem.problem_index for problem in selected)
    presented = tuple(
        replace(
            problem,
            problem_index=presented_index,
            problem_label=CONTEST_LABELS[presented_index - 1],
        )
        for presented_index, problem in enumerate(selected, start=1)
    )
    presented_ids = tuple(problem.problem_id for problem in presented)
    record = PresentationOrderRecord(
        suite_id=suite.suite_id,
        seed_suite_id=seed_suite_id,
        suite_index=suite_index,
        presentation_order=order,
        presentation_seed=seed,
        presented_problem_ids=presented_ids,
        canonical_positions=canonical_positions,
        permutation_sha256=_permutation_sha256(
            presented_ids, canonical_positions
        ),
    )
    return (
        ContestSuite(
            domain=suite.domain,
            split=suite.split,
            suite_id=suite.suite_id,
            problems=presented,
        ),
        record,
    )


__all__ = [
    "PresentationError",
    "PresentationOrder",
    "PresentationOrderRecord",
    "present_contest_suite",
]
