"""Problem-focus state and attribution checks for the public Agentic skeleton."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

from r3bench.agentic.action_accounting import ActionClass


@dataclass(frozen=True, slots=True)
class ScopeDecision:
    """Result of checking whether an action is valid in the active scope."""

    allowed: bool
    active_problem_id: str | None
    attributed_problem_id: str | None
    reason: str


@dataclass(slots=True)
class AgenticScopeState:
    """Track the active problem without exposing any correctness information."""

    valid_problem_ids: frozenset[str]
    problem_labels: Mapping[str, str] = field(default_factory=dict)
    active_problem_id: str | None = None

    def __post_init__(self) -> None:
        if not self.valid_problem_ids:
            raise ValueError("valid_problem_ids cannot be empty")
        if any(not isinstance(value, str) or not value for value in self.valid_problem_ids):
            raise ValueError("problem IDs must be non-empty strings")
        for label, problem_id in self.problem_labels.items():
            if label not in set("ABCDEF"):
                raise ValueError(f"unsupported problem label: {label}")
            if problem_id not in self.valid_problem_ids:
                raise ValueError(f"label {label} refers to an unknown problem")

    def focus_problem(self, problem_id: str) -> None:
        """Set the current attribution target."""

        resolved = self.problem_labels.get(problem_id, problem_id)
        if resolved not in self.valid_problem_ids:
            raise ValueError(f"unknown problem_id: {problem_id}")
        self.active_problem_id = resolved

    def shelve_problem(self) -> None:
        """Clear the current attribution target."""

        self.active_problem_id = None

    def detect_cross_problem_access(self, command: str) -> str | None:
        """Detect references to another problem's scoped files or public ID."""

        if self.active_problem_id is None:
            return None
        if re.search(
            r"(?:/logs/problem_|/app/solution_)[^\s]*[?$*\[\]{}]",
            command,
        ):
            return "dynamic"
        if len(self.valid_problem_ids) > 1 and re.search(
            r"(?:^|/)logs/artifacts/answer\.txt\b", command
        ):
            return "shared"
        active_labels = {
            label
            for label, problem_id in self.problem_labels.items()
            if problem_id == self.active_problem_id
        }
        mentioned = {
            match.group(1)
            for match in re.finditer(
                r"(?:\bsolution_|(?:^|/)problem_)([A-F])(?:\.cpp\b|(?:/|\b))",
                command,
            )
        }
        for label, problem_id in self.problem_labels.items():
            if re.search(
                rf"(?<![A-Za-z0-9_.-]){re.escape(problem_id)}"
                r"(?![A-Za-z0-9_.-])",
                command,
            ):
                mentioned.add(label)
        conflicting = sorted(mentioned - active_labels)
        return conflicting[0] if conflicting else None

    def authorize_action(
        self, action_class: ActionClass, command: str
    ) -> ScopeDecision:
        """Require focus for paid actions and attribute them to one problem."""

        if action_class != ActionClass.COUNTED:
            return ScopeDecision(
                allowed=True,
                active_problem_id=self.active_problem_id,
                attributed_problem_id=None,
                reason="non_paid_action_does_not_require_focus",
            )
        if self.active_problem_id is None:
            return ScopeDecision(
                allowed=False,
                active_problem_id=None,
                attributed_problem_id=None,
                reason="counted_action_requires_active_focus",
            )
        conflict = self.detect_cross_problem_access(command)
        if conflict is not None:
            return ScopeDecision(
                allowed=False,
                active_problem_id=self.active_problem_id,
                attributed_problem_id=None,
                reason=f"cross_problem_access_blocked:{conflict}",
            )
        return ScopeDecision(
            allowed=True,
            active_problem_id=self.active_problem_id,
            attributed_problem_id=self.active_problem_id,
            reason="counted_action_attributed_to_active_problem",
        )


__all__ = ["AgenticScopeState", "ScopeDecision"]
