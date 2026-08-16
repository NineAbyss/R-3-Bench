"""Small counted-action budget state used by future agentic runtimes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ActionBudget:
    """Track unit-cost executed actions and blocked over-budget attempts."""

    limit: int | None
    used: int = 0
    blocked_attempts: int = 0

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit < 0:
            raise ValueError("limit must be non-negative or None")
        if self.used < 0 or self.blocked_attempts < 0:
            raise ValueError("budget counters must be non-negative")
        if self.limit is not None and self.used > self.limit:
            raise ValueError("used cannot exceed limit")

    @property
    def remaining(self) -> int | None:
        if self.limit is None:
            return None
        return self.limit - self.used

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.used >= self.limit

    def consume(self) -> bool:
        """Consume one executable action, or log a blocked attempt."""

        if self.exhausted:
            self.record_blocked()
            return False
        self.used += 1
        return True

    def record_blocked(self) -> None:
        """Record an attempt rejected before execution without spending budget."""

        self.blocked_attempts += 1
