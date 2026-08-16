"""Fixed paper-reference runtime limits shared by task and backend contracts."""

from __future__ import annotations

from typing import Any, Mapping


PAPER_SANDBOX_LIMITS: dict[str, dict[str, int]] = {
    "coding": {
        "agent_timeout_seconds": 7200,
        "build_timeout_seconds": 600,
        "memory_mb": 2048,
        "storage_mb": 10240,
    },
    "math": {
        "agent_timeout_seconds": 7200,
        "verifier_timeout_seconds": 7200,
        "cpu_count": 1,
        "memory_mb": 2048,
        "storage_mb": 10240,
    },
    "abstract_reasoning": {
        "agent_timeout_seconds": 7200,
        "verifier_timeout_seconds": 7200,
        "cpu_count": 1,
        "memory_mb": 2048,
        "storage_mb": 10240,
    },
}


def paper_sandbox_limits(domain: str) -> dict[str, int]:
    """Return an independent copy of the fixed limits for one domain."""

    try:
        return dict(PAPER_SANDBOX_LIMITS[domain])
    except KeyError as exc:
        raise ValueError(f"unsupported Agentic domain: {domain!r}") from exc


def has_exact_paper_sandbox_limits(value: Any) -> bool:
    """Check all domains and reject missing or additional limit fields."""

    if not isinstance(value, Mapping) or set(value) != set(PAPER_SANDBOX_LIMITS):
        return False
    return all(
        isinstance(value.get(domain), Mapping)
        and dict(value[domain]) == expected
        for domain, expected in PAPER_SANDBOX_LIMITS.items()
    )


__all__ = [
    "PAPER_SANDBOX_LIMITS",
    "has_exact_paper_sandbox_limits",
    "paper_sandbox_limits",
]
