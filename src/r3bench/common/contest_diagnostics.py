"""Protocol-level capacity diagnostics for saved contest attempts."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from r3bench.common.provider import UsageInfo


_LETTER_SECTION = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:solution|problem|answer)\s+([A-F])\s*:?\s*$"
)
_NUMBER_SECTION = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?problem\s+([1-6])\s*:?\s*$"
)
_LETTER_TAG = re.compile(
    r"<answer\s+([A-F])\s*>.*?</answer\s+\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_PLAIN_TAG = re.compile(r"<answer\s*>.*?</answer\s*>", re.IGNORECASE | re.DOTALL)
_ANY_PROBLEM_HEADER = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:solution|problem|answer)\s+([^\s:]+)"
)


@dataclass(frozen=True, slots=True)
class ContestCapacityReport:
    parsed_answer_count: int
    observed_labels: tuple[str, ...]
    missing_labels: tuple[str, ...]
    duplicate_labels: tuple[str, ...]
    malformed_labels: tuple[str, ...]
    finish_reason: str | None
    configured_max_tokens: int
    output_cap_reached: bool | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    prompt_characters: int
    response_characters: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_contest_output(
    *,
    domain: str,
    prompt_text: str,
    response_text: str,
    max_tokens: int,
    finish_reason: str | None,
    usage: UsageInfo | None,
) -> ContestCapacityReport:
    """Report format/capacity facts without estimating unavailable token counts."""

    if domain == "coding":
        raw = [match.group(1).upper() for match in _LETTER_SECTION.finditer(response_text)]
    else:
        tagged = [match.group(1).upper() for match in _LETTER_TAG.finditer(response_text)]
        if tagged:
            raw = tagged
        else:
            numbered = [match.group(1) for match in _NUMBER_SECTION.finditer(response_text)]
            raw = [chr(ord("A") + int(number) - 1) for number in numbered]
            if numbered and len(_PLAIN_TAG.findall(response_text)) < len(numbered):
                raw = raw[: len(_PLAIN_TAG.findall(response_text))]
    observed = tuple(dict.fromkeys(raw))
    duplicates = tuple(sorted(label for label in set(raw) if raw.count(label) > 1))
    missing = tuple(label for label in "ABCDEF" if label not in observed)
    malformed = tuple(
        sorted(
            {
                match.group(1)
                for match in _ANY_PROBLEM_HEADER.finditer(response_text)
                if match.group(1).upper() not in set("ABCDEF123456")
            }
        )
    )
    if usage is None or usage.total_tokens == 0:
        input_tokens = output_tokens = reasoning_tokens = None
        cap_reached = finish_reason in {"length", "max_tokens"} or None
    else:
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        reasoning_tokens = usage.reasoning_tokens
        completion = usage.output_tokens + usage.reasoning_tokens
        cap_reached = finish_reason in {"length", "max_tokens"} or (
            max_tokens > 0 and completion >= int(max_tokens * 0.98)
        )
    return ContestCapacityReport(
        parsed_answer_count=len(observed),
        observed_labels=observed,
        missing_labels=missing,
        duplicate_labels=duplicates,
        malformed_labels=malformed,
        finish_reason=finish_reason,
        configured_max_tokens=max_tokens,
        output_cap_reached=cap_reached,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        prompt_characters=len(prompt_text),
        response_characters=len(response_text),
    )


def analyze_attempt_row(row: Mapping[str, Any]) -> ContestCapacityReport:
    usage_value = row.get("usage")
    usage = None
    if isinstance(usage_value, Mapping):
        usage = UsageInfo(
            input_tokens=int(usage_value.get("input_tokens", 0)),
            output_tokens=int(usage_value.get("output_tokens", 0)),
            reasoning_tokens=int(usage_value.get("reasoning_tokens", 0)),
        )
    return analyze_contest_output(
        domain=str(row["domain"]),
        prompt_text=str(row.get("prompt_text", "")),
        response_text=str(row.get("response_text", "")),
        max_tokens=int(row.get("max_tokens", 0) or 0),
        finish_reason=(
            str(row["finish_reason"]) if row.get("finish_reason") is not None else None
        ),
        usage=usage,
    )


__all__ = [
    "ContestCapacityReport",
    "analyze_attempt_row",
    "analyze_contest_output",
]
