"""Offline answer extraction for Abstract Reasoning outputs."""

from __future__ import annotations

import re


_ANSWER_TAG = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.IGNORECASE | re.DOTALL)
_FINAL_ANSWER = re.compile(
    r"(?im)^\s*(?:\*\*)?(?:final\s+answer|answer)(?:\*\*)?\s*[:=]\s*(?P<answer>.+?)\s*$"
)
_MISSING_ANSWERS = frozenset({"no answer", "missing"})


def extract_ar_answer(
    text: str,
    *,
    allow_final_answer_fallback: bool = False,
) -> str | None:
    """Extract a tagged answer, with the paper's optional contest fallback."""

    if not isinstance(text, str) or not text.strip():
        return None
    candidates = [
        (match.start(), match.group("answer").strip())
        for match in _ANSWER_TAG.finditer(text)
        if match.group("answer").strip()
    ]
    if allow_final_answer_fallback:
        candidates.extend(
            (match.start(), match.group("answer").strip())
            for match in _FINAL_ANSWER.finditer(text)
            if match.group("answer").strip()
        )
    if candidates:
        answer = max(candidates, key=lambda item: item[0])[1]
        return None if answer.casefold() in _MISSING_ANSWERS else answer
    return None


__all__ = ["extract_ar_answer"]
