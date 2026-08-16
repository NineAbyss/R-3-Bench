"""Offline answer extraction for Mathematics outputs."""

from __future__ import annotations

import re


_BOX_COMMAND = re.compile(r"\\boxed\s*\{")
_FINAL_ANSWER = re.compile(
    r"(?im)^\s*(?:\*\*)?(?:final\s+answer|answer)(?:\*\*)?\s*[:=]\s*(?P<answer>.+?)\s*$"
)
_MISSING_ANSWERS = frozenset({"no answer", "missing"})


def _balanced_boxed_answers(text: str) -> list[tuple[int, str]]:
    answers: list[tuple[int, str]] = []
    for match in _BOX_COMMAND.finditer(text):
        start = match.end()
        depth = 1
        index = start
        while index < len(text) and depth:
            character = text[index]
            if character == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif character == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
            index += 1
        if depth == 0:
            answer = text[start : index - 1].strip()
            if answer:
                answers.append((match.start(), answer))
    return answers


def extract_math_answer(
    text: str,
    *,
    allow_final_answer_fallback: bool = False,
) -> str | None:
    """Extract a boxed answer, with the paper's optional contest fallback."""

    if not isinstance(text, str) or not text.strip():
        return None

    candidates = _balanced_boxed_answers(text)
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


__all__ = ["extract_math_answer"]
