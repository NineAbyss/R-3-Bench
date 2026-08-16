"""Provider-neutral parsing helpers for Coding model outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass


_FENCED_BLOCK = re.compile(
    r"```[ \t]*(?P<language>[^\n`]*)\r?\n(?P<code>.*?)```",
    re.DOTALL,
)
_CPP_LANGUAGES = frozenset({"", "cpp", "c++", "cc", "cxx"})
_MAIN_FUNCTION = re.compile(
    r"\b(?:int|signed|auto)\s+main\s*\([^)]*\)\s*(?:->\s*int\s*)?\{",
    re.DOTALL,
)
_PLAIN_FINAL_PREFIX = re.compile(
    r"^\s*(?:final\s+(?:answer|code)|answer)\s*:\s*",
    re.IGNORECASE,
)
_FENCE_LINE = re.compile(r"(?m)^[ \t]*```(?P<info>[^`\r\n]*)[ \t]*\r?$")
_CONTEST_LABEL = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?"
    r"(?P<kind>solution|problem|answer)\s+"
    r"(?P<label>[1-6A-F])\s*:?\s*$"
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    code: str
    language: str
    order: int


def _candidate_score(candidate: _Candidate) -> int | None:
    code = candidate.code
    if not code.strip() or not _MAIN_FUNCTION.search(code):
        return None
    if "{" not in code or "}" not in code:
        return None

    score = 10
    if candidate.language in {"cpp", "c++", "cc", "cxx"}:
        score += 4
    if re.search(r"(?m)^\s*#\s*include\b", code):
        score += 4
    if "std::" in code or re.search(r"\busing\s+namespace\s+std\s*;", code):
        score += 2
    if re.search(r"\b(?:cin|cout|scanf|printf)\b", code):
        score += 1
    if ";" in code:
        score += 1
    return score


def _best_candidate(candidates: list[_Candidate]) -> str | None:
    ranked: list[tuple[int, int, str]] = []
    for candidate in candidates:
        score = _candidate_score(candidate)
        if score is not None:
            ranked.append((score, candidate.order, candidate.code.strip()))
    if not ranked:
        return None
    # Prefer the strongest candidate; when equally plausible, use the last one.
    return max(ranked, key=lambda item: (item[0], item[1]))[2]


def extract_cpp_code(text: str) -> str | None:
    """Extract the most plausible complete C++ translation unit.

    C++-tagged and untagged fenced blocks are considered. If no usable fence is
    present, a plain final answer is accepted only when it contains a complete,
    code-like ``main`` function. Prose and empty answers are rejected.
    """

    if not isinstance(text, str) or not text.strip():
        return None

    fence_matches = list(_FENCED_BLOCK.finditer(text))
    candidates: list[_Candidate] = []
    for order, match in enumerate(fence_matches):
        language = match.group("language").strip().lower()
        if language in _CPP_LANGUAGES:
            candidates.append(
                _Candidate(match.group("code"), language=language, order=order)
            )
    fenced = _best_candidate(candidates)
    if fence_matches:
        return fenced

    plain = _PLAIN_FINAL_PREFIX.sub("", text, count=1).strip()
    return _best_candidate([_Candidate(plain, language="", order=0)])


def _strict_fenced_cpp_blocks(text: str) -> tuple[str, ...] | None:
    """Return all complete C++ blocks, or ``None`` for malformed fences/code."""

    markers = list(_FENCE_LINE.finditer(text))
    if text.count("```") != len(markers) or len(markers) % 2:
        return None
    blocks: list[str] = []
    for index in range(0, len(markers), 2):
        opening = markers[index]
        closing = markers[index + 1]
        language = opening.group("info").strip().lower()
        if closing.group("info").strip() or language not in _CPP_LANGUAGES:
            return None
        code = text[opening.end() : closing.start()].strip()
        candidate = _Candidate(code=code, language=language, order=index // 2)
        if _candidate_score(candidate) is None:
            return None
        blocks.append(code)
    return tuple(blocks)


def extract_contest_cpp_sections(text: str) -> dict[str, str]:
    """Parse Appendix-D Coding contest output without heuristic fallbacks.

    Labeled output is split by one consistent A-F heading style and each
    scoreable section must contain exactly one complete fenced program. With no
    labels, exactly six complete fenced programs are assigned to A-F in order.
    """

    if not isinstance(text, str) or not text.strip():
        return {}
    matches = list(_CONTEST_LABEL.finditer(text))
    if not matches:
        blocks = _strict_fenced_cpp_blocks(text)
        return dict(zip("ABCDEF", blocks, strict=True)) if len(blocks or ()) == 6 else {}

    labels = [match.group("label").upper() for match in matches]
    heading_kinds = {match.group("kind").lower() for match in matches}
    if (
        any(label not in "ABCDEF" for label in labels)
        or len(labels) != len(set(labels))
        or len(heading_kinds) != 1
    ):
        return {}
    if "```" in text[: matches[0].start()]:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks = _strict_fenced_cpp_blocks(text[match.end() : end])
        if blocks is not None and len(blocks) == 1:
            sections[labels[index]] = blocks[0]
    return sections


__all__ = ["extract_contest_cpp_sections", "extract_cpp_code"]
