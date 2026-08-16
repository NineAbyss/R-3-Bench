from __future__ import annotations

import pytest

from r3bench.coding.parser import extract_contest_cpp_sections, extract_cpp_code
from r3bench.common.nl_runner import _contest_parse, _contest_sections
from r3bench.common.schema import ProblemRecord


def _program(index: int) -> str:
    return f"#include <iostream>\nint main(){{std::cout << {index}; return 0;}}"


def _block(index: int, *, closed: bool = True) -> str:
    closing = "\n```" if closed else ""
    return f"```cpp\n{_program(index)}{closing}"


def _problem(label: str) -> ProblemRecord:
    return ProblemRecord(
        domain="coding",
        split="test",
        task_type="contest",
        problem_id=f"problem-{label}",
        suite_id="suite",
        problem_index="ABCDEF".index(label) + 1,
        problem_label=label,
        problem_statement="statement",
        difficulty="easy",
        source="synthetic",
        metadata_public={},
        domain_payload={},
    )


def test_labeled_contest_requires_one_closed_program_per_section() -> None:
    text = "\n\n".join(
        f"## Solution {label}\n{_block(index)}"
        for index, label in enumerate("ABCDEF", start=1)
    )

    sections = _contest_sections("coding", text)

    assert sections == {
        label: _program(index) for index, label in enumerate("ABCDEF", start=1)
    }
    assert _contest_parse(_problem("C"), sections) == _program(3)


def test_unlabeled_contest_falls_back_only_for_exactly_six_blocks() -> None:
    six = "\n\n".join(_block(index) for index in range(1, 7))
    assert extract_contest_cpp_sections(six) == {
        label: _program(index) for index, label in enumerate("ABCDEF", start=1)
    }

    five = "\n\n".join(_block(index) for index in range(1, 6))
    seven = "\n\n".join(_block(index) for index in range(1, 8))
    assert extract_contest_cpp_sections(five) == {}
    assert extract_contest_cpp_sections(seven) == {}


@pytest.mark.parametrize(
    "text",
    [
        f"Solution A\n{_block(1)}\nSolution A\n{_block(2)}",
        f"Solution A\n{_block(1)}\nProblem B\n{_block(2)}",
        f"Problem 1\n{_block(1)}\nProblem B\n{_block(2)}",
        f"{_block(6)}\nSolution A\n{_block(1)}",
    ],
)
def test_duplicate_mixed_or_prefixed_labels_invalidate_contest(text: str) -> None:
    assert extract_contest_cpp_sections(text) == {}


def test_multiple_blocks_make_only_that_labeled_section_missing() -> None:
    text = (
        f"Solution A\n{_block(1)}\n{_block(2)}\n\n"
        f"Solution B\n{_block(3)}"
    )

    sections = extract_contest_cpp_sections(text)

    assert sections == {"B": _program(3)}
    assert _contest_parse(_problem("A"), sections) is None
    assert _contest_parse(_problem("B"), sections) == _program(3)


def test_unclosed_fence_is_missing_and_never_consumes_later_section() -> None:
    text = f"Solution A\n{_block(1, closed=False)}\nSolution B\n{_block(2)}"

    sections = extract_contest_cpp_sections(text)

    assert "A" not in sections
    assert sections["B"] == _program(2)


def test_contest_never_uses_best_block_or_plain_main_fallback() -> None:
    plain = f"Final Code: {_program(1)}"
    assert extract_cpp_code(plain) == _program(1)
    assert extract_contest_cpp_sections(plain) == {}

    two_blocks = f"Solution A\n```cpp\nint helper();\n```\n{_block(2)}"
    assert extract_contest_cpp_sections(two_blocks) == {}


def test_unlabeled_fallback_rejects_any_unclosed_extra_fence() -> None:
    text = "\n".join(
        [*(_block(index) for index in range(1, 7)), _block(7, closed=False)]
    )
    assert extract_contest_cpp_sections(text) == {}
