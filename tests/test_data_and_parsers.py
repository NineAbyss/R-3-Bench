from __future__ import annotations

import json
from pathlib import Path

import pytest

from r3bench.abstract_reasoning.parser import extract_ar_answer
from r3bench.cli import main as cli_main
from r3bench.coding.parser import extract_cpp_code
from r3bench.common.loader import DataContractError, load_contest_suites, load_single_problems
from r3bench.math.parser import extract_math_answer


@pytest.mark.parametrize(
    ("domain", "relative"),
    [
        ("coding", ("examples", "data", "coding.jsonl")),
        ("math", ("examples", "data", "math")),
        (
            "abstract_reasoning",
            ("examples", "data", "abstract_reasoning.jsonl"),
        ),
    ],
)
def test_bundled_toy_data_supports_both_loader_views(
    resources: Path, domain: str, relative: tuple[str, ...]
) -> None:
    source = resources.joinpath(*relative)
    problems = load_single_problems(domain, "test", source, strict=False)
    suites = load_contest_suites(domain, "test", source, strict=False)
    assert len(problems) == 6
    assert len(suites) == 1
    assert [problem.problem_index for problem in suites[0].problems] == list(
        range(1, 7)
    )
    assert [problem.problem_label for problem in suites[0].problems] == list(
        "ABCDEF"
    )


def test_strict_loader_rejects_six_row_toy_as_noncanonical(resources: Path) -> None:
    with pytest.raises(DataContractError):
        load_single_problems(
            "coding",
            "test",
            resources / "examples/data/coding.jsonl",
            strict=True,
        )


def test_missing_required_field_reports_required_semantics(
    tmp_path: Path, resources: Path
) -> None:
    row = json.loads(
        (resources / "examples/data/coding.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    del row["problem_id"]
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DataContractError) as error:
        load_single_problems("coding", "test", path, strict=False)
    assert "required field" in str(error.value)
    assert "problem_id" in str(error.value)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Final answer: \\\\boxed{42}", "42"),
        (r"Final answer: \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        ("<answer>x^2</answer>", None),
        ("Final answer: x^2", None),
        (r"Final answer: \fbox{42}", None),
        (r"Final answer: \boxed{No answer}", None),
        (r"Final answer: \boxed{MISSING}", None),
        ("one line", None),
        ("analysis\nwithout explicit answer", None),
    ],
)
def test_math_parser(text: str, expected: str | None) -> None:
    assert extract_math_answer(text) == expected


def test_math_contest_parser_allows_final_answer_line_fallback() -> None:
    assert (
        extract_math_answer(
            "reasoning\nFinal Answer: 42",
            allow_final_answer_fallback=True,
        )
        == "42"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<answer>blue</answer>", "blue"),
        ("<ANSWER>7</ANSWER>", "7"),
        ("[ANSWER] 7 [/ANSWER]", None),
        ("Answer: triangle", None),
        ("<answer>No answer</answer>", None),
        ("<answer>MISSING</answer>", None),
        ("triangle", None),
        ("reasoning\nwithout marker", None),
    ],
)
def test_ar_parser(text: str, expected: str | None) -> None:
    assert extract_ar_answer(text) == expected


def test_ar_contest_parser_allows_final_answer_line_fallback() -> None:
    assert (
        extract_ar_answer(
            "reasoning\nFinal Answer: triangle",
            allow_final_answer_fallback=True,
        )
        == "triangle"
    )


def test_coding_parser_requires_complete_main() -> None:
    source = "#include <iostream>\nint main(){std::cout << 1;}"
    assert extract_cpp_code(f"```cpp\n{source}\n```") == source
    assert extract_cpp_code("```cpp\nint helper();\n```") is None
    assert extract_cpp_code("explanation only") is None


def test_data_fetch_reads_public_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "downloaded.jsonl"
    source.write_text('{"synthetic":true}\n', encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "data_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "repo_id": "example/r3bench-data",
                "revision": "immutable-revision",
                "domains": {
                    domain: {"relative_path": filename, "sha256": digest}
                    for domain, filename in {
                        "coding": "coding.jsonl",
                        "math": "math.jsonl",
                        "abstract_reasoning": "abstract_reasoning.jsonl",
                    }.items()
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "r3bench.cli.resolve_public_data_source",
        lambda domain, value, cache_dir=None: source,
    )
    output = tmp_path / "public_data"
    assert (
        cli_main(
            [
                "data",
                "fetch",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert {path.name for path in output.iterdir()} == {
        "coding.jsonl",
        "math.jsonl",
        "abstract_reasoning.jsonl",
    }
