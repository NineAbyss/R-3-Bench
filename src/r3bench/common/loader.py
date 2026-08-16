"""Read-only adapters for the public R3Bench dataset layouts."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

from r3bench.common.data_source import (
    DataSourceError,
    is_hf_source,
    resolve_public_data_source,
    verify_file_sha256,
)
from r3bench.common.io import read_jsonl
from r3bench.common.schema import CONTEST_LABELS, ContestSuite, Domain, ProblemRecord


class DataContractError(ValueError):
    """Raised when public data violates the evaluator contract."""


TaskType = Literal["single_problem", "contest"]

CODING_METADATA_FIELDS = (
    "upstream_dataset",
    "upstream_id",
    "upstream_difficulty",
    "source_url",
    "statement_sha256",
)
MATH_METADATA_FIELDS = (
    "upstream_dataset",
    "upstream_split",
    "upstream_index",
    "source_suite_position",
    "statement_sha256",
    "upstream_difficulty",
    "topics",
)
AR_METADATA_FIELDS = (
    "category",
    "split",
    "entry_index",
    "row_seed",
    "generation_size",
    "generation_config",
    "dataset_params",
    "generator_metadata",
    "reference_answer_source",
    "dynamic_scorer",
)
EMAIL = re.compile(
    r"(?i)(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])"
)
SECRET_VALUE = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{12,}\b|\bhf_[A-Za-z0-9]{12,}\b|"
    r"\bolp_[A-Za-z0-9]{12,}\b|Bearer\s+[A-Za-z0-9._~+/-]{12,})"
)
MACHINE_PATH = re.compile(
    r"(?<![A-Za-z0-9])/(?:home|mnt|data|Users|app|logs)/[^\s`\"']+"
)


def _required(row: Mapping[str, Any], key: str) -> Any:
    if key not in row:
        raise DataContractError(f"missing required field: {key}")
    return row[key]


def _string(row: Mapping[str, Any], key: str) -> str:
    value = _required(row, key)
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(f"{key} must be a non-empty string")
    return value


def _integer(row: Mapping[str, Any], key: str) -> int:
    value = _required(row, key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataContractError(f"{key} must be an integer")
    return value


def _position(row: Mapping[str, Any]) -> int:
    value = _integer(row, "position")
    if not 1 <= value <= 6:
        raise DataContractError("position must be between 1 and 6")
    return value


def _difficulty(row: Mapping[str, Any]) -> str:
    value = _string(row, "difficulty")
    if value not in {"easy", "medium", "hard"}:
        raise DataContractError(f"unsupported difficulty: {value!r}")
    return value


def _label(task_type: TaskType, position: int) -> str | None:
    return CONTEST_LABELS[position - 1] if task_type == "contest" else None


def _allowlist(row: Mapping[str, Any], fields: Iterable[str]) -> dict[str, Any]:
    return {field: row[field] for field in fields if field in row}


def _verify_statement_hash(statement: str, expected: object) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise DataContractError("statement_sha256 must be a 64-character string")
    actual = hashlib.sha256(statement.encode("utf-8")).hexdigest()
    if actual != expected:
        raise DataContractError(
            f"statement SHA-256 mismatch: expected {expected}, found {actual}"
        )


def _validate_public_value(
    value: Any, *, path: str, reject_email: bool = False
) -> None:
    """Reject sensitive values from allowlisted public Math fields."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _validate_public_value(item, path=f"{path}.{key}", reject_email=reject_email)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_public_value(item, path=f"{path}[{index}]", reject_email=reject_email)
    elif isinstance(value, str):
        if MACHINE_PATH.search(value):
            raise DataContractError(f"machine-specific absolute path found in {path}")
        if SECRET_VALUE.search(value):
            raise DataContractError(f"credential-like value found in {path}")
        if reject_email and EMAIL.search(value):
            raise DataContractError(f"email address found in {path}")


def adapt_coding_row(
    row: Mapping[str, Any], *, split: str, task_type: TaskType
) -> ProblemRecord:
    """Adapt one allowlisted Coding public row."""

    statement = _string(row, "problem_statement")
    position = _position(row)
    platform = _string(row, "platform")
    if platform != "codeforces":
        raise DataContractError(f"unsupported coding platform: {platform!r}")
    title = _string(row, "title")
    time_limit = _integer(row, "time_limit")
    memory_limit = _integer(row, "memory_limit")
    source_url = _string(row, "source_url")
    upstream_id = _string(row, "upstream_id")
    statement_hash = _string(row, "statement_sha256")
    _verify_statement_hash(statement, statement_hash)

    payload = {
        "title": title,
        "time_limit_ms": time_limit,
        "memory_limit_mb": memory_limit,
        "source_url": source_url,
        "platform": platform,
        "upstream_dataset": _string(row, "upstream_dataset"),
        "upstream_id": upstream_id,
        "upstream_difficulty": _string(row, "upstream_difficulty"),
        "statement_sha256": statement_hash,
    }
    return ProblemRecord(
        domain="coding",
        split=split,
        task_type=task_type,
        problem_id=_string(row, "problem_id"),
        suite_id=_string(row, "suite_id"),
        problem_index=position,
        problem_label=_label(task_type, position),
        problem_statement=statement,
        difficulty=_difficulty(row),
        source=platform,
        metadata_public=_allowlist(row, CODING_METADATA_FIELDS),
        domain_payload=payload,
    )


def adapt_math_row(
    row: Mapping[str, Any], *, split: str, task_type: TaskType
) -> ProblemRecord:
    """Adapt one allowlisted Math public row."""

    declared_domain = row.get("domain", "math")
    if declared_domain != "math":
        raise DataContractError("Math rows must omit domain or declare domain='math'")
    statement = _string(row, "problem")
    position = _position(row)
    answer = _string(row, "answer")
    solution = row.get("solution")
    if solution is not None and not isinstance(solution, str):
        raise DataContractError("solution must be a string or null")
    if "statement_sha256" in row:
        _verify_statement_hash(statement, row["statement_sha256"])

    payload = {
        "answer": answer,
        "solution": solution,
    }
    public_metadata = _allowlist(row, MATH_METADATA_FIELDS)
    _validate_public_value(statement, path="problem")
    _validate_public_value(answer, path="answer")
    _validate_public_value(solution, path="solution", reject_email=True)
    _validate_public_value(public_metadata, path="metadata_public")
    return ProblemRecord(
        domain="math",
        split=split,
        task_type=task_type,
        problem_id=_string(row, "problem_id"),
        suite_id=_string(row, "suite_id"),
        problem_index=position,
        problem_label=_label(task_type, position),
        problem_statement=statement,
        difficulty=_difficulty(row),
        source=_string(row, "source"),
        metadata_public=public_metadata,
        domain_payload=payload,
    )


def adapt_abstract_reasoning_row(
    row: Mapping[str, Any], *, split: str, task_type: TaskType
) -> ProblemRecord:
    """Adapt one allowlisted Abstract Reasoning public row."""

    statement = _string(row, "question")
    answer = _string(row, "answer")
    generator = _string(row, "generator")
    metadata = _required(row, "metadata")
    if not isinstance(metadata, Mapping):
        raise DataContractError("metadata must be an object")
    position = _position(row)
    public_metadata = _allowlist(metadata, AR_METADATA_FIELDS)
    scorer_metadata = _allowlist(
        metadata,
        ("generation_config", "dataset_params", "generator_metadata"),
    )
    payload = {
        "answer": answer,
        "generator": generator,
        "scorer_metadata": scorer_metadata,
        "dynamic_scorer": bool(metadata.get("dynamic_scorer", False)),
        "reference_answer_source": metadata.get("reference_answer_source"),
    }
    return ProblemRecord(
        domain="abstract_reasoning",
        split=split,
        task_type=task_type,
        problem_id=_string(row, "problem_id"),
        suite_id=_string(row, "suite_id"),
        problem_index=position,
        problem_label=_label(task_type, position),
        problem_statement=statement,
        difficulty=_difficulty(row),
        source=_string(row, "source"),
        metadata_public=public_metadata,
        domain_payload=payload,
    )


ADAPTERS: dict[Domain, Callable[..., ProblemRecord]] = {
    "coding": adapt_coding_row,
    "math": adapt_math_row,
    "abstract_reasoning": adapt_abstract_reasoning_row,
}


def _find_existing(root: Path, candidates: Iterable[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    raise DataContractError(
        f"could not locate a public data file under {root}; tried {list(candidates)}"
    )


def _resolve_problem_file(
    domain: Domain,
    source: str | Path,
    *,
    base_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    try:
        path = resolve_public_data_source(
            domain, source, base_dir=base_dir, cache_dir=cache_dir
        )
    except DataSourceError as exc:
        raise DataContractError(str(exc)) from exc
    if path.is_file():
        return path
    candidates = {
        "coding": (
            "coding.jsonl",
            "data/coding.jsonl",
            "coding/data/coding.jsonl",
            "coding/public/data/coding.jsonl",
            "release/coding/public/data/coding.jsonl",
        ),
        "math": (
            "math.jsonl",
            "data/math.jsonl",
            "math/data/math.jsonl",
            "math/public/data/math.jsonl",
            "release/math/public/data/math.jsonl",
            "problems.jsonl",
            "data/problems.jsonl",
            "math/data/problems.jsonl",
            "release/huggingface/math/data/problems.jsonl",
        ),
        "abstract_reasoning": (
            "abstract_reasoning.jsonl",
            "data/abstract_reasoning.jsonl",
            "abstract_reasoning/data/abstract_reasoning.jsonl",
            "abstract_reasoning/public/data/abstract_reasoning.jsonl",
            "release/abstract_reasoning/public/data/abstract_reasoning.jsonl",
        ),
    }
    return _find_existing(path, candidates[domain])


def _resolve_math_suite_file(
    source: str | Path,
    problem_file: Path,
    *,
    base_dir: str | Path | None = None,
) -> Path | None:
    if is_hf_source(source):
        return None
    source_path = Path(source).expanduser()
    if not source_path.is_absolute() and base_dir is not None:
        source_path = Path(base_dir) / source_path
    if source_path.is_file():
        sibling = problem_file.with_name("suites.jsonl")
        if sibling.is_file():
            return sibling
        return None
    for candidate in (
        "suites.jsonl",
        "data/suites.jsonl",
        "math/data/suites.jsonl",
        "math/public/data/suites.jsonl",
        "release/huggingface/math/data/suites.jsonl",
    ):
        path = source_path / candidate
        if path.is_file():
            return path
    return None


def _validate_records(records: list[ProblemRecord], *, strict: bool) -> None:
    if not records:
        raise DataContractError("no problem records were loaded")
    ids = [record.problem_id for record in records]
    if len(ids) != len(set(ids)):
        duplicates = sorted(problem_id for problem_id, count in Counter(ids).items() if count > 1)
        raise DataContractError(f"duplicate problem IDs: {duplicates[:10]}")
    if strict and len(records) != 300:
        raise DataContractError(f"strict mode expected 300 problems, found {len(records)}")

    grouped: dict[str, list[ProblemRecord]] = defaultdict(list)
    for record in records:
        grouped[record.suite_id].append(record)
    if strict and len(grouped) != 50:
        raise DataContractError(f"strict mode expected 50 suites, found {len(grouped)}")
    for suite_id, suite_records in grouped.items():
        if len(suite_records) != 6:
            raise DataContractError(f"{suite_id}: expected six problems, found {len(suite_records)}")
        positions = sorted(record.problem_index for record in suite_records)
        if positions != [1, 2, 3, 4, 5, 6]:
            raise DataContractError(f"{suite_id}: invalid positions {positions}")
        composition = Counter(record.difficulty for record in suite_records)
        if composition != Counter({"easy": 3, "medium": 2, "hard": 1}):
            raise DataContractError(f"{suite_id}: invalid difficulty composition {dict(composition)}")


def _adapt_rows(
    domain: Domain,
    rows: list[dict[str, Any]],
    *,
    split: str,
    task_type: TaskType,
    strict: bool,
) -> list[ProblemRecord]:
    adapter = ADAPTERS[domain]
    try:
        records = [adapter(row, split=split, task_type=task_type) for row in rows]
    except (KeyError, TypeError) as exc:
        raise DataContractError(f"cannot adapt {domain} row: {exc}") from exc
    records.sort(key=lambda record: (record.suite_id, record.problem_index))
    _validate_records(records, strict=strict)
    return records


def _validate_domain(domain: str) -> Domain:
    if domain not in ADAPTERS:
        raise DataContractError(f"unsupported domain: {domain!r}")
    return domain  # type: ignore[return-value]


def _validate_split(split: str) -> str:
    if split != "test":
        raise DataContractError("the frozen public dataset supports split='test' only")
    return split


def load_single_problems(
    domain: Domain,
    split: str,
    data_root_or_hf_repo: str | Path,
    *,
    strict: bool = True,
    base_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    expected_sha256: str | None = None,
) -> tuple[ProblemRecord, ...]:
    """Load normalized single-problem records without external side effects."""

    checked_domain = _validate_domain(domain)
    checked_split = _validate_split(split)
    problem_file = _resolve_problem_file(
        checked_domain,
        data_root_or_hf_repo,
        base_dir=base_dir,
        cache_dir=cache_dir,
    )
    if expected_sha256 is not None:
        try:
            verify_file_sha256(problem_file, expected_sha256)
        except DataSourceError as exc:
            raise DataContractError(str(exc)) from exc
    records = _adapt_rows(
        checked_domain,
        read_jsonl(problem_file),
        split=checked_split,
        task_type="single_problem",
        strict=strict,
    )
    return tuple(records)


def _embedded_suites(records: list[ProblemRecord]) -> tuple[ContestSuite, ...]:
    grouped: dict[str, list[ProblemRecord]] = defaultdict(list)
    for record in records:
        grouped[record.suite_id].append(record)
    suites = []
    for suite_id in sorted(grouped):
        ordered = tuple(sorted(grouped[suite_id], key=lambda record: record.problem_index))
        suites.append(
            ContestSuite(
                domain=ordered[0].domain,
                split=ordered[0].split,
                suite_id=suite_id,
                problems=ordered,
            )
        )
    return tuple(suites)


def _math_suites(
    records: list[ProblemRecord], suite_rows: list[dict[str, Any]], *, strict: bool
) -> tuple[ContestSuite, ...]:
    by_id = {record.problem_id: record for record in records}
    if strict and len(suite_rows) != 50:
        raise DataContractError(f"strict mode expected 50 Math suite rows, found {len(suite_rows)}")
    seen_suites: set[str] = set()
    seen_problem_ids: set[str] = set()
    suites: list[ContestSuite] = []

    for suite_row in suite_rows:
        suite_id = _string(suite_row, "suite_id")
        if suite_id in seen_suites:
            raise DataContractError(f"duplicate Math suite ID: {suite_id}")
        seen_suites.add(suite_id)
        problem_ids = _required(suite_row, "problem_ids")
        positions = _required(suite_row, "positions")
        if not isinstance(problem_ids, list) or not isinstance(positions, list):
            raise DataContractError(f"{suite_id}: problem_ids and positions must be arrays")
        if len(problem_ids) != 6 or len(positions) != 6:
            raise DataContractError(f"{suite_id}: expected six problem IDs and positions")
        if any(isinstance(position, bool) or not isinstance(position, int) for position in positions):
            raise DataContractError(f"{suite_id}: positions must be integers")
        ordered_pairs = sorted(zip(positions, problem_ids), key=lambda pair: pair[0])
        if [position for position, _ in ordered_pairs] != [1, 2, 3, 4, 5, 6]:
            raise DataContractError(f"{suite_id}: invalid positions {positions}")

        ordered_records: list[ProblemRecord] = []
        for position, problem_id in ordered_pairs:
            if not isinstance(problem_id, str) or not problem_id:
                raise DataContractError(f"{suite_id}: invalid problem ID")
            if problem_id in seen_problem_ids:
                raise DataContractError(f"Math problem appears in multiple suites: {problem_id}")
            if problem_id not in by_id:
                raise DataContractError(f"{suite_id}: unknown problem ID {problem_id}")
            record = by_id[problem_id]
            if record.suite_id != suite_id or record.problem_index != position:
                raise DataContractError(
                    f"{problem_id}: problem row and suite manifest disagree on suite or position"
                )
            seen_problem_ids.add(problem_id)
            ordered_records.append(record)

        declared_difficulties = suite_row.get("difficulties")
        if declared_difficulties is not None:
            actual = [record.difficulty for record in ordered_records]
            if declared_difficulties != actual:
                raise DataContractError(f"{suite_id}: suite difficulty list does not match problems")
        suites.append(
            ContestSuite(
                domain="math",
                split=ordered_records[0].split,
                suite_id=suite_id,
                problems=tuple(ordered_records),
            )
        )

    if seen_problem_ids != set(by_id):
        missing = sorted(set(by_id) - seen_problem_ids)
        raise DataContractError(f"Math suite manifest does not cover all problems: {missing[:10]}")
    suites.sort(key=lambda suite: suite.suite_id)
    return tuple(suites)


def load_contest_suites(
    domain: Domain,
    split: str,
    data_root_or_hf_repo: str | Path,
    *,
    strict: bool = True,
    base_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    expected_sha256: str | None = None,
) -> tuple[ContestSuite, ...]:
    """Load canonical contest suites without applying a presentation shuffle."""

    checked_domain = _validate_domain(domain)
    checked_split = _validate_split(split)
    problem_file = _resolve_problem_file(
        checked_domain,
        data_root_or_hf_repo,
        base_dir=base_dir,
        cache_dir=cache_dir,
    )
    if expected_sha256 is not None:
        try:
            verify_file_sha256(problem_file, expected_sha256)
        except DataSourceError as exc:
            raise DataContractError(str(exc)) from exc
    records = _adapt_rows(
        checked_domain,
        read_jsonl(problem_file),
        split=checked_split,
        task_type="contest",
        strict=strict,
    )
    if checked_domain == "math":
        suite_file = _resolve_math_suite_file(
            data_root_or_hf_repo,
            problem_file,
            base_dir=base_dir,
        )
        if suite_file is not None:
            return _math_suites(records, read_jsonl(suite_file), strict=strict)
        return _embedded_suites(records)
    return _embedded_suites(records)


__all__ = [
    "DataContractError",
    "adapt_abstract_reasoning_row",
    "adapt_coding_row",
    "adapt_math_row",
    "load_contest_suites",
    "load_single_problems",
]
