"""Explicit local/Hugging Face resolution for public benchmark data."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from r3bench.common.schema import Domain


class DataSourceError(ValueError):
    """Raised when a public data source cannot be resolved safely."""


_HF_PREFIX = "hf://"
_REVISION = re.compile(r"^[A-Za-z0-9._/-]+$")
_HF_CANDIDATES: dict[Domain, tuple[str, ...]] = {
    "coding": (
        "coding/data/coding.jsonl",
        "data/coding.jsonl",
        "coding.jsonl",
    ),
    "math": (
        "math/data/math.jsonl",
        "data/math.jsonl",
        "math.jsonl",
        "math/data/problems.jsonl",
        "problems.jsonl",
    ),
    "abstract_reasoning": (
        "abstract_reasoning/data/abstract_reasoning.jsonl",
        "data/abstract_reasoning.jsonl",
        "abstract_reasoning.jsonl",
    ),
}


@dataclass(frozen=True, slots=True)
class HuggingFaceDataSource:
    repo_id: str
    revision: str | None = None
    filename: str | None = None


def parse_hf_source(value: str) -> HuggingFaceDataSource:
    """Parse ``hf://repo[@revision][::filename]`` without contacting a service."""

    if not value.startswith(_HF_PREFIX):
        raise DataSourceError("Hugging Face sources must start with hf://")
    body = value[len(_HF_PREFIX) :]
    repo_revision, separator, filename = body.partition("::")
    repo_id, revision_separator, revision = repo_revision.partition("@")
    if repo_id.count("/") != 1 or not all(repo_id.split("/")):
        raise DataSourceError("Hugging Face source requires an owner/repository ID")
    if revision_separator and (
        not revision or not _REVISION.fullmatch(revision) or ".." in revision
    ):
        raise DataSourceError("Hugging Face revision is invalid")
    if separator:
        path = Path(filename)
        if (
            not filename
            or path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".jsonl"
        ):
            raise DataSourceError("Hugging Face filename must be a safe JSONL path")
    return HuggingFaceDataSource(
        repo_id=repo_id,
        revision=revision if revision_separator else None,
        filename=filename if separator else None,
    )


def is_hf_source(value: str | Path) -> bool:
    return isinstance(value, str) and value.startswith(_HF_PREFIX)


def _local_path(value: str | Path, base_dir: str | Path | None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path


def resolve_public_data_source(
    domain: Domain,
    source: str | Path,
    *,
    base_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve an explicit public source and return a local file or directory.

    Network access occurs only when ``source`` starts with ``hf://``. Ordinary
    missing paths fail instead of being interpreted as repository IDs.
    """

    if not is_hf_source(source):
        path = _local_path(source, base_dir)
        if not path.exists():
            raise DataSourceError(f"public data source does not exist: {source}")
        return path

    parsed = parse_hf_source(str(source))
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
    except ImportError as exc:
        raise DataSourceError(
            "Hugging Face loading requires the optional huggingface_hub dependency"
        ) from exc

    filenames = (
        (parsed.filename,) if parsed.filename is not None else _HF_CANDIDATES[domain]
    )
    failures: list[str] = []
    for filename in filenames:
        try:
            downloaded = hf_hub_download(
                repo_id=parsed.repo_id,
                filename=filename,
                repo_type="dataset",
                revision=parsed.revision,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            )
            return Path(downloaded)
        except (EntryNotFoundError, HfHubHTTPError, OSError) as exc:
            failures.append(f"{filename}: {type(exc).__name__}")
    raise DataSourceError(
        "could not resolve a public dataset file from "
        f"{parsed.repo_id}; tried {', '.join(failures)}"
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: str | Path, expected: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise DataSourceError("expected SHA-256 must be 64 lowercase hex characters")
    actual = sha256_file(path)
    if actual != expected:
        raise DataSourceError(
            f"public data SHA-256 mismatch for {Path(path).name}: "
            f"expected {expected}, found {actual}"
        )


__all__ = [
    "DataSourceError",
    "HuggingFaceDataSource",
    "is_hf_source",
    "parse_hf_source",
    "resolve_public_data_source",
    "sha256_file",
    "verify_file_sha256",
]
