"""Public Coding asset contracts and non-leaking external-asset validation."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from r3bench.common.io import read_json, read_jsonl


MANIFEST_SCHEMA_VERSION = "1.0"
ASSET_LAYOUT_SCHEMA_VERSION = "lightcpverifier_problem_package_v1"
PACKAGE_HASH_ALGORITHM = "sha256_tree_v1"
ASSET_TREE_HASH_ALGORITHM = "sha256_package_set_v1"
PUBLIC_VALIDATION_STATUS = "manifest_valid_public_only"
MANIFEST_STATUSES = frozenset({"draft", "release"})
ASSET_VALIDATION_STATUSES = frozenset(
    {
        PUBLIC_VALIDATION_STATUS,
        "asset_root_not_configured",
        "assets_complete",
        "assets_incomplete",
        "hash_mismatch",
        "invalid_manifest",
        "data_contract_invalid",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "manifest_schema_version",
        "domain",
        "benchmark_name",
        "upstream_dataset",
        "expected_problem_count",
        "expected_upstream_ids",
        "coding_data_sha256",
        "asset_root_env",
        "lightcpverifier_compatible_version",
        "lightcpverifier_compatible_commit",
        "upstream_asset_revision",
        "asset_layout_schema_version",
        "package_marker",
        "expected_package_count",
        "package_hash_algorithm",
        "package_sha256_by_upstream_id",
        "asset_tree_hash_algorithm",
        "asset_tree_sha256",
        "unresolved_questions",
        "notes",
        "status",
        "requires_owner_approval",
    }
)
_UPSTREAM_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_CREDENTIAL = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{12,}\b|\bhf_[A-Za-z0-9]{12,}\b|"
    r"\bolp_[A-Za-z0-9]{12,}\b|Bearer\s+[A-Za-z0-9._~+/-]{12,})"
)
_FORBIDDEN_KEYS = frozenset(
    {
        "api_key",
        "access_token",
        "password",
        "authorization",
        "service_url",
        "asset_root",
        "assets_root",
        "hidden_tests",
        "hidden_test_path",
        "testcase_path",
        "checker_path",
        "reference_solution",
        "provider_headers",
        "provider_request_id",
    }
)


class CodingAssetManifestError(ValueError):
    """Raised when a public Coding asset manifest is unsafe or inconsistent."""


class CodingAssetTreeError(ValueError):
    """Raised when an external package cannot be hashed safely."""


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CodingAssetManifestError(f"{field} must be a non-empty string")
    return value


def _scan_safe(value: Any, path: str = "asset_manifest") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.lower() in _FORBIDDEN_KEYS:
                raise CodingAssetManifestError(
                    f"{path} contains forbidden field {key!r}"
                )
            _scan_safe(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_safe(item, f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    if _CREDENTIAL.search(value):
        raise CodingAssetManifestError(
            f"{path} contains an API-key-like value"
        )
    if Path(value).is_absolute() or _WINDOWS_ABSOLUTE.match(value):
        raise CodingAssetManifestError(
            f"{path} contains a private absolute path"
        )
    if ".env" in Path(value).parts:
        raise CodingAssetManifestError(f"{path} contains a forbidden .env path")
    if re.match(r"(?i)^https?://", value):
        raise CodingAssetManifestError(f"{path} contains a service endpoint")


@dataclass(frozen=True, slots=True)
class CodingAssetManifest:
    manifest_schema_version: str
    domain: str
    benchmark_name: str
    upstream_dataset: str
    expected_problem_count: int
    expected_upstream_ids: tuple[str, ...]
    coding_data_sha256: str
    asset_root_env: str | None
    lightcpverifier_compatible_version: str
    lightcpverifier_compatible_commit: str
    upstream_asset_revision: str
    asset_layout_schema_version: str
    package_marker: str
    expected_package_count: int
    package_hash_algorithm: str
    package_sha256_by_upstream_id: Mapping[str, str]
    asset_tree_hash_algorithm: str
    asset_tree_sha256: str | None
    unresolved_questions: tuple[str, ...]
    notes: str
    status: str
    requires_owner_approval: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_upstream_ids",
            tuple(self.expected_upstream_ids),
        )
        object.__setattr__(
            self,
            "package_sha256_by_upstream_id",
            MappingProxyType(dict(self.package_sha256_by_upstream_id)),
        )
        object.__setattr__(
            self, "unresolved_questions", tuple(self.unresolved_questions)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_schema_version": self.manifest_schema_version,
            "domain": self.domain,
            "benchmark_name": self.benchmark_name,
            "upstream_dataset": self.upstream_dataset,
            "expected_problem_count": self.expected_problem_count,
            "expected_upstream_ids": list(self.expected_upstream_ids),
            "coding_data_sha256": self.coding_data_sha256,
            "asset_root_env": self.asset_root_env,
            "lightcpverifier_compatible_version": self.lightcpverifier_compatible_version,
            "lightcpverifier_compatible_commit": self.lightcpverifier_compatible_commit,
            "upstream_asset_revision": self.upstream_asset_revision,
            "asset_layout_schema_version": self.asset_layout_schema_version,
            "package_marker": self.package_marker,
            "expected_package_count": self.expected_package_count,
            "package_hash_algorithm": self.package_hash_algorithm,
            "package_sha256_by_upstream_id": dict(
                self.package_sha256_by_upstream_id
            ),
            "asset_tree_hash_algorithm": self.asset_tree_hash_algorithm,
            "asset_tree_sha256": self.asset_tree_sha256,
            "unresolved_questions": list(self.unresolved_questions),
            "notes": self.notes,
            "status": self.status,
            "requires_owner_approval": self.requires_owner_approval,
        }


def _validate_unresolved_or_version(value: object, field: str) -> str:
    text = _text(value, field)
    if text == "unresolved":
        return text
    if not re.fullmatch(r"[A-Za-z0-9_.+-]+", text):
        raise CodingAssetManifestError(f"{field} has an invalid version value")
    return text


def _validate_unresolved_or_commit(value: object, field: str) -> str:
    text = _text(value, field)
    if text not in {"unresolved", "not_recorded"} and not _COMMIT.fullmatch(text):
        raise CodingAssetManifestError(
            f"{field} must be unresolved, not_recorded, or a 40-character Git commit"
        )
    return text


def validate_coding_asset_manifest(
    raw: Mapping[str, Any] | CodingAssetManifest,
) -> CodingAssetManifest:
    if isinstance(raw, CodingAssetManifest):
        raw = raw.to_dict()
    if not isinstance(raw, Mapping) or set(raw) != _MANIFEST_FIELDS:
        raise CodingAssetManifestError(
            "Coding asset manifest fields do not match schema 1.0"
        )
    _scan_safe(raw)
    if raw.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise CodingAssetManifestError("unsupported manifest_schema_version")
    if raw.get("domain") != "coding":
        raise CodingAssetManifestError("asset manifest domain must be coding")
    if raw.get("benchmark_name") != "R3Bench Coding":
        raise CodingAssetManifestError("unexpected benchmark_name")
    if raw.get("upstream_dataset") != "QAQAQAQAQ/LiveCodeBench-Pro":
        raise CodingAssetManifestError("unexpected upstream_dataset")

    problem_count = raw.get("expected_problem_count")
    package_count = raw.get("expected_package_count")
    for value, field in (
        (problem_count, "expected_problem_count"),
        (package_count, "expected_package_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CodingAssetManifestError(f"{field} must be a positive integer")
    ids_raw = raw.get("expected_upstream_ids")
    if not isinstance(ids_raw, list) or not ids_raw:
        raise CodingAssetManifestError(
            "expected_upstream_ids must be a non-empty list"
        )
    ids: tuple[str, ...] = tuple(
        _text(value, "expected_upstream_ids") for value in ids_raw
    )
    if any(not _UPSTREAM_ID.fullmatch(value) for value in ids):
        raise CodingAssetManifestError("expected_upstream_ids contains an invalid ID")
    if len(set(ids)) != len(ids):
        raise CodingAssetManifestError("expected_upstream_ids contains duplicates")
    if problem_count != len(ids) or package_count != len(ids):
        raise CodingAssetManifestError(
            "problem/package counts must match expected_upstream_ids"
        )
    data_sha = _text(raw.get("coding_data_sha256"), "coding_data_sha256")
    if not _SHA256.fullmatch(data_sha):
        raise CodingAssetManifestError("coding_data_sha256 must be SHA-256")
    env_name = raw.get("asset_root_env")
    if env_name is not None and not (
        isinstance(env_name, str) and _ENV_NAME.fullmatch(env_name)
    ):
        raise CodingAssetManifestError(
            "asset_root_env must name an environment variable or be null"
        )
    version = _validate_unresolved_or_version(
        raw.get("lightcpverifier_compatible_version"),
        "lightcpverifier_compatible_version",
    )
    commit = _validate_unresolved_or_commit(
        raw.get("lightcpverifier_compatible_commit"),
        "lightcpverifier_compatible_commit",
    )
    revision = _validate_unresolved_or_version(
        raw.get("upstream_asset_revision"), "upstream_asset_revision"
    )
    if raw.get("asset_layout_schema_version") != ASSET_LAYOUT_SCHEMA_VERSION:
        raise CodingAssetManifestError("unsupported asset_layout_schema_version")
    if raw.get("package_marker") != "config.yaml":
        raise CodingAssetManifestError("package_marker must be config.yaml")
    if raw.get("package_hash_algorithm") != PACKAGE_HASH_ALGORITHM:
        raise CodingAssetManifestError("unsupported package_hash_algorithm")
    if raw.get("asset_tree_hash_algorithm") != ASSET_TREE_HASH_ALGORITHM:
        raise CodingAssetManifestError("unsupported asset_tree_hash_algorithm")

    package_hashes = raw.get("package_sha256_by_upstream_id")
    if not isinstance(package_hashes, Mapping):
        raise CodingAssetManifestError(
            "package_sha256_by_upstream_id must be an object"
        )
    checked_hashes: dict[str, str] = {}
    for raw_id, raw_hash in package_hashes.items():
        upstream_id = _text(raw_id, "package hash upstream_id")
        digest = _text(raw_hash, f"package hash for {upstream_id}")
        if upstream_id not in set(ids):
            raise CodingAssetManifestError(
                "package hash references an unexpected upstream_id"
            )
        if not _SHA256.fullmatch(digest):
            raise CodingAssetManifestError("package hash must be SHA-256")
        checked_hashes[upstream_id] = digest
    tree_hash = raw.get("asset_tree_sha256")
    if tree_hash is not None and not (
        isinstance(tree_hash, str) and _SHA256.fullmatch(tree_hash)
    ):
        raise CodingAssetManifestError(
            "asset_tree_sha256 must be null or SHA-256"
        )
    questions_raw = raw.get("unresolved_questions")
    if not isinstance(questions_raw, list):
        raise CodingAssetManifestError("unresolved_questions must be a list")
    questions = tuple(_text(item, "unresolved question") for item in questions_raw)
    status = raw.get("status")
    if status not in MANIFEST_STATUSES:
        raise CodingAssetManifestError(
            "asset manifest status must be draft or release"
        )
    if tree_hash is not None:
        if set(checked_hashes) != set(ids):
            raise CodingAssetManifestError(
                "fingerprinted manifest must hash every expected package"
            )
        if compute_asset_tree_sha256(ids, checked_hashes) != tree_hash:
            raise CodingAssetManifestError(
                "asset_tree_sha256 does not match package hashes"
            )
    requires_owner_approval = raw.get("requires_owner_approval")
    if not isinstance(requires_owner_approval, bool):
        raise CodingAssetManifestError("requires_owner_approval must be boolean")
    if status == "release":
        if requires_owner_approval:
            raise CodingAssetManifestError(
                "release asset manifest cannot require owner approval"
            )
        if tree_hash is None or set(checked_hashes) != set(ids):
            raise CodingAssetManifestError(
                "release asset manifest must fingerprint every package"
            )
        if questions:
            raise CodingAssetManifestError(
                "release asset manifest cannot contain unresolved questions"
            )

    return CodingAssetManifest(
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        domain="coding",
        benchmark_name="R3Bench Coding",
        upstream_dataset="QAQAQAQAQ/LiveCodeBench-Pro",
        expected_problem_count=problem_count,
        expected_upstream_ids=ids,
        coding_data_sha256=data_sha,
        asset_root_env=env_name,
        lightcpverifier_compatible_version=version,
        lightcpverifier_compatible_commit=commit,
        upstream_asset_revision=revision,
        asset_layout_schema_version=ASSET_LAYOUT_SCHEMA_VERSION,
        package_marker="config.yaml",
        expected_package_count=package_count,
        package_hash_algorithm=PACKAGE_HASH_ALGORITHM,
        package_sha256_by_upstream_id=checked_hashes,
        asset_tree_hash_algorithm=ASSET_TREE_HASH_ALGORITHM,
        asset_tree_sha256=tree_hash,
        unresolved_questions=questions,
        notes=_text(raw.get("notes"), "notes"),
        status=status,
        requires_owner_approval=requires_owner_approval,
    )


def load_coding_asset_manifest(path: str | Path) -> CodingAssetManifest:
    raw = read_json(path)
    if not isinstance(raw, Mapping):
        raise CodingAssetManifestError("asset manifest root must be an object")
    return validate_coding_asset_manifest(raw)


def _public_rows(data_source: str | Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    rows = read_jsonl(data_source)
    ids: list[str] = []
    for line_number, row in enumerate(rows, start=1):
        upstream_id = row.get("upstream_id")
        if not isinstance(upstream_id, str) or not _UPSTREAM_ID.fullmatch(
            upstream_id
        ):
            raise CodingAssetManifestError(
                f"Coding data row {line_number} has an invalid upstream_id"
            )
        if row.get("upstream_dataset") != "QAQAQAQAQ/LiveCodeBench-Pro" and not str(
            row.get("upstream_dataset", "")
        ).startswith("synthetic/"):
            raise CodingAssetManifestError(
                f"Coding data row {line_number} has an unexpected upstream_dataset"
            )
        ids.append(upstream_id)
    if len(set(ids)) != len(ids):
        raise CodingAssetManifestError("Coding data contains duplicate upstream_id values")
    return rows, tuple(ids)


def build_coding_asset_manifest(
    data_source: str | Path,
    *,
    asset_root_env: str | None = "LIGHTCPVERIFIER_ASSET_ROOT",
    lightcpverifier_compatible_version: str = "unresolved",
    lightcpverifier_compatible_commit: str = "unresolved",
    upstream_asset_revision: str = "unresolved",
) -> CodingAssetManifest:
    rows, upstream_ids = _public_rows(data_source)
    del rows
    data_sha = hashlib.sha256(Path(data_source).read_bytes()).hexdigest()
    unresolved: list[str] = []
    if lightcpverifier_compatible_version == "unresolved":
        unresolved.append(
            "Confirm the exact LightCPVerifier version used for formal Coding scoring."
        )
    if lightcpverifier_compatible_commit == "unresolved":
        unresolved.append(
            "Confirm the exact LightCPVerifier Git commit used for formal Coding scoring."
        )
    if upstream_asset_revision == "unresolved":
        unresolved.append(
            "Confirm the exact upstream hidden-asset revision compatible with the benchmark."
        )
    raw = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "domain": "coding",
        "benchmark_name": "R3Bench Coding",
        "upstream_dataset": "QAQAQAQAQ/LiveCodeBench-Pro",
        "expected_problem_count": len(upstream_ids),
        "expected_upstream_ids": list(upstream_ids),
        "coding_data_sha256": data_sha,
        "asset_root_env": asset_root_env,
        "lightcpverifier_compatible_version": lightcpverifier_compatible_version,
        "lightcpverifier_compatible_commit": lightcpverifier_compatible_commit,
        "upstream_asset_revision": upstream_asset_revision,
        "asset_layout_schema_version": ASSET_LAYOUT_SCHEMA_VERSION,
        "package_marker": "config.yaml",
        "expected_package_count": len(upstream_ids),
        "package_hash_algorithm": PACKAGE_HASH_ALGORITHM,
        "package_sha256_by_upstream_id": {},
        "asset_tree_hash_algorithm": ASSET_TREE_HASH_ALGORITHM,
        "asset_tree_sha256": None,
        "unresolved_questions": unresolved,
        "notes": (
            "Public draft contract only. Hidden assets, local paths, "
            "service endpoints, and reference solutions are not included."
        ),
        "status": "draft",
        "requires_owner_approval": True,
    }
    return validate_coding_asset_manifest(raw)


def fingerprint_coding_asset_manifest(
    manifest: CodingAssetManifest,
    data_source: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> CodingAssetManifest:
    """Content-address a complete, separately provisioned Coding asset set.

    Only per-problem digests and the aggregate digest enter the returned
    manifest. Runtime paths, hidden filenames, and hidden file contents are
    never serialized.
    """

    validate_manifest_against_data(manifest, data_source)
    if manifest.asset_root_env is None:
        raise CodingAssetTreeError(
            "asset fingerprinting requires asset_root_env"
        )
    environment = os.environ if environ is None else environ
    root_value = environment.get(manifest.asset_root_env)
    if not root_value:
        raise CodingAssetTreeError(
            f"{manifest.asset_root_env} is not configured"
        )
    root = Path(root_value)
    if not root.is_dir():
        raise CodingAssetTreeError("configured asset root is unavailable")

    package_hashes: dict[str, str] = {}
    for upstream_id in manifest.expected_upstream_ids:
        package = root / upstream_id
        marker = package / manifest.package_marker
        if (
            not package.is_dir()
            or package.is_symlink()
            or not marker.is_file()
            or marker.is_symlink()
        ):
            raise CodingAssetTreeError(
                f"asset package is incomplete for {upstream_id}"
            )
        package_hashes[upstream_id] = compute_package_tree_sha256(package)

    tree_hash = compute_asset_tree_sha256(
        manifest.expected_upstream_ids, package_hashes
    )
    raw = manifest.to_dict()
    raw["package_sha256_by_upstream_id"] = package_hashes
    raw["asset_tree_sha256"] = tree_hash
    if raw["upstream_asset_revision"] in {
        "unresolved",
        "content-addressed",
    }:
        raw["upstream_asset_revision"] = f"sha256-{tree_hash}"
    if raw["lightcpverifier_compatible_commit"] == "unresolved":
        raw["lightcpverifier_compatible_commit"] = "not_recorded"
    raw["unresolved_questions"] = []
    raw["notes"] = (
        "Content-addressed asset contract. It contains public problem "
        "IDs and SHA-256 digests only; hidden files, paths, and contents are "
        "not included."
    )
    raw["status"] = "release"
    raw["requires_owner_approval"] = False
    return validate_coding_asset_manifest(raw)


def validate_manifest_against_data(
    manifest: CodingAssetManifest, data_source: str | Path
) -> tuple[int, tuple[str, ...]]:
    rows, ids = _public_rows(data_source)
    data_sha = hashlib.sha256(Path(data_source).read_bytes()).hexdigest()
    if data_sha != manifest.coding_data_sha256:
        raise CodingAssetManifestError("Coding data SHA-256 does not match manifest")
    if len(rows) != manifest.expected_problem_count:
        raise CodingAssetManifestError("Coding data row count does not match manifest")
    if ids != manifest.expected_upstream_ids:
        raise CodingAssetManifestError(
            "Coding data upstream_id order does not match manifest"
        )
    return len(rows), ids


def compute_package_tree_sha256(package: str | Path) -> str:
    package = Path(package)
    files: list[Path] = []
    for candidate in package.rglob("*"):
        if candidate.is_symlink():
            raise CodingAssetTreeError("asset package contains a symlink")
        if candidate.is_file():
            files.append(candidate)
    digest = hashlib.sha256(b"r3bench-sha256-tree-v1\0")
    for path in sorted(files, key=lambda item: item.relative_to(package).as_posix()):
        relative = path.relative_to(package).as_posix().encode("utf-8")
        size = path.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def compute_asset_tree_sha256(
    upstream_ids: tuple[str, ...], package_hashes: Mapping[str, str]
) -> str:
    digest = hashlib.sha256(b"r3bench-sha256-package-set-v1\0")
    for upstream_id in upstream_ids:
        id_bytes = upstream_id.encode("utf-8")
        digest.update(len(id_bytes).to_bytes(8, "big"))
        digest.update(id_bytes)
        digest.update(bytes.fromhex(package_hashes[upstream_id]))
    return digest.hexdigest()


def _base_result(
    *,
    status: str,
    manifest: CodingAssetManifest | None = None,
    public_problem_count: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": status,
        "public_validation_status": (
            PUBLIC_VALIDATION_STATUS if manifest is not None else None
        ),
        "manifest_schema_version": (
            manifest.manifest_schema_version if manifest is not None else None
        ),
        "public_problem_count": public_problem_count,
        "expected_problem_count": (
            manifest.expected_problem_count if manifest is not None else None
        ),
        "upstream_id_coverage": manifest is not None and public_problem_count > 0,
        "asset_root_env": manifest.asset_root_env if manifest is not None else None,
        "asset_root_configured": False,
        "expected_package_count": (
            manifest.expected_package_count if manifest is not None else None
        ),
        "present_package_count": 0,
        "missing_package_count": (
            manifest.expected_package_count if manifest is not None else None
        ),
        "missing_upstream_ids": [],
        "hashes_declared": bool(
            manifest
            and (
                manifest.package_sha256_by_upstream_id
                or manifest.asset_tree_sha256
            )
        ),
        "hashes_checked": 0,
        "hash_mismatch_count": 0,
        "hash_mismatch_upstream_ids": [],
        "asset_tree_hash_checked": False,
        "asset_tree_hash_matches": None,
        "assets_complete": False,
        "unresolved_questions": (
            list(manifest.unresolved_questions) if manifest is not None else []
        ),
        "paths_serialized": False,
        "hidden_content_serialized": False,
        "docker_started": False,
        "verifier_started": False,
    }


def validate_coding_assets(
    manifest: CodingAssetManifest,
    data_source: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    public_count, _ = validate_manifest_against_data(manifest, data_source)
    result = _base_result(
        status=PUBLIC_VALIDATION_STATUS,
        manifest=manifest,
        public_problem_count=public_count,
    )
    if manifest.asset_root_env is None:
        return result
    environment = os.environ if environ is None else environ
    root_value = environment.get(manifest.asset_root_env)
    if not root_value:
        result["status"] = "asset_root_not_configured"
        return result
    result["asset_root_configured"] = True
    root = Path(root_value)
    if not root.is_dir():
        result["status"] = "assets_incomplete"
        return result

    present: list[str] = []
    missing: list[str] = []
    for upstream_id in manifest.expected_upstream_ids:
        package = root / upstream_id
        marker = package / manifest.package_marker
        if (
            package.is_dir()
            and not package.is_symlink()
            and marker.is_file()
            and not marker.is_symlink()
        ):
            present.append(upstream_id)
        else:
            missing.append(upstream_id)
    result["present_package_count"] = len(present)
    result["missing_package_count"] = len(missing)
    result["missing_upstream_ids"] = missing
    if missing:
        result["status"] = "assets_incomplete"
        return result

    need_all_hashes = manifest.asset_tree_sha256 is not None
    ids_to_hash = (
        manifest.expected_upstream_ids
        if need_all_hashes
        else tuple(manifest.package_sha256_by_upstream_id)
    )
    computed_hashes: dict[str, str] = {}
    mismatches: list[str] = []
    for upstream_id in ids_to_hash:
        try:
            computed = compute_package_tree_sha256(root / upstream_id)
        except (CodingAssetTreeError, OSError):
            mismatches.append(upstream_id)
            continue
        computed_hashes[upstream_id] = computed
        expected = manifest.package_sha256_by_upstream_id.get(upstream_id)
        if expected is not None and expected != computed:
            mismatches.append(upstream_id)
    result["hashes_checked"] = len(computed_hashes)
    if manifest.asset_tree_sha256 is not None and not mismatches:
        result["asset_tree_hash_checked"] = True
        computed_tree = compute_asset_tree_sha256(
            manifest.expected_upstream_ids, computed_hashes
        )
        tree_matches = computed_tree == manifest.asset_tree_sha256
        result["asset_tree_hash_matches"] = tree_matches
        if not tree_matches:
            mismatches.extend(manifest.expected_upstream_ids)
    result["hash_mismatch_upstream_ids"] = list(dict.fromkeys(mismatches))
    result["hash_mismatch_count"] = len(result["hash_mismatch_upstream_ids"])
    if mismatches:
        result["status"] = "hash_mismatch"
        return result
    result["assets_complete"] = True
    result["status"] = "assets_complete"
    return result


def run_coding_asset_validation(
    manifest_path: str | Path,
    data_source: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    try:
        manifest = load_coding_asset_manifest(manifest_path)
    except (OSError, RuntimeError, ValueError):
        return _base_result(status="invalid_manifest")
    try:
        return validate_coding_assets(manifest, data_source, environ=environ)
    except (OSError, RuntimeError, ValueError):
        return _base_result(
            status="data_contract_invalid", manifest=manifest
        )


__all__ = [
    "ASSET_LAYOUT_SCHEMA_VERSION",
    "ASSET_TREE_HASH_ALGORITHM",
    "ASSET_VALIDATION_STATUSES",
    "CodingAssetManifest",
    "CodingAssetManifestError",
    "CodingAssetTreeError",
    "MANIFEST_SCHEMA_VERSION",
    "PACKAGE_HASH_ALGORITHM",
    "PUBLIC_VALIDATION_STATUS",
    "build_coding_asset_manifest",
    "compute_asset_tree_sha256",
    "compute_package_tree_sha256",
    "fingerprint_coding_asset_manifest",
    "load_coding_asset_manifest",
    "run_coding_asset_validation",
    "validate_coding_asset_manifest",
    "validate_coding_assets",
    "validate_manifest_against_data",
]
