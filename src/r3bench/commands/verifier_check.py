#!/usr/bin/env python3
"""Check Coding verifier readiness without starting or exposing the service."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3bench.coding.verifier import (
    LightCPVerifierConfig,
    LightCPVerifierConfigError,
    load_lightcpverifier_config,
    validate_lightcpverifier_config,
)
from r3bench.coding.assets import run_coding_asset_validation
from r3bench.common.io import read_jsonl
from r3bench.common.loader import load_single_problems
from r3bench.resource_paths import resolve_path


DEFAULT_DATA = "examples/data/coding.jsonl"
FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "hidden_tests",
        "hidden_testcase",
        "hidden_test_path",
        "testcase_path",
        "testcase_zip",
        "checker",
        "checker_path",
        "reference_solution",
        "verifier_root",
        "assets_root",
        "problem_assets_root",
        "service_url",
    }
)


class VerifierReadinessError(ValueError):
    """Raised when a verifier readiness request is unsafe."""


def _safe_output(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise VerifierReadinessError("output must be a safe relative path")
    if path.parts[0] != "outputs":
        raise VerifierReadinessError("output must be under outputs/")
    return path


def _data_path(value: str | Path) -> Path:
    return resolve_path(value)


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        if any(str(key).lower() in FORBIDDEN_PUBLIC_KEYS for key in value):
            return True
        return any(_contains_forbidden_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _strict_data(path: Path) -> bool:
    return not (
        path.name == "coding.jsonl"
        and "examples" in path.parts
        and "data" in path.parts
    )


def _check_service(config: LightCPVerifierConfig, timeout: float) -> tuple[bool, str]:
    service_url = config.effective_service_url
    if service_url is None:
        return False, "not_configured"
    import requests

    health_url = urljoin(service_url.rstrip("/") + "/", "health")
    try:
        response = requests.get(health_url, timeout=timeout)
    except requests.RequestException:
        return False, "health_request_failed"
    return response.status_code < 500, f"http_{response.status_code}"


def _runtime_config_from_legacy_args(
    *,
    judge_url: str | None,
    assets_root: str | None,
    verifier_root: str | None,
    timeout: float,
) -> LightCPVerifierConfig | None:
    if not any((judge_url, assets_root, verifier_root)):
        return None
    return LightCPVerifierConfig(
        judge_url=judge_url,
        service_url=judge_url,
        problem_assets_root=Path(assets_root) if assets_root else None,
        verifier_root=Path(verifier_root) if verifier_root else None,
        timeout_seconds=timeout,
        status="configured",
        requires_owner_approval=False,
        runtime_private_config=True,
    )


def check_readiness(
    *,
    data_source: str | Path,
    config_path: str | Path | None = None,
    asset_manifest_path: str | Path | None = None,
    judge_url: str | None = None,
    assets_root: str | None = None,
    verifier_root: str | None = None,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """Return only sanitized readiness facts."""

    data_path = _data_path(data_source)
    try:
        problems = load_single_problems(
            "coding", "test", data_path, strict=_strict_data(data_path)
        )
        raw_rows = read_jsonl(data_path)
    except (OSError, RuntimeError, ValueError):
        return {
            "schema_version": "1.1",
            "status": "data_contract_invalid",
            "public_problem_count": 0,
            "upstream_id_lookup_contract": False,
            "hidden_assets_in_public_dataset": False,
            "service_configured": False,
            "service_reachable": False,
            "service_check": "not_attempted",
            "assets_configured": False,
            "assets_available": False,
            "docker_started": False,
            "external_verifier_started": False,
            "private_runtime_values_serialized": False,
            "asset_manifest_status": "not_supplied",
            "asset_manifest_public_validation_status": None,
            "asset_manifest_upstream_id_coverage": None,
            "asset_manifest_assets_complete": None,
            "asset_manifest_unresolved_questions": [],
        }

    upstream_ids = [
        problem.domain_payload.get("upstream_id") for problem in problems
    ]
    upstream_contract_ok = (
        bool(upstream_ids)
        and all(isinstance(value, str) and value for value in upstream_ids)
        and len(set(upstream_ids)) == len(upstream_ids)
    )
    public_hidden_assets = any(_contains_forbidden_key(row) for row in raw_rows)
    result: dict[str, Any] = {
        "schema_version": "1.1",
        "status": "not_configured",
        "public_problem_count": len(problems),
        "upstream_id_lookup_contract": upstream_contract_ok,
        "hidden_assets_in_public_dataset": public_hidden_assets,
        "service_configured": False,
        "service_reachable": False,
        "service_check": "not_attempted",
        "assets_configured": False,
        "assets_available": False,
        "mode": "unconfigured",
        "docker_started": False,
        "external_verifier_started": False,
        "private_runtime_values_serialized": False,
        "asset_manifest_status": "not_supplied",
        "asset_manifest_public_validation_status": None,
        "asset_manifest_upstream_id_coverage": None,
        "asset_manifest_assets_complete": None,
        "asset_manifest_unresolved_questions": [],
    }
    if not upstream_contract_ok or public_hidden_assets:
        result["status"] = "data_contract_invalid"
        return result

    asset_validation: dict[str, Any] | None = None
    if asset_manifest_path is not None:
        asset_validation = run_coding_asset_validation(
            asset_manifest_path, data_path
        )
        result.update(
            {
                "asset_manifest_status": asset_validation["status"],
                "asset_manifest_public_validation_status": asset_validation[
                    "public_validation_status"
                ],
                "asset_manifest_upstream_id_coverage": asset_validation[
                    "upstream_id_coverage"
                ],
                "asset_manifest_assets_complete": asset_validation[
                    "assets_complete"
                ],
                "asset_manifest_unresolved_questions": asset_validation[
                    "unresolved_questions"
                ],
            }
        )
        if asset_validation["status"] == "invalid_manifest":
            result["status"] = "config_invalid"
            return result
        if asset_validation["status"] == "data_contract_invalid":
            result["status"] = "data_contract_invalid"
            return result
        if asset_validation["status"] in {"assets_incomplete", "hash_mismatch"}:
            result["status"] = "assets_unavailable"
            return result

    try:
        config = (
            load_lightcpverifier_config(config_path)
            if config_path is not None
            else _runtime_config_from_legacy_args(
                judge_url=judge_url,
                assets_root=assets_root,
                verifier_root=verifier_root,
                timeout=timeout,
            )
        )
        if config is None or config.status != "configured":
            return result
        validate_lightcpverifier_config(config, production=True)
        if (
            asset_validation is not None
            and asset_validation["asset_root_env"] is not None
            and config.asset_root_env != asset_validation["asset_root_env"]
        ):
            result["status"] = "config_invalid"
            return result
    except (LightCPVerifierConfigError, OSError, RuntimeError, ValueError):
        result["status"] = "config_invalid"
        return result

    assets_path = config.resolved_assets_root()
    assets_available = bool(assets_path and assets_path.is_dir())
    result.update(
        {
            "mode": config.mode,
            "service_configured": config.effective_service_url is not None,
            "assets_configured": config.asset_root_env is not None
            or config.problem_assets_root is not None,
            "assets_available": assets_available,
        }
    )
    if not assets_available:
        result["status"] = "assets_unavailable"
        return result
    if config.mode == "local":
        # The public package intentionally has no local executable protocol.
        result["status"] = "config_invalid"
        result["service_check"] = "local_client_binding_required"
        return result
    reachable, service_check = _check_service(config, timeout)
    result["service_reachable"] = reachable
    result["service_check"] = service_check
    result["status"] = "configured" if reachable else "service_unreachable"
    return result


def _write_report(output: Path, result: dict[str, Any]) -> None:
    target = output
    target.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    )
    for name in (
        "verifier_readiness.json",
        "coding_verifier_status.json",
        "scoring_summary.json",
    ):
        (target / name).write_text(serialized, encoding="utf-8")
    report = [
        "# Coding Verifier Readiness",
        "",
        f"- Status: `{result['status']}`",
        f"- Public problems: {result['public_problem_count']}",
        f"- Unique upstream ID contract: `{str(result['upstream_id_lookup_contract']).lower()}`",
        f"- Hidden verifier assets in public data: `{str(result['hidden_assets_in_public_dataset']).lower()}`",
        f"- Service configured: `{str(result['service_configured']).lower()}`",
        f"- Service reachable: `{str(result['service_reachable']).lower()}`",
        f"- Assets available: `{str(result['assets_available']).lower()}`",
        f"- Asset manifest status: `{result['asset_manifest_status']}`",
        f"- Asset manifest ID coverage: `{result['asset_manifest_upstream_id_coverage']}`",
        "",
        "This check never starts Docker, Harbor, Terminus, or a verifier service.",
        "Runtime service locations and asset paths are not serialized.",
        "",
    ]
    report_text = "\n".join(report)
    for name in ("verifier_readiness_report.md", "scoring_report.md"):
        (target / name).write_text(report_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--config")
    parser.add_argument("--asset-manifest")
    # Legacy runtime-only flags remain accepted for backward compatibility.
    parser.add_argument("--judge-url")
    parser.add_argument("--assets-root")
    parser.add_argument("--verifier-root")
    parser.add_argument("--timeout", type=float, default=2.0)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = _safe_output(args.output)
        result = check_readiness(
            data_source=args.data,
            config_path=args.config,
            asset_manifest_path=args.asset_manifest,
            judge_url=args.judge_url,
            assets_root=args.assets_root,
            verifier_root=args.verifier_root,
            timeout=args.timeout,
        )
        _write_report(output, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "public_problem_count": result["public_problem_count"],
                    "upstream_id_lookup_contract": result[
                        "upstream_id_lookup_contract"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"coding verifier readiness failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
