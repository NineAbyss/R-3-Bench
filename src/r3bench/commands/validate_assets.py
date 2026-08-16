#!/usr/bin/env python3
"""Validate public Coding metadata and optional external verifier assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3bench.coding.assets import run_coding_asset_validation


def _write_report(output: Path, result: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "asset_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = [
        "# Coding Asset Validation",
        "",
        f"- Status: `{result['status']}`",
        f"- Public validation: `{result['public_validation_status']}`",
        f"- Public problems: {result['public_problem_count']}",
        f"- Upstream ID coverage: `{str(result['upstream_id_coverage']).lower()}`",
        f"- Asset root configured: `{str(result['asset_root_configured']).lower()}`",
        f"- Expected packages: {result['expected_package_count']}",
        f"- Present packages: {result['present_package_count']}",
        f"- Missing packages: {result['missing_package_count']}",
        f"- Hash mismatches: {result['hash_mismatch_count']}",
        "",
        "No verifier was started. Local paths, hidden filenames, and hidden",
        "test content are not serialized.",
        "",
    ]
    (output / "asset_validation_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_coding_asset_validation(args.manifest, args.data)
    try:
        _write_report(Path(args.output), result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "public_validation_status": result[
                        "public_validation_status"
                    ],
                    "expected_package_count": result[
                        "expected_package_count"
                    ],
                    "present_package_count": result["present_package_count"],
                },
                sort_keys=True,
            )
        )
    except OSError as exc:
        print(f"Coding asset validation failed: {exc}", file=sys.stderr)
        return 2
    return (
        0
        if result["status"]
        in {
            "manifest_valid_public_only",
            "asset_root_not_configured",
            "assets_complete",
        }
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
