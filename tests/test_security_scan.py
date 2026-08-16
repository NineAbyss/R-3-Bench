from __future__ import annotations

import json
import re
from pathlib import Path

from r3bench.security import scan_tree


_LOCAL_LINK = re.compile(r"\[[^]]*\]\((?!https?://|mailto:|#)([^)#]+)(?:#[^)]+)?\)")


def test_release_source_safety_scan_passes(release_root: Path) -> None:
    result = scan_tree(release_root, kind="source")
    assert result.passed
    assert result.findings == ()
    assert result.scanned_file_count > 100
    assert result.guard_regex_files


def test_output_scan_reports_fields_without_values(tmp_path: Path) -> None:
    safe = tmp_path / "safe"
    safe.mkdir()
    (safe / "summary.json").write_text(
        json.dumps({"status": "pass", "network_called": False}) + "\n",
        encoding="utf-8",
    )
    assert scan_tree(safe, kind="outputs").passed

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    (unsafe / "result.json").write_text(
        json.dumps({"provider_headers": {"synthetic": "redacted"}}) + "\n",
        encoding="utf-8",
    )
    result = scan_tree(unsafe, kind="outputs")
    assert not result.passed
    assert [finding.category for finding in result.findings] == [
        "forbidden_output_field"
    ]
    assert result.to_dict()["findings"][0] == {
        "category": "forbidden_output_field",
        "path": "result.json",
        "line": None,
    }


def test_public_document_set_and_local_links_are_clean(release_root: Path) -> None:
    docs = release_root / "docs"
    assert {path.name for path in docs.glob("*.md")} == {
        "ARCHITECTURE.md",
        "DATA_AND_SCORING.md",
        "QUICKSTART.md",
        "REPRODUCTION_SCOPE.md",
        "RESPONSE_CURVE_AND_ORACLE.md",
    }
    markdown_files = [release_root / "README.md", *sorted(docs.glob("*.md"))]
    broken: list[str] = []
    for document in markdown_files:
        text = document.read_text(encoding="utf-8")
        for match in _LOCAL_LINK.finditer(text):
            target = (document.parent / match.group(1)).resolve()
            if not target.exists():
                broken.append(f"{document.relative_to(release_root)}: {match.group(1)}")
    assert broken == []


def test_runtime_resources_do_not_depend_on_checkout_directories(
    release_root: Path,
) -> None:
    for name in ("configs", "examples", "prompts", "final", "outputs"):
        assert not (release_root / name).exists()
