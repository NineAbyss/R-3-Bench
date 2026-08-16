from __future__ import annotations

import json
from pathlib import Path

from r3bench.commands.smoke import main


def test_complete_offline_acceptance_smoke(tmp_path: Path) -> None:
    output = tmp_path / "acceptance"
    assert main(["--output-dir", str(output)]) == 0
    summary = json.loads(
        (output / "acceptance_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "pass"
    assert len(summary["checks"]) == 15
    assert summary["network_called"] is False
    assert summary["external_verifier_started"] is False
