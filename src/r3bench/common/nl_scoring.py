"""Campaign-level discovery and status records for saved-output scoring."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


class NLScoringCampaignError(ValueError):
    """Raised when generation outputs cannot form a safe scoring campaign."""


@dataclass(frozen=True, slots=True)
class ScoringUnit:
    unit_id: str
    cell_id: str
    domain: str
    experiment_role: str
    budget: int | None
    data_source: str
    scorer_profile: str
    predictions_path: str
    output_dir: str
    status: str
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_document(path: str | Path) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NLScoringCampaignError(f"cannot read campaign plan: {path}") from exc
    if not isinstance(value, Mapping):
        raise NLScoringCampaignError("campaign plan must be a JSON object")
    return value


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise NLScoringCampaignError(
            "campaign scoring paths must remain under the release root"
        ) from exc


def discover_scoring_units(
    campaign_plan_path: str | Path,
    *,
    release_root: str | Path,
    output_dir: str | Path,
) -> tuple[ScoringUnit, ...]:
    """Discover generated prediction files without calling a scorer."""

    root = Path(release_root)
    output_root = root / output_dir
    document = _read_document(campaign_plan_path)
    cells = document.get("cells")
    if not isinstance(cells, list) or not cells:
        raise NLScoringCampaignError("campaign plan contains no cells")
    units: list[ScoringUnit] = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise NLScoringCampaignError("campaign cell must be an object")
        cell_id = str(cell.get("cell_id", ""))
        domain = str(cell.get("domain", ""))
        role = str(cell.get("experiment_role", ""))
        data_source = str(cell.get("data_source", ""))
        scorer = str(cell.get("scorer_profile", ""))
        generation_dir = root / str(cell.get("output_dir", ""))
        if not all((cell_id, domain, role, data_source, scorer)):
            raise NLScoringCampaignError("campaign cell is missing scoring metadata")

        candidates: list[tuple[int | None, Path]] = []
        if role == "single_problem_response_curve":
            if generation_dir.is_dir():
                for budget_dir in sorted(generation_dir.glob("budget_*")):
                    suffix = budget_dir.name.removeprefix("budget_")
                    if suffix.isdigit() and int(suffix) > 0:
                        candidates.append(
                            (int(suffix), budget_dir / "parsed_answers.jsonl")
                        )
            if not candidates:
                candidates.append((None, generation_dir / "parsed_answers.jsonl"))
        else:
            candidates.append((None, generation_dir / "parsed_answers.jsonl"))

        for budget, predictions in candidates:
            suffix = f"_budget_{budget}" if budget is not None else ""
            unit_id = f"{cell_id}{suffix}"
            target = output_root / "cells" / unit_id
            exists = predictions.is_file()
            units.append(
                ScoringUnit(
                    unit_id=unit_id,
                    cell_id=cell_id,
                    domain=domain,
                    experiment_role=role,
                    budget=budget,
                    data_source=data_source,
                    scorer_profile=scorer,
                    predictions_path=_relative(predictions, root),
                    output_dir=_relative(target, root),
                    status="pending" if exists else "generation_missing",
                    skip_reason=None if exists else "parsed_answers_not_found",
                )
            )
    if not units:
        raise NLScoringCampaignError("no scoring units were discovered")
    return tuple(units)


def write_scoring_plan(
    path: str | Path,
    units: Iterable[ScoringUnit],
    *,
    scoring_mode: str,
) -> None:
    rows = tuple(units)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "scoring_mode": scoring_mode,
                "generation_provider_called": False,
                "units": [row.to_dict() for row in rows],
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "NLScoringCampaignError",
    "ScoringUnit",
    "discover_scoring_units",
    "write_scoring_plan",
]
