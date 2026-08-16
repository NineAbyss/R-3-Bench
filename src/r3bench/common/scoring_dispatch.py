#!/usr/bin/env python3
"""Shared saved-output scoring dispatch for both evaluator settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r3bench.abstract_reasoning.parser import extract_ar_answer
from r3bench.abstract_reasoning.scorer import (
    ARScorer,
    MockARScorer,
    ReasoningGymScorerAdapter,
    ReasoningGymScorerConfig,
)
from r3bench.coding.parser import extract_cpp_code
from r3bench.coding.scoring import score_coding_saved_outputs
from r3bench.coding.verifier import (
    CodingVerifier,
    LightCPVerifierAdapter,
    MockCodingVerifier,
)
from r3bench.coding.verifier import (
    LightCPVerifierConfig,
    LightCPVerifierHTTPExecutor,
)
from r3bench.common.profile_registry import (
    ModelProfile,
    load_run_profiles,
    resolve_run_profile,
    resolve_transport_parameters,
    validate_run_profile_applicability,
)
from r3bench.common.io import read_jsonl, read_jsonl_snapshot
from r3bench.common.loader import DataContractError, load_single_problems
from r3bench.common.schema import Domain, ProblemRecord
from r3bench.common.scorer_registry import (
    ScorerProfile,
    load_scorer_profiles,
    resolve_scorer_profile,
    scorer_profile_contract,
    scorer_profile_contract_sha256,
)
from r3bench.math.judge import (
    MathEquivalenceJudgeAdapter,
    MathEquivalenceJudgeConfig,
    MathJudge,
    MockMathJudge,
    ProviderMathJudgeTransport,
)
from r3bench.math.parser import extract_math_answer
from r3bench.providers.base import ProviderAdapter
from r3bench.providers.registry import create_provider_adapter, load_provider_profile
from r3bench.resource_paths import resolve_path


class SavedOutputError(ValueError):
    """Raised when a saved-prediction file violates the scoring contract."""


_MISSING_ANSWER_SENTINELS = frozenset({"no answer", "missing"})
_PROVENANCE_FIELDS = (
    "run_id",
    "request_id",
    "stage",
    "stage1_request_id",
    "stage2_request_id",
    "source_setting",
    "execution_id",
    "task_id",
    "model_key",
    "repeat_id",
)


@dataclass(frozen=True, slots=True)
class ProductionScoringRuntime:
    """Runtime-only integration values that are never serialized."""

    coding_judge_url: str | None = None
    coding_assets_root: str | None = None
    coding_verifier_root: str | None = None
    coding_verifier_config: str | None = None
    math_provider_profile: str = "configs/providers/deepseek_openai_compatible.yaml"
    math_model_profiles: str = "configs/model_profiles.yaml"
    math_run_profiles: str = "configs/run_profiles.yaml"
    math_run_profile: str | None = None
    math_source_mode: str = "single_problem"


def _saved_text(row: Mapping[str, Any], line_number: int) -> tuple[str | None, bool]:
    for field, already_parsed in (
        ("parsed_answer", True),
        ("prediction", False),
        ("response_text", False),
    ):
        if field in row:
            value = row[field]
            if field == "parsed_answer" and value is None:
                return None, True
            if not isinstance(value, str):
                raise SavedOutputError(
                    f"saved-output row {line_number} has a non-string {field}"
                )
            return value, already_parsed
    raise SavedOutputError(
        f"saved-output row {line_number} requires parsed_answer, prediction, or response_text"
    )


def _parse(domain: Domain, prediction: str) -> str | None:
    if domain == "coding":
        return extract_cpp_code(prediction)
    if domain == "math":
        return extract_math_answer(prediction)
    return extract_ar_answer(prediction)


def _parse_failure(
    domain: Domain, problem_id: str, *, evaluator: str, scoring_mode: str
) -> dict[str, Any]:
    return {
        "domain": domain,
        "problem_id": problem_id,
        "parse_status": "parse_error",
        "parsed_output": None,
        "judge_status": "not_judged",
        "correct": None if scoring_mode == "dry-run" else False,
        "score": None if scoring_mode == "dry-run" else 0.0,
        "verdict": "not_judged" if scoring_mode == "dry-run" else "parse_error",
        "detail": "no scoreable answer could be extracted",
        "evaluator": evaluator,
        "scoring_mode": scoring_mode,
    }


def _dry_run_result(
    domain: Domain, problem_id: str, parsed: str, *, evaluator: str
) -> dict[str, Any]:
    return {
        "domain": domain,
        "problem_id": problem_id,
        "parse_status": "parsed",
        "parsed_output": parsed,
        "judge_status": "not_judged",
        "correct": None,
        "score": None,
        "verdict": "not_judged",
        "detail": "scorer configuration validated; no external service was called",
        "evaluator": evaluator,
        "scoring_mode": "dry-run",
    }


def _judge_failure(
    domain: Domain,
    problem_id: str,
    parsed: str,
    *,
    evaluator: str,
    scoring_mode: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "domain": domain,
        "problem_id": problem_id,
        "parse_status": "parsed",
        "parsed_output": parsed,
        "judge_status": "judge_error",
        "correct": False,
        "score": 0.0,
        "verdict": "judge_error",
        "detail": "downstream judge or scorer failed for this problem",
        "error_type": type(error).__name__,
        "evaluator": evaluator,
        "scoring_mode": scoring_mode,
    }


def _judge_one(
    domain: Domain,
    problem: ProblemRecord,
    parsed: str,
    *,
    coding_verifier: CodingVerifier,
    math_judge: MathJudge,
    ar_scorer: ARScorer,
    evaluator: str = "mock",
    scoring_mode: str = "mock",
) -> dict[str, Any]:
    common = {
        "domain": domain,
        "problem_id": problem.problem_id,
        "parse_status": "parsed",
        "parsed_output": parsed,
        "judge_status": "judged",
        "evaluator": evaluator,
        "scoring_mode": scoring_mode,
    }
    if domain == "coding":
        result = coding_verifier.verify(problem, parsed)
        operational = {
            "not_configured",
            "service_unreachable",
            "assets_unavailable",
            "invalid_config",
            "verifier_error",
        }
        if result.status in operational:
            return {
                **common,
                "judge_status": "judge_error",
                "correct": False,
                "score": 0.0,
                "verdict": result.verdict,
                "detail": result.detail,
            }
        return {
            **common,
            "correct": result.accepted,
            "score": 1.0 if result.accepted else 0.0,
            "verdict": result.verdict,
            "detail": result.detail,
        }
    if domain == "math":
        result = math_judge.judge(problem, parsed)
        return {
            **common,
            "correct": result.correct,
            "score": 1.0 if result.correct else 0.0,
            "verdict": result.verdict,
            "detail": result.detail,
        }
    result = ar_scorer.score(problem, parsed)
    return {
        **common,
        "correct": result.correct,
        "score": result.score,
        "verdict": result.verdict,
        "detail": result.detail,
    }


def _validate_rows(
    domain: Domain,
    prediction_rows: list[dict[str, Any]],
    by_id: Mapping[str, ProblemRecord],
) -> list[tuple[ProblemRecord, str | None, dict[str, Any]]]:
    seen: set[str] = set()
    prepared: list[tuple[ProblemRecord, str | None, dict[str, Any]]] = []
    for line_number, row in enumerate(prediction_rows, start=1):
        row_domain = row.get("domain")
        if row_domain is not None and row_domain != domain:
            raise SavedOutputError(
                f"saved-output row {line_number} declares domain {row_domain!r}, "
                f"expected {domain!r}"
            )
        problem_id = row.get("problem_id")
        if not isinstance(problem_id, str) or not problem_id.strip():
            raise SavedOutputError(
                f"saved-output row {line_number} has no non-empty problem_id"
            )
        if problem_id in seen:
            raise SavedOutputError(
                f"duplicate prediction for problem_id {problem_id!r}"
            )
        seen.add(problem_id)
        if problem_id not in by_id:
            raise SavedOutputError(f"unknown problem_id in predictions: {problem_id!r}")
        prediction, already_parsed = _saved_text(row, line_number)
        parsed = (
            prediction.strip()
            if already_parsed and isinstance(prediction, str) and prediction.strip()
            else None
        )
        if (
            domain in {"math", "abstract_reasoning"}
            and parsed is not None
            and parsed.casefold() in _MISSING_ANSWER_SENTINELS
        ):
            parsed = None
        if not already_parsed:
            assert isinstance(prediction, str)
            parsed = _parse(domain, prediction)
        provenance: dict[str, Any] = {}
        for field in _PROVENANCE_FIELDS:
            if field not in row:
                continue
            value = row[field]
            invalid = (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                if field == "repeat_id"
                else value is not None and not isinstance(value, str)
            )
            if invalid:
                raise SavedOutputError(
                    f"saved-output row {line_number} has invalid {field}"
                )
            provenance[field] = value
        prepared.append((by_id[problem_id], parsed, provenance))
    return prepared


def judge_saved_outputs(
    *,
    domain: Domain,
    data_source: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    split: str = "test",
    strict: bool = True,
    coding_verifier: CodingVerifier | None = None,
    math_judge: MathJudge | None = None,
    ar_scorer: ARScorer | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible mock/injected scoring function."""

    problems = load_single_problems(domain, split, data_source, strict=strict)
    prepared = _validate_rows(
        domain,
        read_jsonl(predictions_path),
        {problem.problem_id: problem for problem in problems},
    )
    verifier = coding_verifier or MockCodingVerifier()
    judge = math_judge or MockMathJudge()
    scorer = ar_scorer or MockARScorer()
    results: list[dict[str, Any]] = []
    for problem, parsed, provenance in prepared:
        if parsed is None:
            results.append(
                {
                    **_parse_failure(
                        domain,
                        problem.problem_id,
                        evaluator="mock",
                        scoring_mode="mock",
                    ),
                    **provenance,
                }
            )
        else:
            results.append(
                {
                    **_judge_one(
                        domain,
                        problem,
                        parsed,
                        coding_verifier=verifier,
                        math_judge=judge,
                        ar_scorer=scorer,
                    ),
                    **provenance,
                }
            )
    _write_jsonl(Path(output_path), results)
    return results


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _input_data_source(rows: list[dict[str, Any]], explicit: str | None) -> str:
    if explicit:
        return explicit
    values = {row.get("data_source") for row in rows}
    if len(values) != 1 or not isinstance(next(iter(values)), str):
        raise SavedOutputError(
            "--data is required unless every saved-output row names one data_source"
        )
    value = next(iter(values))
    assert isinstance(value, str)
    if Path(value).is_absolute() or ".." in Path(value).parts:
        raise SavedOutputError("saved-output data_source must be a safe relative path")
    return value


def _is_synthetic_fixture(rows: list[dict[str, Any]], data_source: str) -> bool:
    return (
        bool(rows)
        and all(row.get("synthetic") is True for row in rows)
        and Path(data_source).parts[:2] == ("examples", "data")
    )


def _profile_for_cli(
    domain: Domain,
    profile_path: str | None,
    scorer_key: str | None,
    scoring_mode: str,
) -> ScorerProfile | None:
    if scoring_mode == "mock" and profile_path is None and scorer_key is None:
        return None
    if not profile_path or not scorer_key:
        raise SavedOutputError(
            "dry-run and production scoring require --scorer-profile and --scorer-key"
        )
    return resolve_scorer_profile(
        scorer_key, load_scorer_profiles(profile_path), domain=domain
    )


def _production_backends(
    domain: Domain,
    profile: ScorerProfile,
    runtime: ProductionScoringRuntime,
) -> tuple[CodingVerifier, MathJudge, ARScorer]:
    if profile.unresolved_fields:
        raise SavedOutputError(
            "production scorer profile has unresolved fields: "
            + ", ".join(profile.unresolved_fields)
        )
    if domain == "coding":
        if not runtime.coding_judge_url or not runtime.coding_assets_root:
            raise SavedOutputError(
                "production Coding scoring requires --coding-judge-url and "
                "--coding-assets-root"
            )
        config = LightCPVerifierConfig(
            judge_url=runtime.coding_judge_url,
            problem_assets_root=Path(runtime.coding_assets_root),
            verifier_root=(
                Path(runtime.coding_verifier_root)
                if runtime.coding_verifier_root
                else None
            ),
        )
        return (
            LightCPVerifierAdapter(config, executor=LightCPVerifierHTTPExecutor()),
            MockMathJudge(),
            MockARScorer(),
        )
    if domain == "math":
        if runtime.math_source_mode not in {"single_problem", "contest"}:
            raise SavedOutputError("math_source_mode must be single_problem or contest")
        model_key = str(profile.config["judge_model"])
        provider_profile = load_provider_profile(runtime.math_provider_profile)
        expected_provider = str(profile.config["provider_profile"])
        if provider_profile.get("provider_name") != expected_provider:
            raise SavedOutputError(
                "Math judge provider profile does not match scorer profile"
            )
        expected_run = str(
            profile.config[
                "contest_run_profile"
                if runtime.math_source_mode == "contest"
                else "single_run_profile"
            ]
        )
        if (
            runtime.math_run_profile is not None
            and runtime.math_run_profile != expected_run
        ):
            raise SavedOutputError(
                "Math judge run profile differs from the released scorer contract"
            )
        configured_run = expected_run
        run_profile = resolve_run_profile(
            configured_run, load_run_profiles(runtime.math_run_profiles)
        )
        validate_run_profile_applicability(
            run_profile,
            model_key=model_key,
            provider_profile=expected_provider,
            domain="math",
            setting="tool_free",
        )
        transport_parameters = resolve_transport_parameters(
            provider_profile, run_profile
        )
        if not transport_parameters.execution_ready:
            raise SavedOutputError("Math judge transport profile is unresolved")
        judge_model = ModelProfile(
            model_key=model_key,
            display_name="DeepSeek V4 Flash",
            evaluator_profile="deepseek_shared",
            provider_profile=expected_provider,
            public_model_id=model_key,
            api_key_env=str(profile.config["api_key_env"]),
            thinking_enabled="unresolved",
            reasoning_effort=None,
            temperature=float(profile.config["temperature"]),
            top_p=float(profile.config["top_p"]),
            notes="Scoring-only Math equivalence judge from the paper protocol.",
            status="release",
            requires_owner_approval=False,
        )
        provider = cast(
            ProviderAdapter,
            create_provider_adapter(
                provider_profile,
                judge_model,
                transport_config=transport_parameters.values,
            ),
        )
        prompt_path = resolve_path(str(profile.config["judge_prompt"]))
        prompt = prompt_path.read_text(encoding="utf-8")
        judge = MathEquivalenceJudgeAdapter(
            MathEquivalenceJudgeConfig(
                judge_model=model_key,
                prompt_template=prompt,
                api_key_env=str(profile.config["api_key_env"]),
                response_format=str(profile.config["response_format"]),
            ),
            transport=ProviderMathJudgeTransport(
                provider,
                max_tokens=int(profile.config["max_tokens"]),
                temperature=float(profile.config["temperature"]),
            ),
        )
        return MockCodingVerifier(), judge, MockARScorer()
    config = ReasoningGymScorerConfig(
        reasoning_gym_version=str(profile.config["reasoning_gym_version"]),
        reasoning_gym_revision=str(profile.config["reasoning_gym_revision"]),
        module_name=str(profile.config["module_name"]),
    )
    return MockCodingVerifier(), MockMathJudge(), ReasoningGymScorerAdapter(config)


def score_saved_outputs_cli(
    *,
    domain: Domain,
    input_path: str | Path,
    output_dir: str | Path,
    data_source: str | None,
    split: str,
    strict: bool,
    scoring_mode: str,
    profile_path: str | None,
    scorer_key: str | None,
    production_runtime: ProductionScoringRuntime | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if scoring_mode not in {"mock", "production", "dry-run"}:
        raise SavedOutputError(
            "scoring_mode must be 'mock', 'production', or 'dry-run'"
        )
    input_bytes, rows = read_jsonl_snapshot(input_path)
    source = _input_data_source(rows, data_source)
    effective_strict = strict and not _is_synthetic_fixture(rows, source)
    problems = load_single_problems(domain, split, source, strict=effective_strict)
    prepared = _validate_rows(domain, rows, {p.problem_id: p for p in problems})
    profile = _profile_for_cli(domain, profile_path, scorer_key, scoring_mode)
    evaluator = profile.profile_id if profile else "mock"

    external_service_called = False
    external_scorer_called = False
    external_service_call_count = 0
    external_scorer_call_count = 0
    if scoring_mode == "dry-run":
        results = [
            (
                _parse_failure(
                    domain,
                    problem.problem_id,
                    evaluator=evaluator,
                    scoring_mode="dry-run",
                )
                if parsed is None
                else _dry_run_result(
                    domain, problem.problem_id, parsed, evaluator=evaluator
                )
            )
            | provenance
            for problem, parsed, provenance in prepared
        ]
    else:
        if scoring_mode == "production":
            assert profile is not None
            verifier, judge, scorer = _production_backends(
                domain, profile, production_runtime or ProductionScoringRuntime()
            )
        else:
            verifier, judge, scorer = (
                MockCodingVerifier(),
                MockMathJudge(),
                MockARScorer(),
            )
        results = []
        for problem, parsed, provenance in prepared:
            if parsed is None:
                results.append(
                    {
                        **_parse_failure(
                            domain,
                            problem.problem_id,
                            evaluator=evaluator,
                            scoring_mode=scoring_mode,
                        ),
                        **provenance,
                    }
                )
            else:
                if scoring_mode == "production":
                    external_scorer_called = True
                    external_scorer_call_count += 1
                    if domain in {"coding", "math"}:
                        external_service_called = True
                        external_service_call_count += 1
                try:
                    judged = _judge_one(
                        domain,
                        problem,
                        parsed,
                        coding_verifier=verifier,
                        math_judge=judge,
                        ar_scorer=scorer,
                        evaluator=evaluator,
                        scoring_mode=scoring_mode,
                    )
                except Exception as exc:
                    if scoring_mode != "production":
                        raise
                    judged = _judge_failure(
                        domain,
                        problem.problem_id,
                        parsed,
                        evaluator=evaluator,
                        scoring_mode=scoring_mode,
                        error=exc,
                    )
                results.append({**judged, **provenance})

    target = Path(output_dir)
    identities = {
        problem.problem_id: {
            "suite_id": problem.suite_id,
            "problem_index": problem.problem_index,
            "problem_label": "ABCDEF"[problem.problem_index - 1],
        }
        for problem, _, _ in prepared
    }
    results = [{**row, **identities[str(row["problem_id"])]} for row in results]
    results_sha256 = _write_jsonl(target / "judge_results.jsonl", results)
    judged = sum(row["judge_status"] == "judged" for row in results)
    summary = {
        "schema_version": "1.0",
        "domain": domain,
        "scoring_mode": scoring_mode,
        "scorer_profile": evaluator,
        "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "results_sha256": results_sha256,
        "input_count": len(rows),
        "parsed_count": sum(row["parse_status"] == "parsed" for row in results),
        "judged_count": judged,
        "correct_count": sum(row["correct"] is True for row in results),
        "status": "dry_run" if scoring_mode == "dry-run" else "complete",
        "generation_provider_called": False,
        "external_service_called": external_service_called,
        "external_service_call_count": external_service_call_count,
        "external_scorer_called": external_scorer_called,
        "external_scorer_call_count": external_scorer_call_count,
        "profile_requires_owner_approval": (
            profile.requires_owner_approval if profile else False
        ),
        "profile_unresolved_fields": list(profile.unresolved_fields) if profile else [],
        "scorer_contract": scorer_profile_contract(profile) if profile else None,
        "scorer_contract_sha256": (
            scorer_profile_contract_sha256(profile) if profile else None
        ),
    }
    target.mkdir(parents=True, exist_ok=True)
    (target / "scoring_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain", required=True, choices=("coding", "math", "abstract_reasoning")
    )
    parser.add_argument("--data", help="Public JSONL file or data root")
    parser.add_argument("--predictions", "--input", dest="input_path", required=True)
    parser.add_argument("--output", help="Legacy result JSONL output")
    parser.add_argument("--output-dir", help="Standard scoring output directory")
    parser.add_argument("--split", default="test")
    parser.add_argument("--relaxed", action="store_true")
    parser.add_argument("--backend", choices=("mock",), default="mock")
    parser.add_argument("--scorer-profile")
    parser.add_argument("--scorer-key")
    parser.add_argument(
        "--scoring-mode", choices=("mock", "production", "dry-run"), default="mock"
    )
    parser.add_argument("--coding-judge-url")
    parser.add_argument("--coding-assets-root")
    parser.add_argument("--coding-verifier-root")
    parser.add_argument(
        "--coding-verifier-config",
        help="User-specific LightCPVerifier config for saved Coding outputs",
    )
    parser.add_argument(
        "--math-provider-profile",
        default="configs/providers/deepseek_openai_compatible.yaml",
    )
    parser.add_argument("--math-model-profiles", default="configs/model_profiles.yaml")
    parser.add_argument("--math-run-profiles", default="configs/run_profiles.yaml")
    parser.add_argument("--math-run-profile")
    parser.add_argument(
        "--math-source-mode",
        choices=("single_problem", "contest"),
        default="single_problem",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.output_dir:
            if args.domain == "coding" and args.coding_verifier_config:
                coding_profile = _profile_for_cli(
                    "coding",
                    args.scorer_profile,
                    args.scorer_key,
                    args.scoring_mode,
                )
                results, summary = score_coding_saved_outputs(
                    predictions_path=args.input_path,
                    data_source=_input_data_source(
                        read_jsonl(args.input_path), args.data
                    ),
                    output_dir=args.output_dir,
                    mode=args.scoring_mode,
                    verifier_config_path=args.coding_verifier_config,
                    split=args.split,
                    strict=not args.relaxed,
                    scorer_profile=coding_profile,
                )
                print(
                    f"wrote {len(results)} saved-output records to {args.output_dir}; "
                    f"status={summary['status']}"
                )
                return 0
            results, summary = score_saved_outputs_cli(
                domain=args.domain,
                input_path=args.input_path,
                output_dir=args.output_dir,
                data_source=args.data,
                split=args.split,
                strict=not args.relaxed,
                scoring_mode=args.scoring_mode,
                profile_path=args.scorer_profile,
                scorer_key=args.scorer_key,
                production_runtime=ProductionScoringRuntime(
                    coding_judge_url=args.coding_judge_url,
                    coding_assets_root=args.coding_assets_root,
                    coding_verifier_root=args.coding_verifier_root,
                    coding_verifier_config=args.coding_verifier_config,
                    math_provider_profile=args.math_provider_profile,
                    math_model_profiles=args.math_model_profiles,
                    math_run_profiles=args.math_run_profiles,
                    math_run_profile=args.math_run_profile,
                    math_source_mode=args.math_source_mode,
                ),
            )
            print(
                f"wrote {len(results)} saved-output records to {args.output_dir}; "
                f"status={summary['status']}"
            )
            return 0
        if args.scoring_mode != "mock" or args.scorer_profile or args.scorer_key:
            raise SavedOutputError(
                "scorer profiles and non-mock modes require --output-dir"
            )
        if not args.output or not args.data:
            raise SavedOutputError(
                "legacy mode requires --data and --output; new mode requires --output-dir"
            )
        results = judge_saved_outputs(
            domain=args.domain,
            data_source=args.data,
            predictions_path=args.input_path,
            output_path=args.output,
            split=args.split,
            strict=not args.relaxed,
        )
        print(f"wrote {len(results)} offline mock judgments to {args.output}")
        return 0
    except (DataContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"saved-output scoring failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
