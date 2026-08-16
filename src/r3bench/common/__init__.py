"""Shared public schemas, data loading, generation, and scoring APIs."""

from r3bench.common.loader import load_contest_suites, load_single_problems
from r3bench.common.nl_runner import run_contest_nl, run_single_problem_nl, run_two_stage_nl
from r3bench.common.prompt import PromptTemplate, render_contest_prompt, render_single_prompt
from r3bench.common.profile_registry import (
    EvaluatorProfile,
    ModelProfile,
    ProfileError,
    ResolvedProfiles,
    load_evaluator_profiles,
    load_model_profiles,
    resolve_profiles_for_cell,
)
from r3bench.common.result_schema import (
    AttemptRecord,
    JudgeResultRecord,
    ParsedAnswerRecord,
    RunMetadata,
    RunSummary,
    UnifiedEvaluationSummary,
)
from r3bench.common.schema import ContestSuite, ProblemRecord
from r3bench.common.scoring_dispatch import (
    ProductionScoringRuntime,
    score_saved_outputs_cli,
)
from r3bench.common.settings import (
    BudgetUnit,
    EvaluationSetting,
    RuntimeKind,
    SettingConfig,
)

__all__ = [
    "AttemptRecord",
    "BudgetUnit",
    "ContestSuite",
    "EvaluationSetting",
    "EvaluatorProfile",
    "JudgeResultRecord",
    "ModelProfile",
    "ParsedAnswerRecord",
    "ProblemRecord",
    "ProductionScoringRuntime",
    "ProfileError",
    "PromptTemplate",
    "ResolvedProfiles",
    "RunMetadata",
    "RunSummary",
    "RuntimeKind",
    "SettingConfig",
    "UnifiedEvaluationSummary",
    "load_contest_suites",
    "load_evaluator_profiles",
    "load_model_profiles",
    "load_single_problems",
    "render_contest_prompt",
    "render_single_prompt",
    "resolve_profiles_for_cell",
    "run_contest_nl",
    "run_single_problem_nl",
    "run_two_stage_nl",
    "score_saved_outputs_cli",
]
