"""Deterministic prompt rendering over normalized public records."""

from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from r3bench.common.schema import ContestSuite, ProblemRecord
from r3bench.resource_paths import resolve_path


Visibility: TypeAlias = Literal["hidden", "labeled"]
PromptStage: TypeAlias = Literal["one_stage", "stage1", "stage2"]

_VISIBILITIES = frozenset({"hidden", "labeled"})
_TEMPLATE_FIELDS = frozenset(
    {
        "budget_tokens",
        "content",
        "domain",
        "mode",
        "num_problems",
        "numbered_content",
        "problem_id",
        "stage",
        "suite_id",
        "visibility",
    }
)


class PromptRenderError(ValueError):
    """Raised when a public prompt template or render request is invalid."""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A strict text template whose placeholders cannot access record fields."""

    text: str
    name: str = "inline"
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise PromptRenderError("prompt template text must be non-empty")
        if not isinstance(self.name, str) or not self.name.strip():
            raise PromptRenderError("prompt template name must be non-empty")
        fields: set[str] = set()
        try:
            parsed = string.Formatter().parse(self.text)
            for _, field_name, _, _ in parsed:
                if field_name is None:
                    continue
                if not field_name or any(token in field_name for token in (".", "[", "]")):
                    raise PromptRenderError(
                        f"unsupported prompt placeholder: {field_name!r}"
                    )
                fields.add(field_name)
        except ValueError as exc:
            raise PromptRenderError(f"invalid prompt template syntax: {exc}") from exc
        unsupported = fields - _TEMPLATE_FIELDS
        if unsupported:
            raise PromptRenderError(
                f"unsupported prompt placeholders: {sorted(unsupported)}"
            )
        if not fields.intersection({"content", "numbered_content"}):
            # Trace-only finalizers receive transient Stage 1 channels through
            # in-memory marker replacement after public template rendering.
            if "stage2" not in self.name:
                raise PromptRenderError(
                    "prompt template must include {content} or {numbered_content}"
                )

    @classmethod
    def from_file(cls, path: str | Path) -> "PromptTemplate":
        file_path = resolve_path(path)
        try:
            text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise PromptRenderError(f"cannot read prompt template {file_path}: {exc}") from exc
        return cls(text=text, name=file_path.stem, source_path=str(file_path))

    def render(self, **values: str) -> str:
        unknown = set(values) - _TEMPLATE_FIELDS
        if unknown:
            raise PromptRenderError(f"unknown render values: {sorted(unknown)}")
        context = {field: "" for field in _TEMPLATE_FIELDS}
        context.update(values)
        try:
            return self.text.format_map(context).rstrip() + "\n"
        except (KeyError, ValueError) as exc:
            raise PromptRenderError(f"cannot render prompt template: {exc}") from exc


def _validate_visibility(visibility: Visibility) -> Visibility:
    if visibility not in _VISIBILITIES:
        raise PromptRenderError(f"unsupported visibility: {visibility!r}")
    return visibility


def _problem_content(
    problem: ProblemRecord,
    *,
    visibility: Visibility,
    label: str | None,
) -> str:
    if problem.domain == "coding":
        title = str(problem.domain_payload.get("title") or "").strip()
        heading = f"## Problem {label}" if label is not None else "## Problem"
        if label is not None and title:
            heading = f"{heading}: {title}"
        lines = [heading]
        if visibility == "labeled":
            lines.append(f"Difficulty: {problem.difficulty}")
        source_url = problem.domain_payload.get("source_url")
        time_limit = problem.domain_payload.get("time_limit_ms")
        memory_limit = problem.domain_payload.get("memory_limit_mb")
        if source_url:
            lines.append(f"Problem link: {source_url}")
        if time_limit is not None:
            lines.append(f"Time limit: {time_limit} ms")
        if memory_limit is not None:
            lines.append(f"Memory limit: {memory_limit} MB")
        lines.extend(("", "### Statement", "", problem.problem_statement))
        return "\n".join(lines)

    if label is None:
        lines: list[str] = []
    else:
        lines = [f"## Problem {label}"]
    if visibility == "labeled":
        lines.append(f"Difficulty: {problem.difficulty}")
    if lines:
        lines.append("")
    lines.append(problem.problem_statement)
    return "\n".join(lines)


def _render(
    *,
    template: PromptTemplate,
    content: str,
    domain: str,
    mode: str,
    visibility: Visibility,
    stage: PromptStage,
    suite_id: str = "",
    problem_id: str = "",
    numbered_content: str = "",
    num_problems: int = 1,
    budget_tokens: int | None = None,
) -> str:
    return template.render(
        budget_tokens="" if budget_tokens is None else str(budget_tokens),
        content=content,
        domain=domain,
        mode=mode,
        numbered_content=numbered_content or content,
        num_problems=str(num_problems),
        visibility=visibility,
        stage=stage,
        suite_id=suite_id,
        problem_id=problem_id,
    )


def render_single_prompt(
    problem: ProblemRecord,
    template: PromptTemplate,
    visibility: Visibility,
    *,
    budget_tokens: int | None = None,
) -> str:
    """Render one problem without assigning a contest presentation label."""

    checked_visibility = _validate_visibility(visibility)
    content = _problem_content(problem, visibility=checked_visibility, label=None)
    return _render(
        template=template,
        content=content,
        domain=problem.domain,
        mode="single_problem",
        visibility=checked_visibility,
        stage="one_stage",
        suite_id=problem.suite_id,
        problem_id=problem.problem_id,
        numbered_content=content,
        num_problems=1,
        budget_tokens=budget_tokens,
    )


def render_contest_prompt(
    suite: ContestSuite,
    template: PromptTemplate,
    visibility: Visibility,
    *,
    budget_tokens: int | None = None,
) -> str:
    """Render a six-problem suite in its existing loader order."""

    checked_visibility = _validate_visibility(visibility)
    blocks = [
        _problem_content(
            problem,
            visibility=checked_visibility,
            label=problem.problem_label,
        )
        for problem in suite.problems
    ]
    numbered_blocks = [
        _problem_content(
            problem,
            visibility=checked_visibility,
            label=str(index),
        )
        for index, problem in enumerate(suite.problems, start=1)
    ]
    return _render(
        template=template,
        content="\n\n".join(blocks),
        numbered_content="\n\n".join(numbered_blocks),
        domain=suite.domain,
        mode="contest",
        visibility=checked_visibility,
        stage="one_stage",
        suite_id=suite.suite_id,
        num_problems=len(suite.problems),
        budget_tokens=budget_tokens,
    )


def render_two_stage_prompt(
    suite_or_problem: ContestSuite | ProblemRecord,
    stage: Literal["stage1", "stage2"],
    template: PromptTemplate,
    visibility: Visibility,
    *,
    budget_tokens: int | None = None,
) -> str:
    """Render the task portion of a two-stage prompt without invoking a model.

    Stage-1 model output is intentionally not embedded here. A future runner
    must pass that saved output to stage 2 as a separate message or input field.
    """

    if stage not in {"stage1", "stage2"}:
        raise PromptRenderError("two-stage rendering requires stage1 or stage2")
    checked_visibility = _validate_visibility(visibility)
    if isinstance(suite_or_problem, ContestSuite):
        blocks = [
            _problem_content(
                problem,
                visibility=checked_visibility,
                label=problem.problem_label,
            )
            for problem in suite_or_problem.problems
        ]
        numbered_blocks = [
            _problem_content(
                problem,
                visibility=checked_visibility,
                label=str(index),
            )
            for index, problem in enumerate(suite_or_problem.problems, start=1)
        ]
        return _render(
            template=template,
            content="\n\n".join(blocks),
            numbered_content="\n\n".join(numbered_blocks),
            domain=suite_or_problem.domain,
            mode="contest",
            visibility=checked_visibility,
            stage=stage,
            suite_id=suite_or_problem.suite_id,
            num_problems=len(suite_or_problem.problems),
            budget_tokens=budget_tokens,
        )

    content = _problem_content(
        suite_or_problem,
        visibility=checked_visibility,
        label=None,
    )
    return _render(
        template=template,
        content=content,
        domain=suite_or_problem.domain,
        mode="single_problem",
        visibility=checked_visibility,
        stage=stage,
        suite_id=suite_or_problem.suite_id,
        problem_id=suite_or_problem.problem_id,
        numbered_content=content,
        num_problems=1,
        budget_tokens=budget_tokens,
    )


__all__ = [
    "PromptRenderError",
    "PromptStage",
    "PromptTemplate",
    "Visibility",
    "render_contest_prompt",
    "render_single_prompt",
    "render_two_stage_prompt",
]
