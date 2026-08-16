"""Shared real-provider profile resolution and dry-run preparation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping

from r3bench.common.experiment import ExperimentConfig
from r3bench.common.nl_runner import (
    PreparedNLRequest,
    prepare_contest_requests,
    prepare_single_problem_requests,
    prepare_two_stage_requests,
)
from r3bench.common.profile_registry import (
    EvaluatorProfile,
    ModelProfile,
    ResolvedTransport,
    RunProfile,
    assert_no_model_specific_runner_fork,
    load_evaluator_profiles,
    load_model_profiles,
    load_run_profiles,
    resolve_evaluator_profile,
    resolve_model_profile,
    resolve_run_profile,
    resolve_transport_parameters,
    validate_run_profile_applicability,
)
from r3bench.common.result_schema import to_public_dict
from r3bench.common.two_stage_profile import TwoStageProtocol
from r3bench.providers.base import ProviderAdapter
from r3bench.providers.errors import ProviderConfigError
from r3bench.providers.openai_compatible import OpenAICompatibleAdapter
from r3bench.providers.registry import create_provider_adapter, load_provider_profile


_CREDENTIAL = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{12,}\b|\bhf_[A-Za-z0-9]{12,}\b|"
    r"\bolp_[A-Za-z0-9]{12,}\b|Bearer\s+[A-Za-z0-9._~+/-]{12,})"
)
_PRIVATE_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:home|mnt)/|/tmp/rbench(?:/|\b)")
_FORBIDDEN_KEYS = frozenset(
    {
        "authorization",
        "api_key",
        "access_token",
        "default_headers",
        "provider_headers",
        "request_id_from_provider",
        "trajectory",
        "raw_logs",
    }
)


@dataclass(frozen=True, slots=True)
class RealProviderContext:
    model_profile: ModelProfile
    evaluator_profile: EvaluatorProfile
    provider_profile: Mapping[str, Any]
    run_profile: RunProfile | None
    transport_profile: ResolvedTransport
    adapter: ProviderAdapter


def _scan_public(value: Any, path: str = "preview") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ProviderConfigError(f"{path} contains a forbidden output field")
            _scan_public(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_public(item, f"{path}[{index}]")
        return
    if isinstance(value, str):
        if _CREDENTIAL.search(value):
            raise ProviderConfigError(f"{path} contains a credential-like value")
        if _PRIVATE_PATH.search(value):
            raise ProviderConfigError(f"{path} contains a private machine path")


def resolve_real_provider_context(
    config: ExperimentConfig,
    *,
    provider_profile_path: str | Path,
    model_profiles_path: str | Path,
    evaluator_profiles_path: str | Path,
    model_key: str,
    dry_run: bool,
    transport: object | None = None,
    run_profiles_path: str | Path | None = None,
    run_profile_id: str | None = None,
    thinking_enabled_override: bool | None = None,
) -> RealProviderContext:
    """Resolve one safe adapter context without reading credentials."""

    models = load_model_profiles(model_profiles_path)
    evaluators = load_evaluator_profiles(evaluator_profiles_path)
    model = resolve_model_profile(model_key, models)
    if thinking_enabled_override is not None:
        model = replace(model, thinking_enabled=thinking_enabled_override)
    evaluator = resolve_evaluator_profile(model, evaluators)
    assert_no_model_specific_runner_fork(
        {
            "model_profile": model.to_dict(),
            "evaluator_profile": evaluator.to_dict(),
        }
    )
    provider_profile = load_provider_profile(provider_profile_path)
    if provider_profile["provider_name"] != model.provider_profile:
        raise ProviderConfigError("model and provider profiles do not match")
    if config.provider.name != model.provider_profile:
        raise ProviderConfigError("experiment config and model provider do not match")
    if config.provider.model != model_key:
        raise ProviderConfigError("experiment config model must equal the selected model key")
    if provider_profile.get("api_key_env") != model.api_key_env:
        raise ProviderConfigError("model and provider API-key environment names do not match")
    if (run_profiles_path is None) != (run_profile_id is None):
        raise ProviderConfigError(
            "run_profiles_path and run_profile_id must be supplied together"
        )
    run_profile: RunProfile | None = None
    if run_profile_id is not None:
        run_profile = resolve_run_profile(
            run_profile_id, load_run_profiles(run_profiles_path)  # type: ignore[arg-type]
        )
        validate_run_profile_applicability(
            run_profile,
            model_key=model.model_key,
            provider_profile=model.provider_profile,
            domain=config.domain,
            setting=config.setting,
        )
    transport_profile = resolve_transport_parameters(
        provider_profile,
        run_profile,
        config.transport_overrides,
    )
    adapter = create_provider_adapter(
        provider_profile,
        model,
        transport=transport,
        transport_config=transport_profile.values,
        dry_run=dry_run,
    )
    if not isinstance(adapter, ProviderAdapter):
        raise ProviderConfigError("real execution requires a ProviderAdapter")
    if not dry_run and isinstance(adapter, OpenAICompatibleAdapter):
        if not adapter.execution_ready:
            raise ProviderConfigError("real provider execution fields are unresolved")
    return RealProviderContext(
        model, evaluator, provider_profile, run_profile, transport_profile, adapter
    )


def _prepared_requests(
    config: ExperimentConfig,
    mode: Literal["single_problem", "contest"],
    limit: int | None,
) -> tuple[PreparedNLRequest, ...]:
    if mode == "single_problem":
        return prepare_single_problem_requests(config, limit=limit)
    return prepare_contest_requests(config, limit_suites=limit)


def write_real_provider_dry_run(
    config: ExperimentConfig,
    context: RealProviderContext,
    output_dir: str | Path,
    *,
    mode: Literal["single_problem", "contest"],
    limit: int | None = None,
) -> None:
    """Write a deterministic request preview without reading a key or calling transport."""

    prepared = _prepared_requests(config, mode, limit)
    if not isinstance(context.adapter, OpenAICompatibleAdapter):
        raise ProviderConfigError("dry-run payload preview requires openai_compatible")
    request_preview = {
        "schema_version": "1.0",
        "status": "dry_run",
        "adapter": context.adapter.adapter_name,
        "request_count": len(prepared),
        "requests": [
            {
                "request_id": row.request.request_id,
                "input_kind": row.request.input_kind,
                "input_sha256": row.request.input_sha256,
                "presentation": (
                    to_public_dict(row.presentation)
                    if row.presentation is not None
                    else None
                ),
                "payload": context.adapter.build_payload(
                    row.request, allow_unresolved=True
                ),
            }
            for row in prepared
        ],
    }
    provider_public = {
        "adapter": context.provider_profile["adapter"],
        "provider_name": context.provider_profile["provider_name"],
        "base_url": context.provider_profile["base_url"],
        "api_key_env": context.provider_profile["api_key_env"],
        "token_budget_field": context.provider_profile.get(
            "token_budget_field", "max_tokens"
        ),
        "extra_body": dict(context.provider_profile["extra_body"]),
    }
    resolved_experiment = {
        "evaluator_interface_version": "1.0",
        "profile_schema_version": "1.0",
        "status": "dry_run",
        "execution_ready": context.adapter.execution_ready,
        "domain": config.domain,
        "mode": config.mode,
        "visibility": config.visibility,
        "stage": config.stage,
        "data_source": config.data_source,
        "prompt_template": config.prompt_template_path,
        "presentation": config.presentation.to_dict(),
        "model_profile": context.model_profile.to_dict(),
        "evaluator_profile": context.evaluator_profile.to_dict(),
        "provider_profile": provider_public,
        "run_profile": (
            context.run_profile.to_dict() if context.run_profile is not None else None
        ),
        "transport_profile": context.transport_profile.to_dict(),
        "transport_resolution_precedence": [
            "cell_transport_override",
            "run_profile",
            "provider_profile_default",
        ],
    }
    summary = {
        "schema_version": "1.0",
        "status": "dry_run",
        "domain": config.domain,
        "mode": config.mode,
        "model_key": context.model_profile.model_key,
        "evaluator_profile": context.evaluator_profile.profile_key,
        "request_count": len(prepared),
        "network_called": False,
        "credential_read": False,
        "transport_execution_ready": context.transport_profile.execution_ready,
    }
    for value, name in (
        (request_preview, "request_preview"),
        (resolved_experiment, "resolved_experiment"),
        (summary, "run_summary"),
    ):
        _scan_public(value, name)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("request_preview.json", request_preview),
        ("resolved_experiment.json", resolved_experiment),
        ("run_summary.json", summary),
    ):
        (target / name).write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )


def write_two_stage_real_provider_dry_run(
    config_stage1: ExperimentConfig,
    config_stage2: ExperimentConfig,
    context_stage1: RealProviderContext,
    context_stage2: RealProviderContext,
    output_dir: str | Path,
    *,
    limit: int | None,
    protocol: TwoStageProtocol | None = None,
) -> None:
    """Preview both provider payloads without credentials or network calls."""

    prepared = prepare_two_stage_requests(
        config_stage1,
        config_stage2,
        limit=limit,
        protocol=protocol,
    )
    if not isinstance(context_stage1.adapter, OpenAICompatibleAdapter) or not isinstance(
        context_stage2.adapter, OpenAICompatibleAdapter
    ):
        raise ProviderConfigError("two-stage preview requires openai_compatible")
    requests: list[dict[str, Any]] = []
    for row in prepared:
        requests.extend(
            [
                {
                    "stage": "stage1",
                    "request_id": row.stage1_request.request_id,
                    "input_sha256": row.stage1_request.input_sha256,
                    "payload": context_stage1.adapter.build_payload(
                        row.stage1_request, allow_unresolved=True
                    ),
                },
                {
                    "stage": "stage2",
                    "request_id": row.stage2_request.request_id,
                    "parent_request_id": row.stage1_request.request_id,
                    "stage_input_sha256": row.stage1_output_sha256,
                    "input_sha256": row.stage2_request.input_sha256,
                    "payload": context_stage2.adapter.build_payload(
                        row.stage2_request, allow_unresolved=True
                    ),
                },
            ]
        )
    preview = {
        "schema_version": "1.0",
        "status": "dry_run",
        "two_stage": True,
        "item_count": len(prepared),
        "provider_request_count": len(requests),
        "stage2_protocol": (
            {
                "handoff_channels": list(protocol.handoff_channels),
                "prompt_assembly": protocol.prompt_assembly,
                "include_original_problems": protocol.include_original_problems,
            }
            if protocol is not None
            else None
        ),
        "presentation_orders": [
            to_public_dict(row.presentation)
            for row in prepared
            if row.presentation is not None
        ],
        "requests": requests,
    }
    resolved = {
        "schema_version": "1.0",
        "status": "dry_run",
        "domain": config_stage1.domain,
        "mode": config_stage1.mode,
        "presentation": config_stage1.presentation.to_dict(),
        "stage1": {
            "model_profile": context_stage1.model_profile.to_dict(),
            "evaluator_profile": context_stage1.evaluator_profile.to_dict(),
            "transport_profile": context_stage1.transport_profile.to_dict(),
            "max_tokens": config_stage1.max_tokens,
        },
        "stage2": {
            "model_profile": context_stage2.model_profile.to_dict(),
            "evaluator_profile": context_stage2.evaluator_profile.to_dict(),
            "transport_profile": context_stage2.transport_profile.to_dict(),
            "max_tokens": config_stage2.max_tokens,
        },
        "stage2_protocol": (
            {
                "handoff_channels": list(protocol.handoff_channels),
                "prompt_assembly": protocol.prompt_assembly,
                "include_original_problems": protocol.include_original_problems,
            }
            if protocol is not None
            else None
        ),
        "credential_read": False,
        "network_called": False,
    }
    summary = {
        "schema_version": "1.0",
        "status": "dry_run",
        "two_stage": True,
        "item_count": len(prepared),
        "provider_request_count": len(requests),
        "credential_read": False,
        "network_called": False,
    }
    for value, name in (
        (preview, "request_preview"),
        (resolved, "resolved_experiment"),
        (summary, "run_summary"),
    ):
        _scan_public(value, name)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("request_preview.json", preview),
        ("resolved_experiment.json", resolved),
        ("run_summary.json", summary),
    ):
        (target / name).write_text(
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
            encoding="utf-8",
        )


__all__ = [
    "RealProviderContext",
    "resolve_real_provider_context",
    "write_real_provider_dry_run",
    "write_two_stage_real_provider_dry_run",
]
