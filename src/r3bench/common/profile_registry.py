"""Safe, provider-neutral evaluator and model profile resolution."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlparse

from r3bench.common.config import load_config
from r3bench.resource_paths import resource_path


DEFAULT_EVALUATOR_PROFILES_PATH = resource_path(
    "configs", "evaluator_profiles.yaml"
)
DEFAULT_MODEL_PROFILES_PATH = resource_path("configs", "model_profiles.yaml")
DEFAULT_RUN_PROFILES_PATH = resource_path("configs", "run_profiles.yaml")

QWEN_SHARED_MODELS = frozenset(
    {"qwen3.7-max", "glm-5.2", "hunyuan-3", "gpt-5.5", "claude-opus-4.8"}
)
DEEPSEEK_SHARED_MODELS = frozenset(
    {"deepseek-chat", "deepseek-reasoner", "deepseek-v4-pro"}
)
FIRST_RELEASE_MODEL_INVENTORY = QWEN_SHARED_MODELS | DEEPSEEK_SHARED_MODELS

_DOMAINS = frozenset({"coding", "math", "abstract_reasoning"})
_MODES = frozenset({"single_problem", "contest"})
_SETTINGS = frozenset({"tool_free", "agentic"})
_PROFILE_KEY = re.compile(r"^[a-z][a-z0-9_]*(?:[.-][a-z0-9_]+)*$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_CREDENTIAL = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{12,}\b|\bhf_[A-Za-z0-9]{12,}\b|"
    r"\bolp_[A-Za-z0-9]{12,}\b|Bearer\s+[A-Za-z0-9._~+/-]{12,})"
)
_ENV_PATH = re.compile(r"(?:^|[/\\])\.env(?:$|[/\\])")
_PRIVATE_HOST = re.compile(r"(?i)(?:^|\.)(?:localhost|internal|local)$")
_UNSAFE_KEYS = frozenset(
    {
        "api_key",
        "access_token",
        "password",
        "authorization",
        "provider_headers",
        "trajectory_path",
        "raw_log_path",
    }
)
_EVALUATOR_ROOT_FIELDS = frozenset(
    {
        "schema_version",
        "evaluator_interface_version",
        "profile_schema_version",
        "status",
        "profiles",
    }
)
_EVALUATOR_FIELDS = frozenset(
    {
        "description",
        "runner",
        "prompt_renderer",
        "one_stage_orchestration",
        "two_stage_orchestration",
        "parser_dispatch",
        "result_schema",
        "supports_domains",
        "supports_modes",
        "supports_settings",
        "notes",
    }
)
_MODEL_ROOT_FIELDS = frozenset(
    {"schema_version", "profile_schema_version", "status", "models"}
)
_MODEL_FIELDS = frozenset(
    {
        "model_key",
        "display_name",
        "evaluator_profile",
        "provider_profile",
        "public_model_id",
        "api_key_env",
        "thinking_enabled",
        "reasoning_effort",
        "temperature",
        "top_p",
        "notes",
        "status",
        "requires_owner_approval",
    }
)
_RUN_ROOT_FIELDS = frozenset({"schema_version", "status", "profiles"})
_RUN_FIELDS = frozenset(
    {
        "profile_id",
        "provider_profile",
        "applicable_models",
        "applicable_domains",
        "applicable_settings",
        "timeout_seconds",
        "max_retries",
        "retry_backoff_seconds",
        "retry_backoff_mode",
        "streaming",
        "request_safety_limits",
        "provenance_source",
        "status",
        "requires_owner_approval",
        "notes",
    }
)
TRANSPORT_FIELD_ORDER = (
    "timeout_seconds",
    "max_retries",
    "retry_backoff_seconds",
    "retry_backoff_mode",
    "streaming",
    "request_safety_limits",
)
TRANSPORT_FIELDS = frozenset(TRANSPORT_FIELD_ORDER)
_SHARED_COMPONENT_FIELDS = (
    "runner",
    "prompt_renderer",
    "one_stage_orchestration",
    "two_stage_orchestration",
    "parser_dispatch",
    "result_schema",
)


class ProfileError(ValueError):
    """Raised when a profile document or resolution is unsafe or inconsistent."""


def _freeze(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return MappingProxyType(dict(value))


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{field} must be a non-empty string")
    return value


def _profile_key(value: object, field: str) -> str:
    text = _text(value, field)
    if not _PROFILE_KEY.fullmatch(text):
        raise ProfileError(f"{field} is not a valid profile key")
    return text


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ProfileError(f"{field} must be a non-empty string array")
    return tuple(_text(item, f"{field}[{index}]") for index, item in enumerate(value))


def _is_private_url(value: str) -> bool:
    if value.startswith("<") and value.endswith(">"):
        return False
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname
    if _PRIVATE_HOST.search(host):
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return False


def _scan_safe(value: Any, path: str = "profile") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.lower() in _UNSAFE_KEYS:
                raise ProfileError(f"{path} contains forbidden field {key!r}")
            _scan_safe(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_safe(item, f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    if _CREDENTIAL.search(value):
        raise ProfileError(f"{path} contains a raw API-key-like value")
    if Path(value).is_absolute() or _WINDOWS_ABSOLUTE.match(value):
        raise ProfileError(f"{path} contains a private absolute path")
    if _ENV_PATH.search(value):
        raise ProfileError(f"{path} contains a forbidden .env path")
    if _is_private_url(value):
        raise ProfileError(f"{path} contains a private provider endpoint")


def _setting_value(value: object, field: str) -> bool | str:
    if isinstance(value, bool) or value == "unresolved":
        return value
    raise ProfileError(f"{field} must be true, false, or unresolved")


def _sampling_value(value: object, field: str) -> float | None | str:
    if value is None or value == "unresolved":
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{field} must be numeric, null, or unresolved")
    numeric = float(value)
    if field.endswith("top_p") and not (0.0 <= numeric <= 1.0):
        raise ProfileError(f"{field} must be within [0, 1]")
    if field.endswith("temperature") and numeric < 0:
        raise ProfileError(f"{field} must be non-negative")
    return numeric


def _transport_value(field: str, value: object, path: str) -> Any:
    if value == "unresolved":
        return value
    if field == "timeout_seconds":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ProfileError(f"{path} must be positive or unresolved")
        return value
    if field == "max_retries":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProfileError(f"{path} must be non-negative or unresolved")
        return value
    if field == "retry_backoff_seconds":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ProfileError(f"{path} must be non-negative or unresolved")
        return float(value)
    if field == "retry_backoff_mode":
        if value not in {"fixed", "exponential", "none"}:
            raise ProfileError(
                f"{path} must be fixed, exponential, none, or unresolved"
            )
        return value
    if field == "streaming":
        if not isinstance(value, bool):
            raise ProfileError(f"{path} must be boolean or unresolved")
        return value
    if field == "request_safety_limits":
        if not isinstance(value, Mapping):
            raise ProfileError(f"{path} must be a mapping or unresolved")
        _scan_safe(value, path)
        return dict(value)
    raise ProfileError(f"unsupported transport field: {field}")


def validate_transport_mapping(
    value: Mapping[str, Any] | None,
    *,
    path: str = "transport",
    require_all: bool = False,
) -> dict[str, Any]:
    """Validate a transport layer without supplying implicit defaults."""

    if value is None:
        if require_all:
            raise ProfileError(f"{path} is required")
        return {}
    if not isinstance(value, Mapping):
        raise ProfileError(f"{path} must be a mapping")
    fields = set(value)
    if not fields.issubset(TRANSPORT_FIELDS):
        raise ProfileError(f"{path} contains unsupported fields")
    if require_all and fields != TRANSPORT_FIELDS:
        raise ProfileError(f"{path} must define every transport field")
    return {
        field: _transport_value(field, item, f"{path}.{field}")
        for field, item in value.items()
    }


@dataclass(frozen=True, slots=True)
class EvaluatorProfile:
    profile_key: str
    description: str
    runner: str
    prompt_renderer: str
    one_stage_orchestration: str
    two_stage_orchestration: str
    parser_dispatch: str
    result_schema: str
    supports_domains: tuple[str, ...]
    supports_modes: tuple[str, ...]
    supports_settings: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_key": self.profile_key,
            "description": self.description,
            "runner": self.runner,
            "prompt_renderer": self.prompt_renderer,
            "one_stage_orchestration": self.one_stage_orchestration,
            "two_stage_orchestration": self.two_stage_orchestration,
            "parser_dispatch": self.parser_dispatch,
            "result_schema": self.result_schema,
            "supports_domains": list(self.supports_domains),
            "supports_modes": list(self.supports_modes),
            "supports_settings": list(self.supports_settings),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model_key: str
    display_name: str
    evaluator_profile: str
    provider_profile: str
    public_model_id: str
    api_key_env: str
    thinking_enabled: bool | str
    reasoning_effort: str | None
    temperature: float | None | str
    top_p: float | None | str
    notes: str
    status: str
    requires_owner_approval: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "display_name": self.display_name,
            "evaluator_profile": self.evaluator_profile,
            "provider_profile": self.provider_profile,
            "public_model_id": self.public_model_id,
            "api_key_env": self.api_key_env,
            "thinking_enabled": self.thinking_enabled,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "notes": self.notes,
            "status": self.status,
            "requires_owner_approval": self.requires_owner_approval,
        }


@dataclass(frozen=True, slots=True)
class RunProfile:
    profile_id: str
    provider_profile: str
    applicable_models: tuple[str, ...]
    applicable_domains: tuple[str, ...]
    applicable_settings: tuple[str, ...]
    timeout_seconds: int | float | str
    max_retries: int | str
    retry_backoff_seconds: float | str
    retry_backoff_mode: str
    streaming: bool | str
    request_safety_limits: Mapping[str, Any] | str
    provenance_source: str
    status: str
    requires_owner_approval: bool
    notes: str

    def __post_init__(self) -> None:
        if isinstance(self.request_safety_limits, Mapping):
            object.__setattr__(
                self,
                "request_safety_limits",
                MappingProxyType(dict(self.request_safety_limits)),
            )

    def transport_dict(self) -> dict[str, Any]:
        return {
            field: (
                dict(value)
                if isinstance((value := getattr(self, field)), Mapping)
                else value
            )
            for field in TRANSPORT_FIELD_ORDER
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "provider_profile": self.provider_profile,
            "applicable_models": list(self.applicable_models),
            "applicable_domains": list(self.applicable_domains),
            "applicable_settings": list(self.applicable_settings),
            **self.transport_dict(),
            "provenance_source": self.provenance_source,
            "status": self.status,
            "requires_owner_approval": self.requires_owner_approval,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ResolvedTransport:
    values: Mapping[str, Any]
    source_by_field: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(
            self, "source_by_field", MappingProxyType(dict(self.source_by_field))
        )

    @property
    def unresolved_fields(self) -> tuple[str, ...]:
        return tuple(
            sorted(field for field, value in self.values.items() if value == "unresolved")
        )

    @property
    def execution_ready(self) -> bool:
        return not self.unresolved_fields

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": {
                key: dict(value) if isinstance(value, Mapping) else value
                for key, value in self.values.items()
            },
            "source_by_field": dict(self.source_by_field),
            "unresolved_fields": list(self.unresolved_fields),
            "execution_ready": self.execution_ready,
        }


@dataclass(frozen=True, slots=True)
class ResolvedProfiles:
    model_profile: ModelProfile
    evaluator_profile: EvaluatorProfile
    provider_profile: Mapping[str, Any] | None
    budget_profile: Mapping[str, Any] | None
    unresolved_fields: tuple[str, ...]
    requires_owner_approval: bool
    run_profile: RunProfile | None = None
    transport_profile: ResolvedTransport | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_profile", _freeze(self.provider_profile))
        object.__setattr__(self, "budget_profile", _freeze(self.budget_profile))

    @property
    def resolved_evaluator_profile(self) -> str:
        return self.evaluator_profile.profile_key

    @property
    def runner(self) -> str:
        return self.evaluator_profile.runner

    def to_dict(self) -> dict[str, Any]:
        result = {
            "resolved_evaluator_profile": self.resolved_evaluator_profile,
            "runner": self.runner,
            "provider_profile": self.model_profile.provider_profile,
            "model_profile": self.model_profile.to_dict(),
            "evaluator_profile": self.evaluator_profile.to_dict(),
            "unresolved_fields": list(self.unresolved_fields),
            "requires_owner_approval": self.requires_owner_approval,
        }
        if self.provider_profile is not None:
            result["provider_profile_config"] = dict(self.provider_profile)
        if self.budget_profile is not None:
            result["budget_profile"] = dict(self.budget_profile)
        if self.run_profile is not None:
            result["run_profile"] = self.run_profile.to_dict()
        if self.transport_profile is not None:
            result["transport_profile"] = self.transport_profile.to_dict()
        _scan_safe(result, "resolved")
        return result


def _validate_evaluator_profile(profile_key: str, value: object) -> EvaluatorProfile:
    if not isinstance(value, Mapping) or set(value) != _EVALUATOR_FIELDS:
        raise ProfileError(f"evaluator profile {profile_key!r} has invalid fields")
    domains = _string_tuple(value["supports_domains"], f"{profile_key}.supports_domains")
    modes = _string_tuple(value["supports_modes"], f"{profile_key}.supports_modes")
    settings = _string_tuple(value["supports_settings"], f"{profile_key}.supports_settings")
    if not set(domains).issubset(_DOMAINS):
        raise ProfileError(f"{profile_key} has an unsupported domain")
    if not set(modes).issubset(_MODES):
        raise ProfileError(f"{profile_key} has an unsupported mode")
    if not set(settings).issubset(_SETTINGS):
        raise ProfileError(f"{profile_key} has an unsupported setting")
    return EvaluatorProfile(
        profile_key=profile_key,
        description=_text(value["description"], f"{profile_key}.description"),
        runner=_profile_key(value["runner"], f"{profile_key}.runner"),
        prompt_renderer=_profile_key(
            value["prompt_renderer"], f"{profile_key}.prompt_renderer"
        ),
        one_stage_orchestration=_profile_key(
            value["one_stage_orchestration"],
            f"{profile_key}.one_stage_orchestration",
        ),
        two_stage_orchestration=_profile_key(
            value["two_stage_orchestration"],
            f"{profile_key}.two_stage_orchestration",
        ),
        parser_dispatch=_profile_key(
            value["parser_dispatch"], f"{profile_key}.parser_dispatch"
        ),
        result_schema=_profile_key(
            value["result_schema"], f"{profile_key}.result_schema"
        ),
        supports_domains=domains,
        supports_modes=modes,
        supports_settings=settings,
        notes=_string_tuple(value["notes"], f"{profile_key}.notes"),
    )


def load_evaluator_profiles(
    path: str | Path = DEFAULT_EVALUATOR_PROFILES_PATH,
) -> Mapping[str, EvaluatorProfile]:
    """Load and validate evaluator profiles without executing them."""

    document = load_config(path)
    _scan_safe(document, "evaluator_profiles")
    if set(document) != _EVALUATOR_ROOT_FIELDS:
        raise ProfileError("evaluator profile document fields mismatch")
    for field in ("schema_version", "evaluator_interface_version", "profile_schema_version"):
        if not _VERSION.fullmatch(_text(document[field], field)):
            raise ProfileError(f"{field} must be a major.minor version")
    _text(document["status"], "status")
    raw_profiles = document["profiles"]
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise ProfileError("evaluator profiles must be a non-empty mapping")
    result: dict[str, EvaluatorProfile] = {}
    for raw_key, value in raw_profiles.items():
        key = _profile_key(raw_key, "evaluator profile key")
        result[key] = _validate_evaluator_profile(key, value)
    if not {"qwen_shared", "deepseek_shared"}.issubset(result):
        raise ProfileError("qwen_shared and deepseek_shared profiles are required")
    return MappingProxyType(result)


def _validate_model_profile(model_key: str, value: object) -> ModelProfile:
    if not isinstance(value, Mapping) or set(value) != _MODEL_FIELDS:
        raise ProfileError(f"model profile {model_key!r} has invalid fields")
    if value["model_key"] != model_key:
        raise ProfileError(f"model profile {model_key!r} has a mismatched model_key")
    api_key_env = _text(value["api_key_env"], f"{model_key}.api_key_env")
    if not _ENV_NAME.fullmatch(api_key_env):
        raise ProfileError(f"{model_key}.api_key_env must name an environment variable")
    status = _text(value["status"], f"{model_key}.status")
    if status not in {"release", "local"} or value["requires_owner_approval"] is not False:
        raise ProfileError(
            f"{model_key} must be an approved release profile or a local user profile"
        )
    public_model_id = _text(value["public_model_id"], f"{model_key}.public_model_id")
    raw_reasoning_effort = value["reasoning_effort"]
    reasoning_effort = (
        None
        if raw_reasoning_effort is None
        else _text(raw_reasoning_effort, f"{model_key}.reasoning_effort")
    )
    return ModelProfile(
        model_key=model_key,
        display_name=_text(value["display_name"], f"{model_key}.display_name"),
        evaluator_profile=_profile_key(
            value["evaluator_profile"], f"{model_key}.evaluator_profile"
        ),
        provider_profile=_profile_key(
            value["provider_profile"], f"{model_key}.provider_profile"
        ),
        public_model_id=public_model_id,
        api_key_env=api_key_env,
        thinking_enabled=_setting_value(
            value["thinking_enabled"], f"{model_key}.thinking_enabled"
        ),
        reasoning_effort=reasoning_effort,
        temperature=_sampling_value(
            value["temperature"], f"{model_key}.temperature"
        ),
        top_p=_sampling_value(value["top_p"], f"{model_key}.top_p"),
        notes=_text(value["notes"], f"{model_key}.notes"),
        status=status,
        requires_owner_approval=False,
    )


def load_model_profiles(
    path: str | Path = DEFAULT_MODEL_PROFILES_PATH,
) -> Mapping[str, ModelProfile]:
    """Load resolved model profiles without limiting users to paper models.

    The public release gate separately requires the bundled first-release
    inventory.  A caller-supplied registry may safely add other models that use
    one of the shared evaluator/provider contracts.
    """

    document = load_config(path)
    _scan_safe(document, "model_profiles")
    if set(document) != _MODEL_ROOT_FIELDS:
        raise ProfileError("model profile document fields mismatch")
    for field in ("schema_version", "profile_schema_version"):
        if not _VERSION.fullmatch(_text(document[field], field)):
            raise ProfileError(f"{field} must be a major.minor version")
    _text(document["status"], "status")
    raw_models = document["models"]
    if not isinstance(raw_models, Mapping) or not raw_models:
        raise ProfileError("model profiles must be a non-empty mapping")
    configured = set(raw_models)
    result = {
        key: _validate_model_profile(key, raw_models[key]) for key in sorted(configured)
    }
    for key in configured & QWEN_SHARED_MODELS:
        if result[key].evaluator_profile != "qwen_shared":
            raise ProfileError(f"{key} must resolve to qwen_shared")
    for key in configured & DEEPSEEK_SHARED_MODELS:
        if result[key].evaluator_profile != "deepseek_shared":
            raise ProfileError(f"{key} must resolve to deepseek_shared")
    return MappingProxyType(result)


def _validate_run_profile(profile_id: str, value: object) -> RunProfile:
    if not isinstance(value, Mapping) or set(value) != _RUN_FIELDS:
        raise ProfileError(f"run profile {profile_id!r} has invalid fields")
    if value["profile_id"] != profile_id:
        raise ProfileError(f"run profile {profile_id!r} has a mismatched profile_id")
    models = _string_tuple(value["applicable_models"], f"{profile_id}.applicable_models")
    domains = _string_tuple(value["applicable_domains"], f"{profile_id}.applicable_domains")
    settings = _string_tuple(value["applicable_settings"], f"{profile_id}.applicable_settings")
    if not set(domains).issubset(_DOMAINS):
        raise ProfileError(f"{profile_id} has an unsupported domain")
    if not set(settings).issubset(_SETTINGS):
        raise ProfileError(f"{profile_id} has an unsupported setting")
    transport = validate_transport_mapping(
        {field: value[field] for field in TRANSPORT_FIELD_ORDER},
        path=f"{profile_id}.transport",
        require_all=True,
    )
    status = _text(value["status"], f"{profile_id}.status")
    if status != "release" or value["requires_owner_approval"] is not False:
        raise ProfileError(f"{profile_id} must be an approved release profile")
    return RunProfile(
        profile_id=profile_id,
        provider_profile=_profile_key(
            value["provider_profile"], f"{profile_id}.provider_profile"
        ),
        applicable_models=models,
        applicable_domains=domains,
        applicable_settings=settings,
        timeout_seconds=transport["timeout_seconds"],
        max_retries=transport["max_retries"],
        retry_backoff_seconds=transport["retry_backoff_seconds"],
        retry_backoff_mode=transport["retry_backoff_mode"],
        streaming=transport["streaming"],
        request_safety_limits=transport["request_safety_limits"],
        provenance_source=_text(
            value["provenance_source"], f"{profile_id}.provenance_source"
        ),
        status=status,
        requires_owner_approval=False,
        notes=_text(value["notes"], f"{profile_id}.notes"),
    )


def load_run_profiles(
    path: str | Path = DEFAULT_RUN_PROFILES_PATH,
) -> Mapping[str, RunProfile]:
    """Load approved transport profiles without resolving credentials."""

    document = load_config(path)
    _scan_safe(document, "run_profiles")
    if set(document) != _RUN_ROOT_FIELDS:
        raise ProfileError("run profile document fields mismatch")
    if not _VERSION.fullmatch(_text(document["schema_version"], "schema_version")):
        raise ProfileError("schema_version must be a major.minor version")
    _text(document["status"], "status")
    raw_profiles = document["profiles"]
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise ProfileError("run profiles must be a non-empty mapping")
    result: dict[str, RunProfile] = {}
    for raw_key, value in raw_profiles.items():
        key = _profile_key(raw_key, "run profile key")
        result[key] = _validate_run_profile(key, value)
    return MappingProxyType(result)


def resolve_run_profile(
    profile_id: str, run_profiles: Mapping[str, RunProfile]
) -> RunProfile:
    checked = _profile_key(profile_id, "run_profile")
    try:
        return run_profiles[checked]
    except KeyError as exc:
        raise ProfileError(f"unknown run profile: {checked!r}") from exc


def validate_run_profile_applicability(
    run_profile: RunProfile,
    *,
    model_key: str,
    provider_profile: str,
    domain: str,
    setting: str,
) -> None:
    if run_profile.provider_profile != provider_profile:
        raise ProfileError("run and provider profile references do not match")
    if model_key not in run_profile.applicable_models:
        raise ProfileError("run profile does not apply to the selected model")
    if domain not in run_profile.applicable_domains:
        raise ProfileError("run profile does not apply to the selected domain")
    if setting not in run_profile.applicable_settings:
        raise ProfileError("run profile does not apply to the selected setting")


def resolve_transport_parameters(
    provider_profile: Mapping[str, Any],
    run_profile: RunProfile | None = None,
    cell_override: Mapping[str, Any] | None = None,
) -> ResolvedTransport:
    """Resolve transport as cell override > run profile > provider defaults."""

    defaults = validate_transport_mapping(
        provider_profile.get("transport_defaults"),
        path="provider_profile.transport_defaults",
        require_all=True,
    )
    values = dict(defaults)
    sources = {field: "provider_profile_default" for field in TRANSPORT_FIELDS}
    if run_profile is not None:
        for field, value in run_profile.transport_dict().items():
            values[field] = value
            sources[field] = f"run_profile:{run_profile.profile_id}"
    overrides = validate_transport_mapping(cell_override, path="cell.transport")
    for field, value in overrides.items():
        values[field] = value
        sources[field] = "cell_transport_override"
    return ResolvedTransport(values=values, source_by_field=sources)


def resolve_model_profile(
    model_key: str, model_profiles: Mapping[str, ModelProfile]
) -> ModelProfile:
    """Resolve one model key without applying defaults or aliases."""

    checked = _profile_key(model_key, "model_key")
    try:
        return model_profiles[checked]
    except KeyError as exc:
        raise ProfileError(f"unknown model profile: {checked!r}") from exc


def resolve_evaluator_profile(
    model_profile: ModelProfile,
    evaluator_profiles: Mapping[str, EvaluatorProfile],
) -> EvaluatorProfile:
    """Resolve the evaluator selected by a validated model profile."""

    try:
        profile = evaluator_profiles[model_profile.evaluator_profile]
    except KeyError as exc:
        raise ProfileError(
            f"unknown evaluator profile: {model_profile.evaluator_profile!r}"
        ) from exc
    return profile


def _coerce_evaluator_profiles(
    value: str | Path | Mapping[str, EvaluatorProfile],
) -> Mapping[str, EvaluatorProfile]:
    if isinstance(value, (str, Path)):
        return load_evaluator_profiles(value)
    return value


def _coerce_model_profiles(
    value: str | Path | Mapping[str, ModelProfile],
) -> Mapping[str, ModelProfile]:
    if isinstance(value, (str, Path)):
        return load_model_profiles(value)
    return value


def _cell_field(cell: object, field: str, default: Any = None) -> Any:
    if isinstance(cell, Mapping):
        return cell.get(field, default)
    return getattr(cell, field, default)


def _find_unresolved(value: Any, path: str) -> list[str]:
    found: list[str] = []
    if value is None or value == "unresolved":
        return [path]
    if isinstance(value, Mapping):
        for key, item in value.items():
            found.extend(_find_unresolved(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_find_unresolved(item, f"{path}[{index}]"))
    return found


def _budget_lookup(budgets: Mapping[str, Any], profile_key: str) -> Mapping[str, Any] | None:
    raw = budgets.get("profiles", budgets)
    if isinstance(raw, Mapping):
        value = raw.get(profile_key)
        return value if isinstance(value, Mapping) else None
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping) and item.get("profile_id") == profile_key:
                return item
    return None


def resolve_profiles_for_cell(
    cell: object,
    model_profiles: str | Path | Mapping[str, ModelProfile],
    evaluator_profiles: str | Path | Mapping[str, EvaluatorProfile],
    provider_profiles: Mapping[str, Any] | None = None,
    budgets: Mapping[str, Any] | None = None,
    run_profiles: Mapping[str, RunProfile] | None = None,
    run_profile_id: str | None = None,
) -> ResolvedProfiles:
    """Resolve profile references for a validated cell without executing it."""

    models = _coerce_model_profiles(model_profiles)
    evaluators = _coerce_evaluator_profiles(evaluator_profiles)
    model = resolve_model_profile(_text(_cell_field(cell, "model_name"), "cell.model_name"), models)
    evaluator = resolve_evaluator_profile(model, evaluators)

    domain = _text(_cell_field(cell, "domain"), "cell.domain")
    mode = _text(_cell_field(cell, "mode"), "cell.mode")
    setting = _text(_cell_field(cell, "setting"), "cell.setting")
    if domain not in evaluator.supports_domains:
        raise ProfileError(f"evaluator {evaluator.profile_key!r} does not support domain")
    if mode not in evaluator.supports_modes:
        raise ProfileError(f"evaluator {evaluator.profile_key!r} does not support mode")
    if setting not in evaluator.supports_settings:
        raise ProfileError(f"evaluator {evaluator.profile_key!r} does not support setting")

    cell_provider = _text(
        _cell_field(cell, "provider_profile"), "cell.provider_profile"
    )
    if cell_provider != model.provider_profile:
        raise ProfileError("cell and model profile provider references do not match")

    provider: Mapping[str, Any] | None = None
    if provider_profiles is not None:
        raw_provider = provider_profiles.get(model.provider_profile)
        if raw_provider is None and provider_profiles.get("provider_name") == model.provider_profile:
            raw_provider = provider_profiles
        if raw_provider is None:
            raise ProfileError("selected provider profile is missing")
        if not isinstance(raw_provider, Mapping):
            raise ProfileError("selected provider profile must be a mapping")
        _scan_safe(raw_provider, "provider_profile")
        provider = dict(raw_provider)

    run_profile: RunProfile | None = None
    transport_profile: ResolvedTransport | None = None
    selected_run_profile = run_profile_id or _cell_field(cell, "run_profile")
    if selected_run_profile is not None:
        if run_profiles is None:
            raise ProfileError("run_profiles are required when a run_profile is selected")
        run_profile = resolve_run_profile(str(selected_run_profile), run_profiles)
        validate_run_profile_applicability(
            run_profile,
            model_key=model.model_key,
            provider_profile=model.provider_profile,
            domain=domain,
            setting=setting,
        )
    if provider is not None:
        transport_profile = resolve_transport_parameters(
            provider,
            run_profile,
            _cell_field(cell, "transport"),
        )

    budget_profile: Mapping[str, Any] | None = None
    if budgets is not None:
        raw_budget = _cell_field(cell, "budget", {})
        if not isinstance(raw_budget, Mapping):
            raise ProfileError("cell budget must be a mapping")
        profile_key = _text(raw_budget.get("profile"), "cell.budget.profile")
        budget_profile = _budget_lookup(budgets, profile_key)
        if budget_profile is None:
            raise ProfileError("selected budget profile is missing")
        _scan_safe(budget_profile, "budget_profile")

    unresolved = _find_unresolved(model.to_dict(), "model_profile")
    if provider is not None:
        identity = {key: value for key, value in provider.items() if key != "transport_defaults"}
        unresolved.extend(_find_unresolved(identity, "provider_profile"))
    if transport_profile is not None:
        unresolved.extend(
            f"transport_profile.{field}"
            for field in transport_profile.unresolved_fields
        )
    if budget_profile is not None:
        unresolved.extend(_find_unresolved(budget_profile, "budget_profile"))
    questions = _cell_field(cell, "unresolved_questions", ()) or ()
    if questions:
        unresolved.append("cell.unresolved_questions")
    unresolved = sorted(set(unresolved))
    requires_approval = bool(
        model.requires_owner_approval
        or _cell_field(cell, "requires_owner_approval", False)
        or unresolved
    )
    resolved = ResolvedProfiles(
        model_profile=model,
        evaluator_profile=evaluator,
        provider_profile=provider,
        budget_profile=budget_profile,
        unresolved_fields=tuple(unresolved),
        requires_owner_approval=requires_approval,
        run_profile=run_profile,
        transport_profile=transport_profile,
    )
    assert_no_model_specific_runner_fork(resolved)
    return resolved


def assert_no_model_specific_runner_fork(
    resolved_profile: ResolvedProfiles | Mapping[str, Any],
) -> None:
    """Enforce the frozen shared-runner contract for all eight model keys."""

    if isinstance(resolved_profile, ResolvedProfiles):
        model_key = resolved_profile.model_profile.model_key
        evaluator_key = resolved_profile.evaluator_profile.profile_key
        evaluator = resolved_profile.evaluator_profile.to_dict()
    else:
        model = resolved_profile.get("model_profile", {})
        evaluator = resolved_profile.get("evaluator_profile", {})
        if not isinstance(model, Mapping) or not isinstance(evaluator, Mapping):
            raise ProfileError("resolved profile must contain model and evaluator mappings")
        model_key = _text(model.get("model_key"), "model_profile.model_key")
        evaluator_key = _text(
            evaluator.get("profile_key"), "evaluator_profile.profile_key"
        )

    expected = (
        "qwen_shared"
        if model_key in QWEN_SHARED_MODELS
        else "deepseek_shared"
        if model_key in DEEPSEEK_SHARED_MODELS
        else None
    )
    if expected is None or evaluator_key != expected:
        raise ProfileError("model does not resolve to its required shared evaluator profile")

    required_shared = {
        "runner": "shared_tool_free_runner",
        "prompt_renderer": "shared_prompt_renderer",
        "one_stage_orchestration": "shared",
        "parser_dispatch": "shared_domain_dispatch",
        "result_schema": "standard_v1",
    }
    for field, expected_value in required_shared.items():
        if evaluator.get(field) != expected_value:
            raise ProfileError(f"{field} violates the shared evaluator contract")
    if model_key in QWEN_SHARED_MODELS and evaluator.get("two_stage_orchestration") != "shared":
        raise ProfileError("qwen_shared models must use shared two-stage orchestration")

    model_token = re.sub(r"[^a-z0-9]+", "_", model_key.lower()).strip("_")
    for field in _SHARED_COMPONENT_FIELDS:
        value = evaluator.get(field)
        if isinstance(value, str) and model_token and model_token in value.lower():
            raise ProfileError(f"{field} contains a model-specific evaluator fork")


__all__ = [
    "DEFAULT_EVALUATOR_PROFILES_PATH",
    "DEFAULT_MODEL_PROFILES_PATH",
    "DEFAULT_RUN_PROFILES_PATH",
    "DEEPSEEK_SHARED_MODELS",
    "FIRST_RELEASE_MODEL_INVENTORY",
    "EvaluatorProfile",
    "ModelProfile",
    "ProfileError",
    "QWEN_SHARED_MODELS",
    "ResolvedProfiles",
    "ResolvedTransport",
    "RunProfile",
    "TRANSPORT_FIELDS",
    "TRANSPORT_FIELD_ORDER",
    "assert_no_model_specific_runner_fork",
    "load_evaluator_profiles",
    "load_model_profiles",
    "load_run_profiles",
    "resolve_evaluator_profile",
    "resolve_model_profile",
    "resolve_profiles_for_cell",
    "resolve_run_profile",
    "resolve_transport_parameters",
    "validate_run_profile_applicability",
    "validate_transport_mapping",
]
