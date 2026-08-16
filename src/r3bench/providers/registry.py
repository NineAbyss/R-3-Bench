"""Safe provider-profile loading and adapter construction."""

from __future__ import annotations

import ipaddress
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from r3bench.common.config import load_config
from r3bench.common.provider import MockProvider, ReplayProvider
from r3bench.providers.errors import ProviderConfigError
from r3bench.providers.openai_compatible import OpenAICompatibleAdapter


_ADAPTERS = ("mock", "openai_compatible", "replay")
_PROFILE_FIELDS = frozenset(
    {
        "adapter",
        "provider_name",
        "base_url",
        "api_key_env",
        "token_budget_field",
        "transport_defaults",
        "default_headers",
        "extra_body",
    }
)
_REQUIRED_PROFILE_FIELDS = _PROFILE_FIELDS - {"token_budget_field"}
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_CREDENTIAL = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{12,}\b|\bhf_[A-Za-z0-9]{12,}\b|"
    r"\bolp_[A-Za-z0-9]{12,}\b|Bearer\s+[A-Za-z0-9._~+/-]{12,})"
)
_ENV_PATH = re.compile(r"(?:^|[/\\])\.env(?:$|[/\\])")
_FORBIDDEN_HEADER_NAMES = frozenset(
    {"authorization", "proxy-authorization", "x-api-key", "api-key", "cookie"}
)


def _scan_safe(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in {
                "api_key",
                "access_token",
                "password",
                "provider_headers",
            }:
                raise ProviderConfigError(f"{path} contains a forbidden field")
            _scan_safe(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_safe(item, f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    if _CREDENTIAL.search(value):
        raise ProviderConfigError(f"{path} contains a raw API-key-like value")
    if Path(value).is_absolute() or _WINDOWS_ABSOLUTE.match(value):
        raise ProviderConfigError(f"{path} contains a private absolute path")
    if _ENV_PATH.search(value):
        raise ProviderConfigError(f"{path} contains a forbidden .env path")


def _validate_base_url(value: object) -> None:
    if value == "unresolved" or value is None:
        return
    if not isinstance(value, str):
        raise ProviderConfigError("base_url must be unresolved, null, or a public URL")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ProviderConfigError("base_url must use a public HTTPS endpoint")
    host = parsed.hostname
    if host in {"localhost", "localhost.localdomain"}:
        raise ProviderConfigError("base_url must not use a private endpoint")
    try:
        if ipaddress.ip_address(host).is_private:
            raise ProviderConfigError("base_url must not use a private endpoint")
    except ValueError:
        if host.endswith((".internal", ".local")):
            raise ProviderConfigError("base_url must not use a private endpoint")


def validate_provider_profile(provider_profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated safe copy without reading an API key."""

    if not isinstance(provider_profile, Mapping):
        raise ProviderConfigError("provider profile must be a mapping")
    fields = set(provider_profile)
    if not _REQUIRED_PROFILE_FIELDS <= fields or not fields <= _PROFILE_FIELDS:
        raise ProviderConfigError("provider profile fields do not match the public schema")
    _scan_safe(provider_profile, "provider_profile")
    adapter = provider_profile.get("adapter")
    if adapter not in _ADAPTERS:
        raise ProviderConfigError("unsupported provider adapter")
    provider_name = provider_profile.get("provider_name")
    if not isinstance(provider_name, str) or not provider_name.strip():
        raise ProviderConfigError("provider_name must be a non-empty string")
    _validate_base_url(provider_profile.get("base_url"))
    env_name = provider_profile.get("api_key_env")
    if env_name is not None and not (
        isinstance(env_name, str) and _ENV_NAME.fullmatch(env_name)
    ):
        raise ProviderConfigError("api_key_env must be an environment-variable name")
    if adapter == "openai_compatible" and env_name is None:
        raise ProviderConfigError("openai_compatible requires api_key_env")
    token_budget_field = provider_profile.get("token_budget_field", "max_tokens")
    if token_budget_field not in {"max_tokens", "max_completion_tokens"}:
        raise ProviderConfigError(
            "token_budget_field must be max_tokens or max_completion_tokens"
        )
    from r3bench.common.profile_registry import ProfileError, validate_transport_mapping

    try:
        validate_transport_mapping(
            provider_profile.get("transport_defaults"),
            path="provider_profile.transport_defaults",
            require_all=True,
        )
    except ProfileError as exc:
        raise ProviderConfigError(str(exc)) from exc
    headers = provider_profile.get("default_headers")
    extra = provider_profile.get("extra_body")
    if not isinstance(headers, Mapping) or not isinstance(extra, Mapping):
        raise ProviderConfigError("default_headers and extra_body must be mappings")
    if _FORBIDDEN_HEADER_NAMES & {str(key).lower() for key in headers}:
        raise ProviderConfigError("default_headers contains a credential-bearing header")
    return dict(provider_profile)


def load_provider_profile(path: str | Path) -> dict[str, Any]:
    return validate_provider_profile(load_config(path))


def list_provider_adapters() -> tuple[str, ...]:
    return _ADAPTERS


def create_provider_adapter(
    provider_profile: Mapping[str, Any],
    model_profile: Mapping[str, Any] | object,
    *,
    transport: object | None = None,
    transport_config: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    mock_response: str | None = None,
    replay_path: str | None = None,
) -> object:
    """Create a provider behind the shared runner without model-specific code."""

    checked = validate_provider_profile(provider_profile)
    adapter = checked["adapter"]
    if adapter == "openai_compatible":
        return OpenAICompatibleAdapter(
            checked,
            model_profile,
            transport=transport,  # type: ignore[arg-type]
            transport_config=transport_config,
            allow_unresolved=dry_run,
        )
    if adapter == "mock":
        return MockProvider(mock_response or "Synthetic mock response.")
    if replay_path is None:
        raise ProviderConfigError("replay adapter requires an explicit local replay path")
    return ReplayProvider(replay_path)


__all__ = [
    "create_provider_adapter",
    "list_provider_adapters",
    "load_provider_profile",
    "validate_provider_profile",
]
