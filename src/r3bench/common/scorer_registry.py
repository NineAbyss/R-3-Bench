"""Safe release scorer-profile loading with no scorer execution."""

from __future__ import annotations

import ipaddress
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlparse

from r3bench.common.config import load_config
from r3bench.resource_paths import resource_path, resolve_path


DEFAULT_SCORER_PROFILES_PATH = resource_path("configs", "scorer_profiles.yaml")
_ROOT_FIELDS = frozenset({"schema_version", "status", "profiles"})
_COMMON_FIELDS = frozenset(
    {
        "profile_id",
        "domain",
        "scorer_type",
        "status",
        "requires_owner_approval",
        "notes",
    }
)
_FIELDS = {
    "external_verifier": _COMMON_FIELDS
    | {
        "verifier",
        "asset_lookup",
        "requires_external_assets",
        "hidden_tests_in_public_dataset",
        "binding",
    },
    "model_equivalence_judge": _COMMON_FIELDS
    | {
        "judge_prompt",
        "judge_prompt_sha256",
        "judge_model",
        "provider_profile",
        "single_run_profile",
        "contest_run_profile",
        "api_key_env",
        "response_format",
        "max_tokens",
        "temperature",
        "top_p",
    },
    "reasoning_gym": _COMMON_FIELDS
    | {
        "reasoning_gym_version",
        "reasoning_gym_revision",
        "module_name",
        "generator_aware",
        "universal_exact_match",
        "bridge",
    },
}
_DOMAIN_FOR_TYPE = {
    "external_verifier": "coding",
    "model_equivalence_judge": "math",
    "reasoning_gym": "abstract_reasoning",
}
_PROFILE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_VERSION = re.compile(r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_CREDENTIAL = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{12,}\b|\bhf_[A-Za-z0-9]{12,}\b|"
    r"\bolp_[A-Za-z0-9]{12,}\b|Bearer\s+[A-Za-z0-9._~+/-]{12,})"
)
_FORBIDDEN_KEYS = frozenset(
    {
        "api_key",
        "access_token",
        "password",
        "authorization",
        "base_url",
        "endpoint",
        "hidden_test_path",
        "assets_root",
        "verifier_root",
        "provider_headers",
        "provider_request_id",
    }
)


class ScorerProfileError(ValueError):
    """Raised when a scorer profile is unsafe or inconsistent."""


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScorerProfileError(f"{field} must be a non-empty string")
    return value


def _scan_safe(value: Any, path: str = "scorer_profile") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.lower() in _FORBIDDEN_KEYS:
                raise ScorerProfileError(f"{path} contains forbidden field {key!r}")
            _scan_safe(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_safe(item, f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    if _CREDENTIAL.search(value):
        raise ScorerProfileError(f"{path} contains an API-key-like value")
    if Path(value).is_absolute() or _WINDOWS_ABSOLUTE.match(value):
        raise ScorerProfileError(f"{path} contains a private absolute path")
    if ".env" in Path(value).parts:
        raise ScorerProfileError(f"{path} contains a forbidden .env path")
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        try:
            private = ipaddress.ip_address(parsed.hostname).is_private
        except ValueError:
            private = (
                parsed.hostname.endswith((".internal", ".local"))
                or parsed.hostname == "localhost"
            )
        if private:
            raise ScorerProfileError(f"{path} contains a private endpoint")


@dataclass(frozen=True, slots=True)
class ScorerProfile:
    profile_id: str
    domain: str
    scorer_type: str
    config: Mapping[str, Any]
    status: str
    requires_owner_approval: bool
    notes: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", MappingProxyType(dict(self.config)))

    @property
    def unresolved_fields(self) -> tuple[str, ...]:
        return tuple(
            sorted(key for key, value in self.config.items() if value == "unresolved")
        )

    @property
    def production_ready(self) -> bool:
        return not self.unresolved_fields

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "domain": self.domain,
            "scorer_type": self.scorer_type,
            **dict(self.config),
            "status": self.status,
            "requires_owner_approval": self.requires_owner_approval,
            "notes": self.notes,
            "unresolved_fields": list(self.unresolved_fields),
            "production_ready": self.production_ready,
        }


def scorer_profile_contract(profile: ScorerProfile) -> dict[str, Any]:
    """Return the public behavior-bearing scorer contract."""

    return {
        "schema_version": "1.0",
        "profile_id": profile.profile_id,
        "domain": profile.domain,
        "scorer_type": profile.scorer_type,
        "config": dict(profile.config),
    }


def scorer_profile_contract_sha256(profile: ScorerProfile) -> str:
    payload = json.dumps(
        scorer_profile_contract(profile),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_profile(profile_id: str, raw: object) -> ScorerProfile:
    if not isinstance(raw, Mapping):
        raise ScorerProfileError(f"profile {profile_id!r} must be a mapping")
    scorer_type = _text(raw.get("scorer_type"), f"{profile_id}.scorer_type")
    expected = _FIELDS.get(scorer_type)
    if expected is None or set(raw) != expected:
        raise ScorerProfileError(f"profile {profile_id!r} has invalid fields")
    if raw.get("profile_id") != profile_id or not _PROFILE_ID.fullmatch(profile_id):
        raise ScorerProfileError(f"profile {profile_id!r} has an invalid profile_id")
    domain = _text(raw["domain"], f"{profile_id}.domain")
    if domain != _DOMAIN_FOR_TYPE[scorer_type]:
        raise ScorerProfileError(f"{profile_id} domain does not match scorer_type")
    if (
        raw.get("status") != "release"
        or raw.get("requires_owner_approval") is not False
    ):
        raise ScorerProfileError(f"{profile_id} must be an approved release profile")

    config = {key: raw[key] for key in sorted(expected - _COMMON_FIELDS)}
    if scorer_type == "external_verifier":
        if config != {
            "verifier": "LightCPVerifier",
            "asset_lookup": "upstream_id",
            "requires_external_assets": True,
            "hidden_tests_in_public_dataset": False,
            "binding": "lightcp_http_v1",
        }:
            raise ScorerProfileError(
                "Coding scorer must use external upstream_id assets"
            )
    elif scorer_type == "model_equivalence_judge":
        prompt = _text(config["judge_prompt"], f"{profile_id}.judge_prompt")
        if Path(prompt).is_absolute() or ".." in Path(prompt).parts:
            raise ScorerProfileError("judge_prompt must be a safe relative path")
        prompt_path = resolve_path(prompt)
        if not prompt_path.is_file():
            raise ScorerProfileError("judge_prompt does not exist in the release")
        prompt_sha = _text(
            config["judge_prompt_sha256"], f"{profile_id}.judge_prompt_sha256"
        )
        actual_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        if prompt_sha != actual_sha:
            raise ScorerProfileError("judge_prompt_sha256 does not match judge_prompt")
        env_name = _text(config["api_key_env"], f"{profile_id}.api_key_env")
        if not _ENV_NAME.fullmatch(env_name):
            raise ScorerProfileError("api_key_env must name an environment variable")
        judge_model = _text(config["judge_model"], f"{profile_id}.judge_model")
        if judge_model != "deepseek-v4-flash":
            raise ScorerProfileError("Math scorer must use DeepSeek V4 Flash")
        for field in ("provider_profile", "single_run_profile", "contest_run_profile"):
            _text(config[field], f"{profile_id}.{field}")
        if config["response_format"] != "omnimath_markdown":
            raise ScorerProfileError("Math scorer must use the formal Omni-MATH parser")
        if (
            config["max_tokens"] != 16384
            or config["temperature"] != 0
            or config["top_p"] != 1
        ):
            raise ScorerProfileError(
                "Math scorer request fields differ from formal provenance"
            )
    else:
        version = _text(
            config["reasoning_gym_version"], f"{profile_id}.reasoning_gym_version"
        )
        if version != "unresolved" and not _VERSION.fullmatch(version):
            raise ScorerProfileError("reasoning_gym_version is invalid")
        if (
            config["generator_aware"] is not True
            or config["universal_exact_match"] is not False
        ):
            raise ScorerProfileError(
                "production AR scoring must remain generator-aware"
            )
        if config["bridge"] != "create_dataset_score_answer":
            raise ScorerProfileError(
                "AR scorer bridge does not match formal provenance"
            )
        revision = _text(
            config["reasoning_gym_revision"], f"{profile_id}.reasoning_gym_revision"
        )
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ScorerProfileError("reasoning_gym_revision must be a Git commit")
        _text(config["module_name"], f"{profile_id}.module_name")

    _scan_safe(raw, f"profiles.{profile_id}")
    return ScorerProfile(
        profile_id=profile_id,
        domain=domain,
        scorer_type=scorer_type,
        config=config,
        status="release",
        requires_owner_approval=False,
        notes=_text(raw["notes"], f"{profile_id}.notes"),
    )


def load_scorer_profiles(
    path: str | Path = DEFAULT_SCORER_PROFILES_PATH,
) -> Mapping[str, ScorerProfile]:
    document = load_config(path)
    _scan_safe(document, "scorer_profiles")
    if set(document) != _ROOT_FIELDS:
        raise ScorerProfileError("scorer profile document fields mismatch")
    if document.get("schema_version") != "1.0":
        raise ScorerProfileError("scorer profile schema_version must be '1.0'")
    _text(document.get("status"), "status")
    raw_profiles = document.get("profiles")
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise ScorerProfileError("scorer profiles must be a non-empty mapping")
    result = {
        str(key): _validate_profile(str(key), value)
        for key, value in raw_profiles.items()
    }
    return MappingProxyType(result)


def resolve_scorer_profile(
    profile_id: str,
    profiles: Mapping[str, ScorerProfile],
    *,
    domain: str | None = None,
) -> ScorerProfile:
    try:
        profile = profiles[profile_id]
    except KeyError as exc:
        raise ScorerProfileError(f"unknown scorer profile: {profile_id!r}") from exc
    if domain is not None and profile.domain != domain:
        raise ScorerProfileError(
            "scorer profile domain does not match requested domain"
        )
    return profile


__all__ = [
    "DEFAULT_SCORER_PROFILES_PATH",
    "ScorerProfile",
    "ScorerProfileError",
    "load_scorer_profiles",
    "resolve_scorer_profile",
    "scorer_profile_contract",
    "scorer_profile_contract_sha256",
]
