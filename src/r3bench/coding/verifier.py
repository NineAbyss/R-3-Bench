"""Coding verifier contracts with external hidden assets kept runtime-only.

The production binding mirrors the formal LightCPVerifier HTTP contract, but
it deliberately does not start containers or download hidden assets. A caller
must provision those assets and opt into the external verifier separately.
"""

from __future__ import annotations

import ipaddress
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, TypeAlias
from urllib.parse import urlparse

from r3bench.common.config import load_config
from r3bench.common.schema import ProblemRecord


CodingVerifierStatus: TypeAlias = Literal[
    "accepted",
    "wrong_answer",
    "compilation_error",
    "runtime_error",
    "time_limit_exceeded",
    "verifier_error",
    "missing_solution",
    "not_configured",
    "service_unreachable",
    "assets_unavailable",
    "invalid_config",
]

CODING_VERIFIER_STATUSES = frozenset(
    {
        "accepted",
        "wrong_answer",
        "compilation_error",
        "runtime_error",
        "time_limit_exceeded",
        "verifier_error",
        "missing_solution",
        "not_configured",
        "service_unreachable",
        "assets_unavailable",
        "invalid_config",
    }
)
_PUBLIC_CONFIG_FIELDS = frozenset(
    {
        "verifier_type",
        "mode",
        "service_url",
        "asset_root_env",
        "timeout_seconds",
        "max_retries",
        "status",
        "requires_owner_approval",
    }
)
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_CREDENTIAL = re.compile(
    r"(?i)(?:\bsk-[A-Za-z0-9_-]{12,}\b|\bhf_[A-Za-z0-9]{12,}\b|"
    r"\bolp_[A-Za-z0-9]{12,}\b|Bearer\s+[A-Za-z0-9._~+/-]{12,})"
)
_FORBIDDEN_CONFIG_KEYS = frozenset(
    {
        "api_key",
        "access_token",
        "password",
        "authorization",
        "hidden_tests",
        "hidden_test_path",
        "testcase_path",
        "checker_path",
        "asset_root",
        "assets_root",
        "problem_assets_root",
        "verifier_root",
        "provider_headers",
    }
)


class LightCPVerifierError(RuntimeError):
    """Base error for sanitized external-verifier failures."""


class LightCPVerifierConfigError(LightCPVerifierError):
    """Raised when a public or runtime verifier config is invalid."""


class LightCPVerifierServiceError(LightCPVerifierError):
    """Raised when an explicitly configured verifier service cannot be reached."""


class LightCPVerifierAssetsError(LightCPVerifierError):
    """Raised when separately provisioned verifier assets are unavailable."""


@dataclass(frozen=True, slots=True)
class LightCPVerifierConfig:
    """Runtime-only configuration for a separately installed LightCPVerifier.

    The path fields are retained for backward-compatible injected bindings.
    Public YAML files use only ``service_url`` and ``asset_root_env``; the
    environment-variable value is resolved only at scoring time.
    """

    verifier_root: Path | None = None
    problem_assets_root: Path | None = None
    judge_url: str | None = None
    timeout_seconds: float = 120.0
    submit_timeout_seconds: float = 30.0
    poll_timeout_seconds: float = 10.0
    poll_interval_seconds: float = 1.0
    asset_config_name: str = "config.yaml"
    verifier_type: str = "lightcpverifier"
    mode: str = "service"
    service_url: str | None = None
    asset_root_env: str | None = None
    max_retries: int = 1
    status: str = "runtime"
    requires_owner_approval: bool = False
    runtime_private_config: bool = False

    def __post_init__(self) -> None:
        for field in (
            "timeout_seconds",
            "submit_timeout_seconds",
            "poll_timeout_seconds",
            "poll_interval_seconds",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{field} must be positive")
        if not isinstance(self.asset_config_name, str) or not self.asset_config_name:
            raise ValueError("asset_config_name must be non-empty")
        if self.verifier_type != "lightcpverifier":
            raise ValueError("verifier_type must be lightcpverifier")
        if self.mode not in {"service", "local"}:
            raise ValueError("mode must be service or local")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise ValueError("max_retries must be an integer")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.asset_root_env is not None and not _ENV_NAME.fullmatch(
            self.asset_root_env
        ):
            raise ValueError("asset_root_env must name an environment variable")
        for field in ("status",):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be non-empty")
        if not isinstance(self.requires_owner_approval, bool):
            raise ValueError("requires_owner_approval must be boolean")
        if not isinstance(self.runtime_private_config, bool):
            raise ValueError("runtime_private_config must be boolean")

    @property
    def effective_service_url(self) -> str | None:
        value = self.service_url if self.service_url not in {None, "unresolved"} else self.judge_url
        return value if isinstance(value, str) and value else None

    def resolved_assets_root(self) -> Path | None:
        if self.problem_assets_root is not None:
            return self.problem_assets_root
        if self.asset_root_env is None:
            return None
        value = os.environ.get(self.asset_root_env)
        return Path(value) if value else None


@dataclass(frozen=True, slots=True)
class CodingVerifierResult:
    upstream_id: str
    accepted: bool
    verdict: str
    detail: str | None = None
    time_ms: float | None = None
    memory_mb: float | None = None
    status: CodingVerifierStatus | None = None

    def __post_init__(self) -> None:
        status = self.status or _status_from_verdict(self.verdict, self.accepted)
        if status not in CODING_VERIFIER_STATUSES:
            raise ValueError("unsupported Coding verifier status")
        object.__setattr__(self, "status", status)


class CodingVerifier(Protocol):
    """Verifier interface keyed by a public row's stable upstream ID."""

    def verify(
        self, problem_or_upstream_id: ProblemRecord | str, cpp_source: str
    ) -> CodingVerifierResult:
        ...


class LightCPVerifierExecutor(Protocol):
    """Injected external binding; not implemented or invoked by default."""

    def __call__(
        self,
        upstream_id: str,
        cpp_source: str,
        config: LightCPVerifierConfig,
    ) -> CodingVerifierResult | Mapping[str, object]:
        ...


@dataclass(frozen=True, slots=True)
class LightCPHTTPResponse:
    """Header-free HTTP response used by the injectable verifier transport."""

    status_code: int
    body: Mapping[str, Any]


class LightCPHTTPTransport(Protocol):
    def post_json(
        self, url: str, payload: Mapping[str, Any], *, timeout: float
    ) -> LightCPHTTPResponse:
        ...

    def get_json(self, url: str, *, timeout: float) -> LightCPHTTPResponse:
        ...


class RequestsLightCPTransport:
    """Lazy requests transport used only by explicit production scoring."""

    @staticmethod
    def _response(response: object) -> LightCPHTTPResponse:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, bool) or not isinstance(status_code, int):
            raise RuntimeError("LightCPVerifier response has no HTTP status")
        try:
            body = response.json()  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - requests-specific edge
            if status_code == 404 or not 200 <= status_code < 300:
                body = {}
            else:
                raise RuntimeError("LightCPVerifier returned invalid JSON") from exc
        if not isinstance(body, Mapping):
            raise RuntimeError("LightCPVerifier response root must be an object")
        return LightCPHTTPResponse(status_code=status_code, body=dict(body))

    def post_json(
        self, url: str, payload: Mapping[str, Any], *, timeout: float
    ) -> LightCPHTTPResponse:
        import requests

        try:
            response = requests.post(url, json=dict(payload), timeout=timeout)
        except requests.RequestException as exc:
            raise LightCPVerifierServiceError(
                "LightCPVerifier submit request failed"
            ) from exc
        return self._response(response)

    def get_json(self, url: str, *, timeout: float) -> LightCPHTTPResponse:
        import requests

        try:
            response = requests.get(url, timeout=timeout)
        except requests.RequestException as exc:
            raise LightCPVerifierServiceError(
                "LightCPVerifier result request failed"
            ) from exc
        return self._response(response)


def _optional_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _status_from_verdict(
    verdict: str, accepted: bool
) -> CodingVerifierStatus:
    if accepted:
        return "accepted"
    normalized = re.sub(r"[^a-z0-9]+", "_", verdict.lower()).strip("_")
    if normalized in {"wrong_answer", "rejected", "incorrect"}:
        return "wrong_answer"
    if normalized in {"compilation_error", "compile_error"}:
        return "compilation_error"
    if normalized in {"runtime_error", "runtime_failure"}:
        return "runtime_error"
    if normalized in {"time_limit_exceeded", "judge_timeout", "timeout"}:
        return "time_limit_exceeded"
    if normalized in CODING_VERIFIER_STATUSES:
        return normalized  # type: ignore[return-value]
    return "verifier_error"


def _private_host(host: str) -> bool:
    if host in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        return host.endswith((".internal", ".local"))


def _validate_service_url(
    value: str | None, *, allow_private_runtime: bool
) -> None:
    if value in {None, "unresolved"}:
        return
    if not isinstance(value, str):
        raise LightCPVerifierConfigError(
            "service_url must be unresolved, null, or an HTTP(S) URL"
        )
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise LightCPVerifierConfigError("service_url must be an HTTP(S) URL")
    if _private_host(parsed.hostname) and not allow_private_runtime:
        raise LightCPVerifierConfigError(
            "public verifier templates must not contain private endpoints"
        )


def _scan_public_config(value: Any, path: str = "verifier_config") -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.lower() in _FORBIDDEN_CONFIG_KEYS:
                raise LightCPVerifierConfigError(
                    f"{path} contains forbidden field {key!r}"
                )
            _scan_public_config(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_public_config(item, f"{path}[{index}]")
        return
    if not isinstance(value, str):
        return
    if _CREDENTIAL.search(value):
        raise LightCPVerifierConfigError(
            f"{path} contains an API-key-like value"
        )
    if Path(value).is_absolute() or _WINDOWS_ABSOLUTE.match(value):
        raise LightCPVerifierConfigError(
            f"{path} contains a private absolute path"
        )
    if ".env" in Path(value).parts:
        raise LightCPVerifierConfigError(f"{path} contains a forbidden .env path")


def _is_private_runtime_config(path: Path) -> bool:
    return path.name == "lightcpverifier.local.yaml" or path.name.endswith(
        ".private.yaml"
    )


def validate_lightcpverifier_config(
    config: LightCPVerifierConfig, *, production: bool = False
) -> LightCPVerifierConfig:
    """Validate runtime readiness without exposing path or endpoint values."""

    _validate_service_url(
        config.effective_service_url,
        allow_private_runtime=config.runtime_private_config,
    )
    if production:
        if config.status != "configured":
            raise LightCPVerifierConfigError(
                "production scoring requires a user-specific configured verifier file"
            )
        if config.requires_owner_approval:
            raise LightCPVerifierConfigError(
                "production verifier config still requires owner approval"
            )
        if config.mode == "service" and config.effective_service_url is None:
            raise LightCPVerifierConfigError(
                "service mode requires a configured service_url"
            )
        if config.asset_root_env is None and config.problem_assets_root is None:
            raise LightCPVerifierConfigError(
                "production scoring requires asset_root_env"
            )
    return config


def load_lightcpverifier_config(path: str | Path) -> LightCPVerifierConfig:
    """Load a path-free public template or an ignored user runtime config."""

    config_path = Path(path)
    raw = load_config(config_path)
    if set(raw) != _PUBLIC_CONFIG_FIELDS:
        raise LightCPVerifierConfigError(
            "LightCPVerifier config fields do not match the public schema"
        )
    _scan_public_config(raw)
    private_runtime = _is_private_runtime_config(config_path)
    service_url = raw.get("service_url")
    if service_url is not None and not isinstance(service_url, str):
        raise LightCPVerifierConfigError(
            "service_url must be unresolved, null, or an HTTP(S) URL"
        )
    _validate_service_url(
        service_url, allow_private_runtime=private_runtime
    )
    asset_root_env = raw.get("asset_root_env")
    if asset_root_env is not None and not isinstance(asset_root_env, str):
        raise LightCPVerifierConfigError(
            "asset_root_env must be an environment-variable name or null"
        )
    config = LightCPVerifierConfig(
        verifier_type=str(raw.get("verifier_type", "")),
        mode=str(raw.get("mode", "")),
        service_url=service_url,
        judge_url=service_url if service_url not in {None, "unresolved"} else None,
        asset_root_env=asset_root_env,
        timeout_seconds=float(raw.get("timeout_seconds", 0)),
        max_retries=raw.get("max_retries"),  # type: ignore[arg-type]
        status=str(raw.get("status", "")),
        requires_owner_approval=raw.get("requires_owner_approval"),  # type: ignore[arg-type]
        runtime_private_config=private_runtime,
    )
    return validate_lightcpverifier_config(config)


class LightCPVerifierHTTPExecutor:
    """Formal ``/submit`` + ``/result/{sid}`` LightCPVerifier binding.

    Hidden test details returned by the service are intentionally discarded.
    Only the verdict and aggregate runtime measurements cross this boundary.
    """

    def __init__(
        self,
        transport: LightCPHTTPTransport | None = None,
        *,
        sleep: Any = time.sleep,
        monotonic: Any = time.monotonic,
    ) -> None:
        self._transport = transport or RequestsLightCPTransport()
        self._sleep = sleep
        self._monotonic = monotonic

    @staticmethod
    def _judge_url(config: LightCPVerifierConfig) -> str:
        value = config.effective_service_url
        if not isinstance(value, str) or not value.startswith(("http://", "https://")):
            raise LightCPVerifierConfigError(
                "LightCPVerifier service URL is not configured"
            )
        return value.rstrip("/")

    @staticmethod
    def _require_assets(upstream_id: str, config: LightCPVerifierConfig) -> None:
        root = config.resolved_assets_root()
        if root is None:
            raise LightCPVerifierAssetsError(
                "LightCPVerifier assets are not configured"
            )
        config_path = root / upstream_id / config.asset_config_name
        if not config_path.is_file():
            raise LightCPVerifierAssetsError(
                "external LightCPVerifier assets are unavailable for the requested upstream_id"
            )

    def __call__(
        self,
        upstream_id: str,
        cpp_source: str,
        config: LightCPVerifierConfig,
    ) -> CodingVerifierResult:
        self._require_assets(upstream_id, config)
        base_url = self._judge_url(config)
        submitted: LightCPHTTPResponse | None = None
        for attempt in range(config.max_retries + 1):
            try:
                submitted = self._transport.post_json(
                    f"{base_url}/submit",
                    {"pid": upstream_id, "lang": "cpp", "code": cpp_source},
                    timeout=float(config.submit_timeout_seconds),
                )
                break
            except LightCPVerifierServiceError:
                if attempt >= config.max_retries:
                    raise
        assert submitted is not None
        if not 200 <= submitted.status_code < 300:
            raise RuntimeError("LightCPVerifier rejected the submit request")
        sid = submitted.body.get("sid")
        if not isinstance(sid, (str, int)) or not str(sid):
            raise RuntimeError("LightCPVerifier submit response has no sid")

        deadline = self._monotonic() + float(config.timeout_seconds)
        while self._monotonic() < deadline:
            polled = self._transport.get_json(
                f"{base_url}/result/{sid}",
                timeout=float(config.poll_timeout_seconds),
            )
            if polled.status_code == 404:
                self._sleep(float(config.poll_interval_seconds))
                continue
            if not 200 <= polled.status_code < 300:
                raise RuntimeError("LightCPVerifier result request failed")
            status = polled.body.get("status")
            if status == "queued":
                self._sleep(float(config.poll_interval_seconds))
                continue
            if status == "error":
                message = str(polled.body.get("message") or polled.body.get("error") or "")
                verdict = (
                    "Compilation Error"
                    if "compile failed" in message.lower()
                    else "Judge Error"
                )
                return CodingVerifierResult(
                    upstream_id=upstream_id,
                    accepted=False,
                    verdict=verdict,
                    detail="external verifier reported an execution error; hidden details omitted",
                    time_ms=0.0,
                    memory_mb=0.0,
                    status=(
                        "compilation_error"
                        if verdict == "Compilation Error"
                        else "verifier_error"
                    ),
                )
            verdict = str(polled.body.get("result", "Unknown"))
            return CodingVerifierResult(
                upstream_id=upstream_id,
                accepted=verdict == "Accepted",
                verdict=verdict,
                detail="external verifier completed; hidden test details omitted",
                time_ms=_optional_number(polled.body.get("time")),
                memory_mb=_optional_number(polled.body.get("memory")),
            )

        return CodingVerifierResult(
            upstream_id=upstream_id,
            accepted=False,
            verdict="Judge Timeout",
            detail="external verifier did not finish before the configured timeout",
            status="time_limit_exceeded",
        )


def _require_upstream_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "upstream_id is required to resolve separately installed Coding "
            "verifier assets"
        )
    return value


def _upstream_id(problem_or_upstream_id: ProblemRecord | str) -> str:
    if isinstance(problem_or_upstream_id, ProblemRecord):
        if problem_or_upstream_id.domain != "coding":
            raise ValueError("LightCPVerifierAdapter requires a Coding ProblemRecord")
        value = problem_or_upstream_id.domain_payload.get("upstream_id")
        if not isinstance(value, str):
            value = ""
        return _require_upstream_id(value)
    return _require_upstream_id(problem_or_upstream_id)


class MockCodingVerifier:
    """Deterministic in-memory verifier for interface and CLI tests only."""

    def __init__(self, verdicts: Mapping[str, bool] | None = None) -> None:
        self._verdicts = dict(verdicts) if verdicts is not None else None

    def verify(
        self, problem_or_upstream_id: ProblemRecord | str, cpp_source: str
    ) -> CodingVerifierResult:
        checked_id = _upstream_id(problem_or_upstream_id)
        if not isinstance(cpp_source, str) or not cpp_source.strip():
            return CodingVerifierResult(
                upstream_id=checked_id,
                accepted=False,
                verdict="rejected",
                detail="mock verifier received empty source",
            )
        accepted = (
            self._verdicts.get(checked_id, False)
            if self._verdicts is not None
            else True
        )
        return CodingVerifierResult(
            upstream_id=checked_id,
            accepted=accepted,
            verdict="accepted" if accepted else "rejected",
            detail="mock result; no compiler or hidden tests were run",
        )


class LightCPVerifierAdapter:
    """Safe adapter boundary for an explicitly configured external verifier."""

    def __init__(
        self,
        config: LightCPVerifierConfig | None = None,
        *,
        executor: LightCPVerifierExecutor | None = None,
    ) -> None:
        self.config = config or LightCPVerifierConfig()
        self._executor = executor

    def _validate_external_configuration(self) -> tuple[Path | None, Path]:
        verifier_root = self.config.verifier_root
        assets_root = self.config.resolved_assets_root()
        if assets_root is None or (
            verifier_root is None and self.config.effective_service_url is None
        ):
            raise RuntimeError(
                "LightCPVerifierAdapter is not configured. Provide explicit "
                "problem_assets_root and verifier_root or judge_url at runtime; hidden "
                "tests are not stored in the public R3Bench dataset."
            )
        if verifier_root is not None and not verifier_root.exists():
            raise FileNotFoundError("configured verifier_root does not exist")
        if not assets_root.exists():
            raise FileNotFoundError("configured problem_assets_root does not exist")
        return verifier_root, assets_root

    def verify(
        self, problem_or_upstream_id: ProblemRecord | str, cpp_source: str
    ) -> CodingVerifierResult:
        upstream_id = _upstream_id(problem_or_upstream_id)
        if not isinstance(cpp_source, str) or not cpp_source.strip():
            return CodingVerifierResult(
                upstream_id=upstream_id,
                accepted=False,
                verdict="missing_solution",
                detail="empty C++ source",
                status="missing_solution",
            )
        self._validate_external_configuration()
        if self._executor is None:
            raise NotImplementedError(
                "No LightCPVerifier executor is bound. Install and configure the "
                "external verifier explicitly; hidden tests never run by default."
            )
        raw = self._executor(upstream_id, cpp_source, self.config)
        time_ms = None
        memory_mb = None
        if isinstance(raw, CodingVerifierResult):
            accepted, verdict = raw.accepted, raw.verdict
            time_ms, memory_mb = raw.time_ms, raw.memory_mb
        elif isinstance(raw, Mapping):
            accepted = raw.get("accepted")
            verdict = raw.get("verdict")
            if not isinstance(accepted, bool) or not isinstance(verdict, str):
                raise RuntimeError("external verifier returned an invalid result")
            time_ms = _optional_number(raw.get("time_ms"))
            memory_mb = _optional_number(raw.get("memory_mb"))
        else:
            raise RuntimeError("external verifier returned an invalid result")
        return CodingVerifierResult(
            upstream_id=upstream_id,
            accepted=accepted,
            verdict=verdict,
            detail="external verifier completed; runtime asset paths are omitted",
            time_ms=time_ms,
            memory_mb=memory_mb,
        )


class LightCPVerifierServiceClient(LightCPVerifierHTTPExecutor):
    """Named service-mode client for the public integration contract."""


class LightCPVerifierLocalClient:
    """Placeholder boundary for user-supplied local executable bindings."""

    def __call__(
        self,
        upstream_id: str,
        cpp_source: str,
        config: LightCPVerifierConfig,
    ) -> CodingVerifierResult:
        del upstream_id, cpp_source, config
        raise LightCPVerifierConfigError(
            "local mode requires a separately installed client binding; "
            "the public release does not start a verifier executable"
        )


def verify_saved_solution(
    problem: ProblemRecord,
    parsed_answer: str | None,
    config: LightCPVerifierConfig | None,
    *,
    executor: LightCPVerifierExecutor | None = None,
) -> CodingVerifierResult:
    """Score one saved C++ artifact with sanitized operational failures."""

    upstream_id = _upstream_id(problem)
    if not isinstance(parsed_answer, str) or not parsed_answer.strip():
        return CodingVerifierResult(
            upstream_id=upstream_id,
            accepted=False,
            verdict="missing_solution",
            detail="no parsed C++ solution was supplied",
            status="missing_solution",
        )
    if config is None:
        return CodingVerifierResult(
            upstream_id=upstream_id,
            accepted=False,
            verdict="not_configured",
            detail="external verifier configuration was not supplied",
            status="not_configured",
        )
    try:
        validate_lightcpverifier_config(config, production=True)
        selected_executor = executor
        if selected_executor is None:
            selected_executor = (
                LightCPVerifierServiceClient()
                if config.mode == "service"
                else LightCPVerifierLocalClient()
            )
        return LightCPVerifierAdapter(
            config, executor=selected_executor
        ).verify(problem, parsed_answer)
    except LightCPVerifierAssetsError:
        status: CodingVerifierStatus = "assets_unavailable"
    except LightCPVerifierServiceError:
        status = "service_unreachable"
    except (LightCPVerifierConfigError, ValueError):
        status = "invalid_config"
    except (FileNotFoundError, NotADirectoryError):
        status = "assets_unavailable"
    except (OSError, RuntimeError):
        status = "verifier_error"
    return CodingVerifierResult(
        upstream_id=upstream_id,
        accepted=False,
        verdict=status,
        detail="external verifier could not produce a score; private details omitted",
        status=status,
    )


__all__ = [
    "CodingVerifier",
    "CodingVerifierResult",
    "CodingVerifierStatus",
    "CODING_VERIFIER_STATUSES",
    "LightCPVerifierAdapter",
    "LightCPVerifierAssetsError",
    "LightCPVerifierConfig",
    "LightCPVerifierConfigError",
    "LightCPVerifierError",
    "LightCPVerifierExecutor",
    "LightCPVerifierHTTPExecutor",
    "LightCPVerifierLocalClient",
    "LightCPVerifierServiceClient",
    "LightCPVerifierServiceError",
    "LightCPHTTPResponse",
    "LightCPHTTPTransport",
    "MockCodingVerifier",
    "RequestsLightCPTransport",
    "load_lightcpverifier_config",
    "validate_lightcpverifier_config",
    "verify_saved_solution",
]
