"""Minimal OpenAI-compatible HTTP adapter with injectable transport."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Mapping, Protocol

from r3bench.common.provider import (
    ModelRequest,
    ModelResponse,
    TransientStageHandoff,
    UsageInfo,
)
from r3bench.providers.base import ProviderAdapter
from r3bench.providers.errors import (
    ProviderAuthError,
    ProviderConfigError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderRetryError,
)


_UNRESOLVED = "unresolved"
_INLINE_THINK = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_CORE_PAYLOAD_FIELDS = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "reasoning_effort",
        "stream",
    }
)


class HTTPTransport(Protocol):
    def post(
        self,
        url: str,
        *,
        json: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: int | float,
    ) -> object:
        ...


class RequestsTransport:
    """Lazy requests-based transport used only by explicit real execution."""

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, Any],
        headers: Mapping[str, str],
        timeout: int | float,
    ) -> object:
        import requests

        try:
            return requests.post(
                url,
                json=dict(json),
                headers=dict(headers),
                timeout=timeout,
            )
        except requests.Timeout as exc:
            raise TimeoutError("provider request timed out") from exc
        except requests.RequestException as exc:
            raise ProviderRequestError("provider transport failed") from exc


def _field(value: Mapping[str, Any] | object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _resolved(value: Any) -> bool:
    return value is not None and value != _UNRESOLVED


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


class OpenAICompatibleAdapter(ProviderAdapter):
    """OpenAI-compatible chat-completions transport with no provider SDK."""

    adapter_name = "openai_compatible"

    def __init__(
        self,
        provider_profile: Mapping[str, Any],
        model_profile: Mapping[str, Any] | object,
        *,
        transport: HTTPTransport | None = None,
        transport_config: Mapping[str, Any] | None = None,
        allow_unresolved: bool = False,
    ) -> None:
        self.validate_config(provider_profile, model_profile)
        self._provider = dict(provider_profile)
        self._model = model_profile
        self._transport = transport or RequestsTransport()
        from r3bench.common.profile_registry import validate_transport_mapping

        self._transport_config = validate_transport_mapping(
            transport_config
            if transport_config is not None
            else provider_profile.get("transport_defaults"),
            path="effective_transport",
            require_all=True,
        )
        self._allow_unresolved = allow_unresolved
        self._stage_handoff_cache: dict[str, TransientStageHandoff] = {}

    @classmethod
    def validate_config(
        cls,
        provider_profile: Mapping[str, Any],
        model_profile: Mapping[str, Any] | object,
    ) -> None:
        from r3bench.providers.registry import validate_provider_profile

        validate_provider_profile(provider_profile)
        if provider_profile.get("adapter") != cls.adapter_name:
            raise ProviderConfigError("provider profile does not select openai_compatible")
        if not isinstance(provider_profile.get("provider_name"), str):
            raise ProviderConfigError("provider_name must be a string")
        if not isinstance(_field(model_profile, "model_key"), str):
            raise ProviderConfigError("model profile must contain model_key")
        if not isinstance(_field(model_profile, "public_model_id"), str):
            raise ProviderConfigError("model profile must contain public_model_id")

    @property
    def execution_ready(self) -> bool:
        resolved = all(
            _resolved(value)
            for value in (
                self._provider.get("base_url"),
                *(self._transport_config.get(field) for field in (
                    "timeout_seconds",
                    "max_retries",
                    "retry_backoff_seconds",
                    "retry_backoff_mode",
                    "streaming",
                    "request_safety_limits",
                )),
                _field(self._model, "public_model_id"),
            )
        )
        return resolved and self._transport_config.get("streaming") is False

    def _model_id(self, allow_unresolved: bool) -> str:
        value = _field(self._model, "public_model_id")
        if _resolved(value):
            return str(value)
        if allow_unresolved:
            return _UNRESOLVED
        raise ProviderConfigError("public model identifier is unresolved")

    def build_payload(
        self,
        request: ModelRequest,
        *,
        allow_unresolved: bool | None = None,
    ) -> dict[str, Any]:
        """Build a header-free request body for execution or dry-run review."""

        allow = self._allow_unresolved if allow_unresolved is None else allow_unresolved
        if request.messages:
            messages = [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ]
        else:
            messages = [{"role": "user", "content": request.prompt_text}]
        payload: dict[str, Any] = {
            "model": self._model_id(allow),
            "messages": messages,
            "stream": False,
        }
        model_temperature = _field(self._model, "temperature")
        if _resolved(model_temperature) and request.temperature is not None:
            payload["temperature"] = float(request.temperature)
        token_budget_field = self._provider.get(
            "token_budget_field", "max_tokens"
        )
        if token_budget_field not in {"max_tokens", "max_completion_tokens"}:
            raise ProviderConfigError("provider token budget field is unsupported")
        if request.max_tokens is not None:
            payload[token_budget_field] = request.max_tokens

        model_top_p = _field(self._model, "top_p")
        if _resolved(model_top_p):
            top_p = request.top_p if request.top_p is not None else model_top_p
            payload["top_p"] = float(top_p)
        reasoning_effort = _field(self._model, "reasoning_effort")
        if _resolved(reasoning_effort):
            payload["reasoning_effort"] = reasoning_effort

        extra = dict(self._provider.get("extra_body") or {})
        thinking = _field(self._model, "thinking_enabled")
        if isinstance(thinking, bool):
            extra["enable_thinking"] = thinking
        collision = _CORE_PAYLOAD_FIELDS & set(extra)
        if collision:
            raise ProviderConfigError("extra_body cannot override core request fields")
        payload.update(extra)
        return payload

    def _execution_config(self) -> tuple[str, int | float, int, str, float, str]:
        if not self.execution_ready:
            raise ProviderConfigError("provider or model execution fields are unresolved")
        base_url = str(self._provider["base_url"]).rstrip("/")
        timeout = self._transport_config["timeout_seconds"]
        retries = self._transport_config["max_retries"]
        backoff = self._transport_config["retry_backoff_seconds"]
        backoff_mode = self._transport_config["retry_backoff_mode"]
        env_name = self._provider.get("api_key_env")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ProviderConfigError("timeout_seconds must be positive")
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ProviderConfigError("max_retries must be non-negative")
        if isinstance(backoff, bool) or not isinstance(backoff, (int, float)) or backoff < 0:
            raise ProviderConfigError("retry_backoff_seconds must be non-negative")
        if backoff_mode not in {"fixed", "exponential", "none"}:
            raise ProviderConfigError("retry_backoff_mode is unsupported")
        if self._transport_config.get("streaming") is not False:
            raise ProviderConfigError("streaming transport is not implemented")
        if not isinstance(self._transport_config.get("request_safety_limits"), Mapping):
            raise ProviderConfigError("request_safety_limits must be resolved")
        if not isinstance(env_name, str) or not env_name:
            raise ProviderConfigError("api_key_env must name an environment variable")
        return base_url, timeout, retries, env_name, float(backoff), str(backoff_mode)

    @staticmethod
    def _retry_delay(base: float, mode: str, attempt: int) -> float:
        if mode == "none":
            return 0.0
        if mode == "exponential":
            return base * (2**attempt)
        return base

    @staticmethod
    def _response_body(response: object) -> Mapping[str, Any]:
        try:
            value = response.json()  # type: ignore[attr-defined]
        except Exception as exc:
            raise ProviderResponseError("provider response is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise ProviderResponseError("provider response root must be an object")
        return value

    @staticmethod
    def _response_channels(
        choice: Mapping[str, Any],
    ) -> tuple[str, str]:
        """Return visible and transient reasoning channels from one choice.

        Some formal OpenAI-compatible endpoints expose ``reasoning_content``.
        Others place the same material inside visible ``<think>`` tags. The
        latter fallback mirrors the formal Math/AR runners and removes the
        thinking block from the scoreable visible response.
        """

        message = choice.get("message")
        if isinstance(message, Mapping):
            visible = message.get("content")
            reasoning = message.get("reasoning_content")
            if visible is None:
                visible = ""
        else:
            visible = choice.get("text")
            reasoning = None
        if not isinstance(visible, str):
            raise ProviderResponseError("provider response has no visible text")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ProviderResponseError("provider reasoning_content must be text")
        reasoning_text = reasoning or ""
        if not reasoning_text.strip():
            inline = _INLINE_THINK.findall(visible)
            if inline:
                reasoning_text = "\n\n".join(
                    block.strip() for block in inline if block.strip()
                )
                visible = _INLINE_THINK.sub("", visible).strip()
        return visible, reasoning_text

    @classmethod
    def _parse_response(
        cls,
        request: ModelRequest,
        body: Mapping[str, Any],
    ) -> ModelResponse:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise ProviderResponseError("provider response has no valid choices")
        choice = choices[0]
        text, _ = cls._response_channels(choice)

        usage_value = body.get("usage", {})
        if not isinstance(usage_value, Mapping):
            raise ProviderResponseError("provider usage must be an object")
        input_tokens = _nonnegative_int(
            usage_value.get("prompt_tokens", usage_value.get("input_tokens", 0))
        )
        completion_tokens = _nonnegative_int(
            usage_value.get("completion_tokens", usage_value.get("output_tokens", 0))
        )
        details = usage_value.get("completion_tokens_details", {})
        reasoning_tokens = (
            _nonnegative_int(details.get("reasoning_tokens", 0))
            if isinstance(details, Mapping)
            else 0
        )
        if "completion_tokens" in usage_value:
            visible_output_tokens = max(completion_tokens - reasoning_tokens, 0)
        else:
            visible_output_tokens = completion_tokens
        finish_reason = choice.get("finish_reason", "stop")
        if not isinstance(finish_reason, str) or not finish_reason:
            finish_reason = "unknown"
        return ModelResponse(
            request_id=request.request_id,
            model=request.model,
            response_text=text,
            usage=UsageInfo(
                input_tokens=input_tokens,
                output_tokens=visible_output_tokens,
                reasoning_tokens=reasoning_tokens,
            ),
            finish_reason=finish_reason,
        )

    def _retain_stage_handoff(
        self,
        request: ModelRequest,
        body: Mapping[str, Any],
        response: ModelResponse,
    ) -> None:
        """Retain hidden reasoning only in memory for an immediate Stage 2 call."""

        choices = body.get("choices")
        if (
            isinstance(choices, list)
            and choices
            and isinstance(choices[0], Mapping)
        ):
            _, reasoning = self._response_channels(choices[0])
            if reasoning.strip():
                self._stage_handoff_cache[request.request_id] = (
                    TransientStageHandoff(
                        reasoning_content=reasoning,
                        visible_output=response.response_text,
                    )
                )

    def consume_stage_handoff(
        self,
        request_id: str,
        visible_output: str,
    ) -> TransientStageHandoff:
        """Return and erase the transient Stage 1 handoff material."""

        return self._stage_handoff_cache.pop(
            request_id,
            TransientStageHandoff(visible_output=visible_output),
        )

    def generate(self, request: ModelRequest) -> ModelResponse:
        base_url, timeout, retries, env_name, backoff, backoff_mode = (
            self._execution_config()
        )
        api_key = os.environ.get(env_name)
        if not api_key:
            raise ProviderAuthError("required provider credential is not available")

        payload = self.build_payload(request, allow_unresolved=False)
        headers = dict(self._provider.get("default_headers") or {})
        headers["Authorization"] = f"Bearer {api_key}"
        headers.setdefault("Content-Type", "application/json")
        endpoint = f"{base_url}/chat/completions"
        last_retryable: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = self._transport.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=timeout,
                )
            except (TimeoutError, ProviderRequestError) as exc:
                last_retryable = exc
                if attempt < retries:
                    delay = self._retry_delay(backoff, backoff_mode, attempt)
                    if delay:
                        time.sleep(delay)
                    continue
                raise ProviderRetryError("provider retries were exhausted") from exc
            except Exception as exc:
                last_retryable = exc
                if attempt < retries:
                    delay = self._retry_delay(backoff, backoff_mode, attempt)
                    if delay:
                        time.sleep(delay)
                    continue
                raise ProviderRetryError("provider retries were exhausted") from exc

            status = getattr(response, "status_code", None)
            if status in {401, 403}:
                raise ProviderAuthError("provider rejected authentication")
            if status == 429 or (isinstance(status, int) and status >= 500):
                last_retryable = ProviderRequestError("provider returned a retryable status")
                if attempt < retries:
                    delay = self._retry_delay(backoff, backoff_mode, attempt)
                    if delay:
                        time.sleep(delay)
                    continue
                raise ProviderRetryError("provider retries were exhausted")
            if not isinstance(status, int) or status < 200 or status >= 300:
                raise ProviderRequestError("provider returned a non-success status")
            body = self._response_body(response)
            parsed = self._parse_response(request, body)
            self._retain_stage_handoff(request, body, parsed)
            return parsed

        raise ProviderRetryError("provider retries were exhausted") from last_retryable


__all__ = [
    "HTTPTransport",
    "OpenAICompatibleAdapter",
    "RequestsTransport",
]
