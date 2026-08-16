"""Provider-neutral request/response objects and offline provider doubles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, TypeAlias, runtime_checkable

from r3bench.common.io import read_jsonl


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


MessageRole: TypeAlias = Literal["system", "user", "assistant"]
_MESSAGE_ROLES = frozenset({"system", "user", "assistant"})


@dataclass(frozen=True, slots=True)
class Message:
    """One provider-neutral visible chat message."""

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if self.role not in _MESSAGE_ROLES:
            raise ValueError(f"unsupported message role: {self.role!r}")
        _nonempty(self.content, "message.content")


@dataclass(frozen=True, slots=True)
class UsageInfo:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        for field_name in ("input_tokens", "output_tokens", "reasoning_tokens"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens


@dataclass(frozen=True, slots=True)
class TransientStageHandoff:
    """Provider-exposed Stage 1 channels retained only for immediate handoff.

    The reasoning channel is intentionally excluded from ``repr`` and is never
    part of the public result schema. Consumers must hash or assemble it in
    memory and then discard it.
    """

    reasoning_content: str = field(default="", repr=False)
    visible_output: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.reasoning_content, str):
            raise ValueError("reasoning_content must be a string")
        if not isinstance(self.visible_output, str):
            raise ValueError("visible_output must be a string")

    @property
    def has_content(self) -> bool:
        return bool(self.reasoning_content.strip() or self.visible_output.strip())


@dataclass(frozen=True, slots=True, init=False)
class ModelRequest:
    request_id: str
    model: str
    max_tokens: int | None
    prompt_text: str | None
    messages: tuple[Message, ...]
    temperature: float | None = 0.0
    top_p: float | None = None
    metadata: Mapping[str, Any] = MappingProxyType({})
    allow_mixed_input: bool = False

    def __init__(
        self,
        request_id: str,
        model: str,
        prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = 0.0,
        metadata: Mapping[str, Any] = MappingProxyType({}),
        *,
        top_p: float | None = None,
        prompt_text: str | None = None,
        messages: Sequence[Message] = (),
        allow_mixed_input: bool = False,
    ) -> None:
        """Create a request from text or structured visible messages.

        ``prompt`` is a backward-compatible alias for ``prompt_text``. New code
        should use ``prompt_text`` or ``messages``. Mixed input is rejected by
        default and is not used by the public two-stage protocol.
        """

        if prompt is not None:
            if prompt_text is not None:
                raise ValueError("provide prompt or prompt_text, not both")
            prompt_text = prompt
        checked_messages = tuple(messages)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "max_tokens", max_tokens)  # type: ignore[arg-type]
        object.__setattr__(self, "prompt_text", prompt_text)
        object.__setattr__(self, "messages", checked_messages)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "top_p", top_p)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "allow_mixed_input", allow_mixed_input)
        self._validate()

    def _validate(self) -> None:
        _nonempty(self.request_id, "request_id")
        _nonempty(self.model, "model")
        if self.max_tokens is not None:
            if isinstance(self.max_tokens, bool) or not isinstance(
                self.max_tokens, int
            ):
                raise ValueError("max_tokens must be an integer or null")
            if self.max_tokens < 0:
                raise ValueError("max_tokens must be non-negative when provided")
        if self.temperature is not None and (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
        ):
            raise ValueError("temperature must be numeric or null")
        if self.top_p is not None:
            if not isinstance(self.top_p, (int, float)) or isinstance(self.top_p, bool):
                raise ValueError("top_p must be numeric or null")
            if not 0 <= float(self.top_p) <= 1:
                raise ValueError("top_p must be between 0 and 1")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        if not isinstance(self.allow_mixed_input, bool):
            raise ValueError("allow_mixed_input must be boolean")
        if self.prompt_text is not None:
            _nonempty(self.prompt_text, "prompt_text")
        for message in self.messages:
            if not isinstance(message, Message):
                raise ValueError("messages must contain Message objects")
        if self.prompt_text is None and not self.messages:
            raise ValueError("request must contain prompt_text or messages")
        if self.prompt_text is not None and self.messages and not self.allow_mixed_input:
            raise ValueError(
                "request cannot contain both prompt_text and messages unless "
                "allow_mixed_input=True"
            )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def prompt(self) -> str | None:
        """Backward-compatible alias for the old single-prompt field."""

        return self.prompt_text

    @property
    def input_kind(self) -> str:
        if self.prompt_text is not None and self.messages:
            return "mixed"
        if self.messages:
            return "messages"
        return "prompt_text"

    def canonical_input(self) -> str:
        """Return a deterministic representation for request hashing."""

        payload = {
            "prompt_text": self.prompt_text,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in self.messages
            ],
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def input_sha256(self) -> str:
        return hashlib.sha256(self.canonical_input().encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelResponse:
    request_id: str
    model: str
    response_text: str
    usage: UsageInfo
    finish_reason: str = "stop"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _nonempty(self.request_id, "request_id")
        _nonempty(self.model, "model")
        if not isinstance(self.response_text, str):
            raise ValueError("response_text must be a string")
        _nonempty(self.finish_reason, "finish_reason")
        if not isinstance(self.usage, UsageInfo):
            raise ValueError("usage must be UsageInfo")

@runtime_checkable
class Provider(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse:
        ...


class MockProvider:
    """Return deterministic canned text without any external side effect."""

    def __init__(
        self,
        responses: str | Mapping[str, str],
        *,
        usage: UsageInfo | None = None,
        finish_reason: str = "stop",
    ) -> None:
        if isinstance(responses, str):
            self._default = responses
            self._responses: dict[str, str] = {}
        else:
            self._default = None
            self._responses = dict(responses)
        self._usage = usage or UsageInfo()
        self._finish_reason = finish_reason

    def complete(self, request: ModelRequest) -> ModelResponse:
        if request.request_id in self._responses:
            output = self._responses[request.request_id]
        elif self._default is not None:
            output = self._default
        else:
            raise KeyError(f"no mock response for request_id {request.request_id!r}")
        if not isinstance(output, str):
            raise ValueError("mock response values must be strings")
        return ModelResponse(
            request_id=request.request_id,
            model=request.model,
            response_text=output,
            usage=self._usage,
            finish_reason=self._finish_reason,
        )


class ReplayProvider:
    """Replay allowlisted response fields from a local JSONL file by request ID."""

    def __init__(self, path: str) -> None:
        rows = read_jsonl(path)
        self._responses: dict[str, ModelResponse] = {}
        self._responses_by_suffix: dict[str, ModelResponse | None] = {}
        for row_number, row in enumerate(rows, start=1):
            request_id = _nonempty(row.get("request_id"), f"row {row_number} request_id")
            if request_id in self._responses:
                raise ValueError(f"duplicate replay request_id: {request_id!r}")
            usage_value = row.get("usage", {})
            if not isinstance(usage_value, Mapping):
                raise ValueError(f"row {row_number} usage must be a mapping")
            usage = UsageInfo(
                input_tokens=usage_value.get("input_tokens", 0),
                output_tokens=usage_value.get("output_tokens", 0),
                reasoning_tokens=usage_value.get("reasoning_tokens", 0),
            )
            response_text = row.get("response_text")
            if response_text is None:
                response_text = ""
            self._responses[request_id] = ModelResponse(
                request_id=request_id,
                model=_nonempty(row.get("model"), f"row {row_number} model"),
                response_text=response_text,
                usage=usage,
                finish_reason=row.get("finish_reason", "stop"),
            )
            suffix = self._stable_request_suffix(request_id)
            if suffix is not None:
                if suffix in self._responses_by_suffix:
                    self._responses_by_suffix[suffix] = None
                else:
                    self._responses_by_suffix[suffix] = self._responses[request_id]

    @staticmethod
    def _stable_request_suffix(request_id: str) -> str | None:
        """Return the stable ``stage:item`` portion of a generated request ID."""

        parts = request_id.split(":", 1)
        if len(parts) != 2 or ":" not in parts[1]:
            return None
        return parts[1]

    def complete(self, request: ModelRequest) -> ModelResponse:
        replay = self._responses.get(request.request_id)
        if replay is None:
            suffix = self._stable_request_suffix(request.request_id)
            replay = self._responses_by_suffix.get(suffix) if suffix is not None else None
        if replay is None:
            raise KeyError(f"no unique replay response for request_id {request.request_id!r}")
        if replay.model != request.model:
            raise ValueError(
                f"replay model mismatch for {request.request_id!r}: "
                f"expected {request.model!r}, found {replay.model!r}"
            )
        return ModelResponse(
            request_id=request.request_id,
            model=replay.model,
            response_text=replay.response_text,
            usage=replay.usage,
            finish_reason=replay.finish_reason,
        )


__all__ = [
    "Message",
    "MockProvider",
    "ModelRequest",
    "ModelResponse",
    "Provider",
    "ReplayProvider",
    "TransientStageHandoff",
    "UsageInfo",
]
