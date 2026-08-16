"""Base interface for opt-in model-provider transports."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from r3bench.common.provider import ModelRequest, ModelResponse


class ProviderAdapter(ABC):
    """Transport adapter used by the single shared evaluator runner."""

    adapter_name: str

    @classmethod
    @abstractmethod
    def validate_config(
        cls,
        provider_profile: Mapping[str, Any],
        model_profile: Mapping[str, Any] | object,
    ) -> None:
        """Validate safe configuration without reading credentials."""

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Execute one request after explicit caller opt-in."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Compatibility method consumed by the shared NL runner."""

        return self.generate(request)


__all__ = ["ProviderAdapter"]
