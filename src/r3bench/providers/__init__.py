"""Opt-in provider transports behind the shared R3Bench evaluator interface."""

from r3bench.providers.base import ProviderAdapter
from r3bench.providers.errors import (
    ProviderAuthError,
    ProviderConfigError,
    ProviderRequestError,
    ProviderResponseError,
    ProviderRetryError,
)
from r3bench.providers.openai_compatible import OpenAICompatibleAdapter
from r3bench.providers.registry import (
    create_provider_adapter,
    list_provider_adapters,
    load_provider_profile,
    validate_provider_profile,
)

__all__ = [
    "OpenAICompatibleAdapter",
    "ProviderAdapter",
    "ProviderAuthError",
    "ProviderConfigError",
    "ProviderRequestError",
    "ProviderResponseError",
    "ProviderRetryError",
    "create_provider_adapter",
    "list_provider_adapters",
    "load_provider_profile",
    "validate_provider_profile",
]
