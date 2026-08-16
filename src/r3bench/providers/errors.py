"""Sanitized provider-adapter error hierarchy."""


class ProviderError(RuntimeError):
    """Base class for provider failures with secret-free messages."""


class ProviderConfigError(ProviderError):
    """Provider or model configuration is invalid or unresolved."""


class ProviderAuthError(ProviderError):
    """The configured environment variable does not contain a credential."""


class ProviderRequestError(ProviderError):
    """A provider request failed before a valid response was available."""


class ProviderResponseError(ProviderError):
    """A provider returned an unsupported or malformed response."""


class ProviderRetryError(ProviderRequestError):
    """Retryable transport failures exhausted the configured retry count."""


__all__ = [
    "ProviderAuthError",
    "ProviderConfigError",
    "ProviderError",
    "ProviderRequestError",
    "ProviderResponseError",
    "ProviderRetryError",
]
