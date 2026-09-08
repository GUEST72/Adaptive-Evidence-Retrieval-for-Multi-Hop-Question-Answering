"""Generic protocol and a thin retry-free client abstraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    model: str
    max_tokens: int = 512
    temperature: float = 0.0


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    provider: str


class LLMClient(Protocol):
    provider: str

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Complete a request without prescribing transport details."""


class ProviderError(RuntimeError):
    """Classified provider failure; only rate limits are safe to rotate."""

    def __init__(self, message: str, *, kind: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


class RateLimitError(ProviderError):
    def __init__(self, message: str = "Provider rate limit or quota exceeded.") -> None:
        super().__init__(message, kind="rate_limit", retryable=True)


class AuthenticationError(ProviderError):
    def __init__(self, message: str = "Provider authentication failed.") -> None:
        super().__init__(message, kind="auth")


class BadRequestError(ProviderError):
    def __init__(self, message: str = "Provider rejected the request.") -> None:
        super().__init__(message, kind="bad_request")
