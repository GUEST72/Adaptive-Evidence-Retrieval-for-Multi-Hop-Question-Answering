"""Groq OpenAI-compatible client.

Importing this module never loads a dotenv file or contacts the network.
"""
from __future__ import annotations

import httpx

from .client import (AuthenticationError, BadRequestError, LLMRequest,
                     LLMResponse, ProviderError, RateLimitError)
from .key_manager import KeyManager, NoUsableKeyError


class GroqClient:
    provider = "groq"
    url = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        model: str = "qwen/qwen3.8-27b",
        *,
        key_manager: KeyManager | None = None,
        timeout: float = 120.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.key_manager = key_manager or KeyManager()
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload = {
            "model": request.model or self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }
        attempts = 0
        while attempts < max(1, self.key_manager.key_count):
            key = self.key_manager.get()
            attempts += 1
            try:
                response = self._client.post(
                    self.url, json=payload, headers={"Authorization": "Bearer " + key}
                )
            except httpx.TransportError as exc:
                raise ProviderError("Groq network failure.", kind="network", retryable=True) from exc
            if response.status_code == 429:
                self.key_manager.mark_rate_limited(key)
                continue
            if response.status_code in (401, 403):
                raise AuthenticationError()
            if 400 <= response.status_code < 500:
                detail = "Provider rejected the request."
                try:
                    detail = response.json().get("error", {}).get("message", detail)
                except (TypeError, ValueError):
                    pass
                raise BadRequestError(detail)
            if response.status_code >= 500:
                raise ProviderError("Groq server failure.", kind="server", retryable=True)
            text = response.json()["choices"][0]["message"].get("content", "") or ""
            return LLMResponse(text=text.strip(), model=payload["model"], provider=self.provider)
        raise NoUsableKeyError("All configured API keys were rate limited.")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
