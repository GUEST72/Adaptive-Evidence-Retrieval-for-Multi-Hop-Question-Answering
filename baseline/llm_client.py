"""Provider-agnostic entry point for the QA reader.

Which backend answers is a config choice (`provider:` in configs/*.yaml), not a
code change: qa_pipeline.py only ever calls `call_llm`. See baseline/providers.py
for the individual backends.

Responses are cached on disk (baseline/llm_cache.py) because free tiers meter a
daily token budget, so paying twice for the same prompt is the most expensive
mistake available.
"""

from __future__ import annotations

from dotenv import load_dotenv

from baseline import llm_cache
from baseline.providers import (
    PROVIDERS,
    DailyTokenLimitExceeded,
    ProviderUnavailable,
    reset_clients,
)

load_dotenv()

DEFAULT_PROVIDER = "groq"

__all__ = [
    "call_llm",
    "set_cache_enabled",
    "DailyTokenLimitExceeded",
    "ProviderUnavailable",
    "reset_clients",
]

_cache_enabled = True


def set_cache_enabled(enabled: bool) -> None:
    global _cache_enabled
    _cache_enabled = enabled


def call_llm(
    prompt: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    provider: str = DEFAULT_PROVIDER,
) -> str:
    try:
        complete = PROVIDERS[provider]
    except KeyError:
        raise ValueError(
            f"Unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}."
        ) from None

    # Provider is part of the key: the same prompt answered by a different
    # backend is a different answer, and must not be served from this entry.
    key = llm_cache.cache_key(
        provider=provider,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if _cache_enabled:
        cached = llm_cache.get(key)
        if cached is not None:
            return cached

    text = complete(prompt, model, max_tokens, temperature)

    if _cache_enabled:
        llm_cache.put(key, text, f"{provider}:{model}")
    return text
