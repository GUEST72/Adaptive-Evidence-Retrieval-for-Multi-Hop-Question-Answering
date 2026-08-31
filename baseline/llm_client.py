"""Provider-agnostic entry point for the QA reader.

Which backend answers is a config choice (`provider:` in configs/*.yaml), not a
code change: qa_pipeline.py only ever calls `call_llm`. See baseline/providers.py
for the individual backends.

Responses are cached on disk (baseline/llm_cache.py) because free tiers meter a
daily token budget, so paying twice for the same prompt is the most expensive
mistake available.
"""

from __future__ import annotations

<<<<<<< Updated upstream
=======
import os
import time

import httpx
>>>>>>> Stashed changes
from dotenv import load_dotenv

from baseline import llm_cache
from baseline.providers import (
    PROVIDERS,
    DailyTokenLimitExceeded,
    ProviderUnavailable,
    reset_clients,
)

load_dotenv()

<<<<<<< Updated upstream
DEFAULT_PROVIDER = "groq"
=======
API_URL = "https://api.groq.com/openai/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_ATTEMPTS = 6
RETRY_STATUS_CODES = {429, 500, 502, 503, 529}
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
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
=======
    # A full sweep is ~900 calls, so rate limiting is expected rather than
    # exceptional; back off instead of losing the whole run to one 429.
    for attempt in range(MAX_ATTEMPTS):
        response = client.post(API_URL, json=payload)

        if response.status_code in RETRY_STATUS_CODES and attempt < MAX_ATTEMPTS - 1:
            retry_after = response.headers.get("retry-after")
            delay = float(retry_after) if retry_after else 2.0**attempt
            time.sleep(min(delay, 60.0))
            continue

        response.raise_for_status()
        return (response.json()["choices"][0]["message"]["content"] or "").strip()

    raise RuntimeError(f"Groq API still failing after {MAX_ATTEMPTS} attempts.")
>>>>>>> Stashed changes
