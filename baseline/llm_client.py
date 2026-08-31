"""Thin wrapper around the Groq chat-completions API. Model name/config comes
from configs/baseline.yaml, never hardcoded here. API key comes from .env only.

Groq exposes an OpenAI-compatible endpoint, so this is a plain POST rather than
a vendor SDK — one less dependency, and the call shape stays obvious.

Responses are cached on disk (see baseline/llm_cache.py) because the free tier's
real constraint is a daily token budget, so paying twice for the same prompt is
the most expensive mistake available.
"""

from __future__ import annotations

import os
import re
import time

import httpx
from dotenv import load_dotenv

from baseline import llm_cache

load_dotenv()

API_URL = "https://api.groq.com/openai/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_ATTEMPTS = 8
RETRY_STATUS_CODES = {429, 500, 502, 503, 529}
MAX_SLEEP_SECONDS = 90.0

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

_client: httpx.Client | None = None
_cache_enabled = True


class DailyTokenLimitExceeded(RuntimeError):
    """The account's tokens-per-day allowance is gone.

    Worth its own type: unlike a per-minute limit this cannot be waited out in
    any useful time, so retrying only burns wall-clock. Note the TPD counter is
    reported *only* in the 429 body — no response header exposes it.
    """


def set_cache_enabled(enabled: bool) -> None:
    global _cache_enabled
    _cache_enabled = enabled


def _parse_duration(text: str | None) -> float:
    """Parse Groq's duration headers, e.g. '667ms', '7.66s', '7h40m48s', '73'."""
    if not text:
        return 0.0
    parsed = sum(float(value) * _UNIT_SECONDS[unit] for value, unit in _DURATION_RE.findall(text))
    if parsed:
        return parsed
    try:
        return float(text)  # bare seconds, as retry-after usually sends
    except ValueError:
        return 0.0


def _reset_client() -> None:
    """Drop the connection pool after a transport failure.

    A dropped or slept-through connection leaves dead sockets pooled, so the
    next request fails the same way; rebuilding the client clears them.
    """
    global _client
    if _client is not None:
        _client.close()
        _client = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        _client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    return _client


def _raise_if_daily_limit(response: httpx.Response) -> None:
    if response.status_code != 429:
        return
    try:
        message = response.json()["error"]["message"]
    except Exception:
        return
    if "tokens per day" in message.lower() or "(tpd)" in message.lower():
        raise DailyTokenLimitExceeded(message)


def call_llm(prompt: str, model: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    key = llm_cache.cache_key(
        model=model, prompt=prompt, max_tokens=max_tokens, temperature=temperature
    )
    if _cache_enabled:
        cached = llm_cache.get(key)
        if cached is not None:
            return cached

    client = _get_client()
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        # gpt-oss models reason before answering; keep that budget small so the
        # token allowance is spent on the answer, not on hidden deliberation.
        "reasoning_effort": "low",
    }

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = client.post(API_URL, json=payload)
        except httpx.TransportError:
            # Dropped connections and read timeouts are transport-level, so
            # they never reach the status check below.
            if attempt == MAX_ATTEMPTS - 1:
                raise
            _reset_client()
            client = _get_client()
            time.sleep(min(2.0**attempt, MAX_SLEEP_SECONDS))
            continue

        # Check before retrying: a daily-budget 429 is not worth 8 attempts.
        _raise_if_daily_limit(response)

        if response.status_code in RETRY_STATUS_CODES and attempt < MAX_ATTEMPTS - 1:
            delay = _parse_duration(response.headers.get("retry-after"))
            time.sleep(min(delay or 2.0**attempt, MAX_SLEEP_SECONDS))
            continue

        response.raise_for_status()
        text = (response.json()["choices"][0]["message"]["content"] or "").strip()

        if _cache_enabled:
            llm_cache.put(key, text, model)
        return text

    raise RuntimeError(f"Groq API still failing after {MAX_ATTEMPTS} attempts.")
