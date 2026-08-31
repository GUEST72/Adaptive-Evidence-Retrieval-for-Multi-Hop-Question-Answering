"""Thin wrapper around the Groq chat-completions API. Model name/config comes
from configs/baseline.yaml, never hardcoded here. API key comes from .env only.

Groq exposes an OpenAI-compatible endpoint, so this is a plain POST rather than
a vendor SDK — one less dependency, and the call shape stays obvious.
"""

from __future__ import annotations

import os
import re
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.groq.com/openai/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_ATTEMPTS = 8
RETRY_STATUS_CODES = {429, 500, 502, 503, 529}
MAX_SLEEP_SECONDS = 90.0
# Groq's free tier allows only a few thousand tokens per minute, so pause once
# the remaining allowance drops below roughly one more request's worth.
TOKEN_HEADROOM = 2_000

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

_client: httpx.Client | None = None


def _parse_duration(text: str | None) -> float:
    """Parse Groq's reset headers, e.g. '667ms', '7.66s', '7h40m48s'."""
    if not text:
        return 0.0
    return sum(float(value) * _UNIT_SECONDS[unit] for value, unit in _DURATION_RE.findall(text))


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


def _throttle(response: httpx.Response) -> None:
    """Wait out the per-minute token window before it is actually exhausted."""
    remaining = response.headers.get("x-ratelimit-remaining-tokens")
    if remaining is None:
        return
    try:
        if int(remaining) > TOKEN_HEADROOM:
            return
    except ValueError:
        return

    delay = _parse_duration(response.headers.get("x-ratelimit-reset-tokens"))
    if delay > 0:
        time.sleep(min(delay + 0.5, MAX_SLEEP_SECONDS))


def call_llm(prompt: str, model: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
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

    # A full sweep is hundreds of calls against a few-thousand-token-per-minute
    # budget, so rate limiting is expected rather than exceptional. Back off and
    # retry instead of losing an entire run to one 429.
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

        if response.status_code in RETRY_STATUS_CODES and attempt < MAX_ATTEMPTS - 1:
            delay = _parse_duration(response.headers.get("retry-after")) or float(
                response.headers.get("retry-after") or 0
            )
            time.sleep(min(delay or 2.0**attempt, MAX_SLEEP_SECONDS))
            continue

        response.raise_for_status()
        _throttle(response)
        return (response.json()["choices"][0]["message"]["content"] or "").strip()

    raise RuntimeError(f"Groq API still failing after {MAX_ATTEMPTS} attempts.")
