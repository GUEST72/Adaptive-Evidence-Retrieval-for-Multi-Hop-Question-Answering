"""Backends that turn a prompt into text.

Each provider exposes the same `complete(prompt, model, max_tokens, temperature)`
so `llm_client.call_llm` — and therefore `qa_pipeline` — never learns which one
is active. Adding a provider means adding one function and one PROVIDERS entry.

Why more than one: Groq's free tier caps at 200k tokens/day, which a single
300-question sweep nearly exhausts. Gemini's free tier is metered far more
generously, and Ollama runs locally with no quota at all.
"""

from __future__ import annotations

import os
import re
import time
from typing import Callable

import httpx

REQUEST_TIMEOUT_SECONDS = 120.0
MAX_ATTEMPTS = 8
MAX_SLEEP_SECONDS = 90.0
RETRY_STATUS_CODES = {408, 429, 500, 502, 503, 529}

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}

_clients: dict[str, httpx.Client] = {}


class DailyTokenLimitExceeded(RuntimeError):
    """The account's daily allowance is gone.

    Distinct from a per-minute limit: it cannot be waited out in any useful
    time, so retrying only burns wall-clock. Groq reports this *only* in the
    429 body, never in a response header.
    """


class ProviderUnavailable(RuntimeError):
    """The backend could not be reached at all (e.g. Ollama is not running)."""


def parse_duration(text: str | None) -> float:
    """Parse duration headers, e.g. '667ms', '7.66s', '7h40m48s', or bare '73'."""
    if not text:
        return 0.0
    parsed = sum(float(value) * _UNIT_SECONDS[unit] for value, unit in _DURATION_RE.findall(text))
    if parsed:
        return parsed
    try:
        return float(text)
    except ValueError:
        return 0.0


def _client(name: str, **kwargs) -> httpx.Client:
    if name not in _clients:
        _clients[name] = httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, **kwargs)
    return _clients[name]


def reset_clients() -> None:
    """Drop pooled connections; dead sockets otherwise fail the same way twice."""
    for client in _clients.values():
        client.close()
    _clients.clear()


def _require_key(variable: str) -> str:
    value = os.getenv(variable)
    if not value:
        raise RuntimeError(f"{variable} not set. Copy .env.example to .env and fill it in.")
    return value


def _send_with_retry(
    send: Callable[[], httpx.Response],
    is_daily_limit: Callable[[httpx.Response], bool],
) -> httpx.Response:
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = send()
        except httpx.TransportError:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            reset_clients()
            time.sleep(min(2.0**attempt, MAX_SLEEP_SECONDS))
            continue

        if is_daily_limit(response):
            raise DailyTokenLimitExceeded(_error_message(response))

        if response.status_code in RETRY_STATUS_CODES and attempt < MAX_ATTEMPTS - 1:
            delay = _retry_delay(response)
            time.sleep(min(delay or 2.0**attempt, MAX_SLEEP_SECONDS))
            continue

        response.raise_for_status()
        return response

    raise RuntimeError(f"Provider still failing after {MAX_ATTEMPTS} attempts.")


def _retry_delay(response: httpx.Response) -> float:
    """Seconds to wait before retrying.

    Groq sends a `retry-after` header; Gemini sends none and puts the delay in
    a RetryInfo entry inside the error body instead.
    """
    header = parse_duration(response.headers.get("retry-after"))
    if header:
        return header

    try:
        details = response.json()["error"]["details"]
    except Exception:
        return 0.0

    for detail in details:
        if detail.get("@type", "").endswith("RetryInfo"):
            return parse_duration(detail.get("retryDelay"))
    return 0.0


def _quota_ids(response: httpx.Response) -> list[str]:
    """Quota identifiers named in a Google API error body."""
    try:
        details = response.json()["error"]["details"]
    except Exception:
        return []

    ids = []
    for detail in details:
        for violation in detail.get("violations", []) or []:
            if violation.get("quotaId"):
                ids.append(violation["quotaId"])
    return ids


def _error_message(response: httpx.Response) -> str:
    try:
        return response.json()["error"]["message"]
    except Exception:
        return response.text[:400]


# --------------------------------------------------------------------------- Groq

def _groq_is_daily_limit(response: httpx.Response) -> bool:
    if response.status_code != 429:
        return False
    message = _error_message(response).lower()
    return "tokens per day" in message or "(tpd)" in message


def groq_complete(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    api_key = _require_key("GROQ_API_KEY")
    client = _client("groq", headers={"Authorization": f"Bearer {api_key}"})
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
        # gpt-oss models reason before answering; keep that budget small so the
        # token allowance is spent on the answer, not on hidden deliberation.
        "reasoning_effort": "low",
    }

    response = _send_with_retry(
        lambda: client.post(GROQ_URL, json=payload), _groq_is_daily_limit
    )
    return (response.json()["choices"][0]["message"]["content"] or "").strip()


# ------------------------------------------------------------------------- Gemini

def _gemini_is_daily_limit(response: httpx.Response) -> bool:
    if response.status_code != 429:
        return False

    # The human-readable message says only "Quota exceeded for metric ...";
    # the per-day nature is visible in the quotaId, e.g.
    # GenerateRequestsPerDayPerProjectPerModel-FreeTier.
    if any("perday" in quota_id.lower() for quota_id in _quota_ids(response)):
        return True

    message = _error_message(response).lower()
    return "per day" in message or "perday" in message


def gemini_complete(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    api_key = _require_key("GEMINI_API_KEY")
    client = _client("gemini")

    generation_config: dict = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }
    if "lite" not in model:
        # Gemini 2.5 and 3.x think before answering — measured at 75 hidden
        # tokens just to reply "OK" — which wastes the daily budget and can
        # consume the whole output allowance. The -lite variants do not think
        # and reject this field outright with a 400, so skip it for them.
        generation_config["thinkingConfig"] = {"thinkingBudget": 0}

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }

    response = _send_with_retry(
        lambda: client.post(
            GEMINI_URL.format(model=model),
            json=payload,
            headers={"x-goog-api-key": api_key},
        ),
        _gemini_is_daily_limit,
    )

    candidates = response.json().get("candidates") or []
    if not candidates:
        return ""  # blocked or empty; scored as a wrong answer rather than a crash

    parts = candidates[0].get("content", {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts).strip()


# ------------------------------------------------------------------------- Ollama

def ollama_complete(prompt: str, model: str, max_tokens: int, temperature: float) -> str:
    host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")
    client = _client("ollama")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    try:
        response = _send_with_retry(
            lambda: client.post(f"{host}/api/chat", json=payload), lambda _: False
        )
    except httpx.TransportError as error:
        raise ProviderUnavailable(
            f"Could not reach Ollama at {host}. Start it with `ollama serve` "
            f"and pull the model with `ollama pull {model}`."
        ) from error

    return (response.json().get("message", {}).get("content") or "").strip()


PROVIDERS: dict[str, Callable[[str, str, int, float], str]] = {
    "groq": groq_complete,
    "gemini": gemini_complete,
    "ollama": ollama_complete,
}
