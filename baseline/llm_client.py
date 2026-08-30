"""Thin wrapper around the Groq chat-completions API. Model name/config comes
from configs/baseline.yaml, never hardcoded here. API key comes from .env only.

Groq exposes an OpenAI-compatible endpoint, so this is a plain POST rather than
a vendor SDK — one less dependency, and the call shape stays obvious.
"""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api.groq.com/openai/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 60.0

_client: httpx.Client | None = None


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

    response = client.post(API_URL, json=payload)
    response.raise_for_status()

    return (response.json()["choices"][0]["message"]["content"] or "").strip()
