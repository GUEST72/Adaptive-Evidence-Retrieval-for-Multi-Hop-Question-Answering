"""Thin wrapper around the Anthropic API. Model name/config comes from
configs/baseline.yaml, never hardcoded here. API key comes from .env only."""

from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def call_llm(prompt: str, model: str, max_tokens: int = 64, temperature: float = 0.0) -> str:
    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()
