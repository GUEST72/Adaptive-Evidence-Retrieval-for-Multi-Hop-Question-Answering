"""Tests for the provider layer. No network: the HTTP client is stubbed, but
the payload shaping, response parsing, and error classification under test are
the real implementations.
"""

from __future__ import annotations

import httpx
import pytest

from baseline import llm_cache, llm_client, providers


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response

    def close(self):
        pass


@pytest.fixture
def stub_client(monkeypatch):
    def install(response):
        client = FakeClient(response)
        monkeypatch.setattr(providers, "_client", lambda name, **kwargs: client)
        return client

    return install


# ----------------------------------------------------------------- duration parsing

@pytest.mark.parametrize(
    "text,expected",
    [("667ms", 0.667), ("7.66s", 7.66), ("1m30s", 90.0), ("73", 73.0), (None, 0.0), ("", 0.0)],
)
def test_parse_duration(text, expected) -> None:
    assert providers.parse_duration(text) == pytest.approx(expected)


# --------------------------------------------------------------- limit classification

def test_groq_daily_limit_is_recognised_from_the_body() -> None:
    # Groq reports the daily budget ONLY here, never in a header.
    response = FakeResponse(
        {"error": {"message": "Rate limit reached ... on tokens per day (TPD): Limit 200000"}},
        status_code=429,
    )
    assert providers._groq_is_daily_limit(response) is True


def test_groq_per_minute_limit_is_not_treated_as_daily() -> None:
    response = FakeResponse(
        {"error": {"message": "Rate limit reached ... on tokens per minute (TPM)"}},
        status_code=429,
    )
    assert providers._groq_is_daily_limit(response) is False


def test_non_429_is_never_a_daily_limit() -> None:
    assert providers._groq_is_daily_limit(FakeResponse({}, status_code=500)) is False


def test_gemini_daily_quota_is_recognised_from_the_quota_id() -> None:
    # The message text never says "per day" — only the quotaId does.
    response = FakeResponse(
        {
            "error": {
                "message": "Quota exceeded for metric: generate_content_free_tier_requests, limit: 20",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                             "quotaValue": "20"}
                        ],
                    }
                ],
            }
        },
        status_code=429,
    )
    assert providers._gemini_is_daily_limit(response) is True


def test_gemini_per_minute_quota_is_not_treated_as_daily() -> None:
    response = FakeResponse(
        {
            "error": {
                "message": "Quota exceeded for metric: requests",
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                        "violations": [
                            {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}
                        ],
                    }
                ],
            }
        },
        status_code=429,
    )
    assert providers._gemini_is_daily_limit(response) is False


def test_retry_delay_falls_back_to_retryinfo_when_no_header() -> None:
    # Gemini sends no retry-after header; the delay lives in the body.
    response = FakeResponse(
        {
            "error": {
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "34s"}
                ]
            }
        },
        status_code=429,
    )
    assert providers._retry_delay(response) == pytest.approx(34.0)


def test_retry_delay_prefers_the_header_when_present() -> None:
    response = FakeResponse({}, status_code=429, headers={"retry-after": "12"})
    assert providers._retry_delay(response) == pytest.approx(12.0)


def test_gemini_lite_models_omit_thinking_config(stub_client, monkeypatch) -> None:
    # -lite variants reject thinkingConfig with a 400 and do not think anyway.
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = stub_client(
        FakeResponse({"candidates": [{"content": {"parts": [{"text": "x"}]}}]})
    )

    providers.gemini_complete("q", "gemini-3.5-flash-lite", 64, 0.0)
    assert "thinkingConfig" not in client.calls[0]["json"]["generationConfig"]


def test_gemini_non_lite_models_disable_thinking(stub_client, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = stub_client(
        FakeResponse({"candidates": [{"content": {"parts": [{"text": "x"}]}}]})
    )

    providers.gemini_complete("q", "gemini-3.5-flash", 64, 0.0)
    config = client.calls[0]["json"]["generationConfig"]
    assert config["thinkingConfig"]["thinkingBudget"] == 0


def test_gemini_legacy_daily_quota_message_is_recognised() -> None:
    response = FakeResponse(
        {"error": {"message": "Quota exceeded for quota metric 'Requests per day'"}},
        status_code=429,
    )
    assert providers._gemini_is_daily_limit(response) is True


# ------------------------------------------------------------------------ completions

def test_groq_sends_the_prompt_and_returns_the_message(stub_client, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    client = stub_client(FakeResponse({"choices": [{"message": {"content": " Paris "}}]}))

    assert providers.groq_complete("Where?", "some-model", 64, 0.0) == "Paris"
    assert client.calls[0]["json"]["messages"][0]["content"] == "Where?"
    assert client.calls[0]["json"]["max_tokens"] == 64


def test_groq_requires_a_key(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        providers.groq_complete("q", "m", 64, 0.0)


def test_gemini_extracts_text_and_disables_thinking_for_2_5(stub_client, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = stub_client(
        FakeResponse({"candidates": [{"content": {"parts": [{"text": "Berlin"}]}}]})
    )

    assert providers.gemini_complete("Where?", "gemini-2.5-flash", 64, 0.0) == "Berlin"
    config = client.calls[0]["json"]["generationConfig"]
    assert config["thinkingConfig"]["thinkingBudget"] == 0
    assert config["maxOutputTokens"] == 64


def test_gemini_blocked_response_yields_empty_string(stub_client, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    stub_client(FakeResponse({"candidates": []}))

    assert providers.gemini_complete("q", "gemini-2.5-flash", 64, 0.0) == ""


def test_ollama_posts_to_the_chat_endpoint(stub_client, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    client = stub_client(FakeResponse({"message": {"content": "Rome"}}))

    assert providers.ollama_complete("Where?", "qwen2.5:7b-instruct", 64, 0.0) == "Rome"
    assert client.calls[0]["url"].endswith("/api/chat")
    assert client.calls[0]["json"]["stream"] is False
    assert client.calls[0]["json"]["options"]["num_predict"] == 64


# -------------------------------------------------------------------------- dispatch

def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        llm_client.call_llm("q", model="m", provider="nope")


def test_cache_key_distinguishes_providers() -> None:
    common = {"model": "m", "prompt": "p", "max_tokens": 64, "temperature": 0.0}
    assert llm_cache.cache_key(provider="groq", **common) != llm_cache.cache_key(
        provider="ollama", **common
    )


def test_call_llm_dispatches_to_the_named_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    llm_cache.close()

    seen = {}

    def fake_complete(prompt, model, max_tokens, temperature):
        seen["provider_used"] = True
        return "dispatched"

    monkeypatch.setitem(providers.PROVIDERS, "ollama", fake_complete)

    assert llm_client.call_llm("q", model="m", provider="ollama") == "dispatched"
    assert seen["provider_used"] is True
    llm_cache.close()
