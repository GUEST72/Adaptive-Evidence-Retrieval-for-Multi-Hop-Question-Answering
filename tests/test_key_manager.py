import pytest
from src.llm.key_manager import KeyManager, NoUsableKeyError
from src.llm.groq import GroqClient
from src.llm.client import LLMRequest, AuthenticationError, BadRequestError


def test_key_manager_rotates_without_exposing_values():
    manager = KeyManager(("A", "B"), environ={"A": "first-secret", "B": "second-secret"}, cooldown=1)
    first = manager.get()
    second = manager.get()
    assert {first, second} == {"first-secret", "second-secret"}
    manager.mark_failed(first)
    assert manager.get() == second


def test_round_robin_repeats_for_multiple_keys():
    manager = KeyManager(("A", "B", "C"), environ={"A": "a", "B": "b", "C": "c"})
    assert [manager.get() for _ in range(6)] == ["a", "b", "c", "a", "b", "c"]


def test_key_manager_missing_key_does_not_print_secret():
    manager = KeyManager(("MISSING",), environ={})
    try:
        manager.get()
    except RuntimeError as exc:
        assert "MISSING" in str(exc)
        assert "secret" not in str(exc).lower()


def test_unknown_key_state_is_ignored():
    manager = KeyManager(("A",), environ={"A": "a"})
    manager.mark_rate_limited("not-configured")
    assert manager.get() == "a"


def test_all_keys_raise_then_recover_after_cooldown():
    manager = KeyManager(("A", "B"), environ={"A": "a", "B": "b"}, cooldown=1)
    manager.mark_rate_limited("a")
    manager.mark_rate_limited("b")
    with pytest.raises(NoUsableKeyError):
        manager.get()
    assert manager.get() in {"a", "b"}


class Response:
    def __init__(self, status, payload=None):
        self.status_code, self.payload = status, payload or {}

    def json(self):
        return self.payload


class HTTP:
    def __init__(self, responses):
        self.responses, self.calls = iter(responses), []

    def post(self, url, **kwargs):
        self.calls.append(kwargs["headers"]["Authorization"])
        return next(self.responses)


def test_groq_rotates_only_rate_limits_and_is_bounded():
    km = KeyManager(("A", "B"), environ={"A": "a", "B": "b"})
    http = HTTP([Response(429), Response(200, {"choices": [{"message": {"content": "ok"}}]})])
    result = GroqClient(key_manager=km, http_client=http).complete(LLMRequest("q", "m"))
    assert result.text == "ok"
    assert len(http.calls) == 2
    assert http.calls[1] == "Bearer b"


def test_default_groq_key_names_are_extensible():
    manager = KeyManager(environ={"GROQ_API_KEY": "a", "GROQ_API_KEY_2": "b"})
    assert manager.key_count == 2


def test_groq_auth_failure_does_not_rotate():
    km = KeyManager(("A", "B"), environ={"A": "a", "B": "b"})
    http = HTTP([Response(401)])
    with pytest.raises(AuthenticationError):
        GroqClient(key_manager=km, http_client=http).complete(LLMRequest("q", "m"))
    assert km.get() == "b"


def test_groq_bad_request_preserves_provider_message_without_key():
    km = KeyManager(("A",), environ={"A": "secret"})
    http = HTTP([Response(400, {"error": {"message": "model is unavailable"}})])
    with pytest.raises(BadRequestError, match="model is unavailable"):
        GroqClient(key_manager=km, http_client=http).complete(LLMRequest("q", "m"))
