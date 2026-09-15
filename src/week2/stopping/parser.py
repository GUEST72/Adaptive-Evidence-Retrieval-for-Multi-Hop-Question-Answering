"""Parse structured stopping-model output into a stop/continue decision."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


class StoppingParseError(ValueError):
    """Raised when the model output is not a usable stopping decision."""


_TRUE = {"true", "yes", "stop", "sufficient", "1"}
_FALSE = {"false", "no", "continue", "insufficient", "0"}


def parse_stopping_output(text: str) -> dict[str, Any]:
    """Return ``{"stop": bool, "reason": str, "answer": str | None}``.

    Accepts a JSON object, optionally wrapped in a Markdown fence or preamble.
    Does not try to interpret free-form prose as a decision.
    """

    raw = _load_json_object(text)
    if "stop" not in raw:
        raise StoppingParseError("Stopping output must contain a boolean 'stop' field.")
    stop = _as_bool(raw["stop"])
    reason = raw.get("reason", "")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        raise StoppingParseError("'reason' must be a string when present.")
    answer = raw.get("answer")
    if answer is not None and not isinstance(answer, str):
        raise StoppingParseError("'answer' must be a string when present.")
    return {"stop": stop, "reason": reason.strip(), "answer": answer}


def _load_json_object(text: str) -> Mapping[str, Any]:
    candidate = text.strip()
    if not candidate:
        raise StoppingParseError("Stopping output is empty.")
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)
    try:
        raw = json.loads(candidate)
    except json.JSONDecodeError as exc:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                raw = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                raise StoppingParseError(f"Expected a JSON stopping object: {exc.msg}") from exc
        else:
            raise StoppingParseError(f"Expected a JSON stopping object: {exc.msg}") from exc
    if not isinstance(raw, Mapping):
        raise StoppingParseError("Stopping output must be a JSON object.")
    return raw


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in _TRUE:
        return True
    if isinstance(value, str) and value.strip().lower() in _FALSE:
        return False
    raise StoppingParseError(f"'stop' must be a boolean; got {value!r}.")
