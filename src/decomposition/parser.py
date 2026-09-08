"""Strict JSON parser and graph validator for model-generated decompositions."""
from __future__ import annotations

import json
import re
from collections.abc import Mapping

from .models import Decomposition, DecompositionStep, ParseError

_PLACEHOLDER = re.compile(r"\[ANSWER_(\d+)\]")


def parse_decomposition(text: str, *, min_hops: int = 2, max_hops: int = 4) -> Decomposition:
    candidate = text.strip()
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
                raise ParseError(f"Expected a JSON decomposition: {exc.msg}") from exc
        else:
            raise ParseError(f"Expected a JSON decomposition: {exc.msg}") from exc
    if isinstance(raw, list):
        raw = {"steps": raw}
    if not isinstance(raw, Mapping):
        raise ParseError("Decomposition must be a JSON object.")
    items = raw.get("steps")
    if not isinstance(items, list):
        raise ParseError("Decomposition must contain a 'steps' list.")
    steps = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ParseError("Every step must be an object.")
        step_id = item.get("id")
        question = item.get("question")
        deps = item.get("depends_on", item.get("dependencies", []))
        if not isinstance(step_id, str) or not step_id.strip():
            raise ParseError("Each step needs a non-empty string id.")
        if not isinstance(question, str) or not question.strip():
            raise ParseError(f"Step {step_id!r} needs a non-empty question.")
        if deps is None:
            deps = []
        if not isinstance(deps, list) or not all(isinstance(dep, str) for dep in deps):
            raise ParseError(f"Step {step_id!r} dependencies must be a list of strings.")
        steps.append(DecompositionStep(step_id.strip(), question.strip(), tuple(deps)))
    result = Decomposition(tuple(steps), dict(raw))
    validate_decomposition(result, min_hops=min_hops, max_hops=max_hops)
    return result


def validate_decomposition(
    decomposition: Decomposition, *, min_hops: int = 2, max_hops: int = 4
) -> None:
    count = decomposition.hop_count
    if not min_hops <= count <= max_hops:
        raise ParseError(f"Expected between {min_hops} and {max_hops} steps; found {count}.")
    ids = [step.id for step in decomposition.steps]
    if len(ids) != len(set(ids)):
        raise ParseError("Step ids must be unique.")
    known = set(ids)
    for position, step in enumerate(decomposition.steps):
        if step.id in step.depends_on:
            raise ParseError(f"Step {step.id!r} cannot depend on itself.")
        unknown = set(step.depends_on) - known
        if unknown:
            raise ParseError(f"Step {step.id!r} has unknown dependency {sorted(unknown)!r}.")
        if any(ids.index(dep) >= position for dep in step.depends_on):
            raise ParseError("Dependencies must refer to an earlier step.")
        for number in _PLACEHOLDER.findall(step.question):
            reference = f"step_{number}"
            if reference not in known:
                raise ParseError(f"Question references unknown placeholder [ANSWER_{number}].")
            if reference not in step.depends_on:
                raise ParseError(
                    f"Placeholder [ANSWER_{number}] must also appear in step {step.id!r}'s dependencies."
                )
