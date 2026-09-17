"""Prompt construction with runtime-derived few-shot examples."""
from __future__ import annotations

import random
import re
from collections.abc import Sequence

from src.data.musique_loader import MuSiQueRecord


_GOLD_REFERENCE = re.compile(r"#(\d+)")


def build_training_examples(
    records: Sequence[MuSiQueRecord], *, count: int = 3, seed: int = 0
) -> list[dict[str, object]]:
    """Build examples from train records; no development examples are embedded."""
    if count < 3 or count > 5:
        raise ValueError("count must be between 3 and 5.")
    candidates = [r for r in records if 2 <= r.hop_count <= 4]
    if len(candidates) < count:
        raise ValueError(f"Need at least {count} records to build training examples.")
    selected = random.Random(seed).sample(candidates, count)
    return [_example_from_record(record) for record in selected]


def _example_from_record(record: MuSiQueRecord) -> dict[str, object]:
    """Translate MuSiQue's ``#N`` syntax into the generated-plan contract."""
    steps: list[dict[str, object]] = []
    for position, step in enumerate(record.question_decomposition, start=1):
        references: list[int] = []

        def translate(match: re.Match[str]) -> str:
            reference = int(match.group(1))
            # MuSiQue also contains literal entity titles such as ``#9 Dream``.
            # A reference is identifiable from the released data by pointing to
            # an already completed hop; non-backward ``#N`` text stays literal.
            if 1 <= reference < position:
                references.append(reference)
                return f"[ANSWER_{reference}]"
            return match.group(0)

        question = _GOLD_REFERENCE.sub(translate, step.question)
        # Preserve first-reference order while declaring each dependency once.
        unique_references = list(dict.fromkeys(references))
        steps.append(
            {
                "id": f"step_{position}",
                "question": question,
                "depends_on": [f"step_{reference}" for reference in unique_references],
            }
        )
    return {"question": record.question, "steps": steps}


def build_decomposition_prompt(
    question: str, training_examples: Sequence[dict[str, object]]
) -> str:
    lines = [
        "Decompose the question into 2-4 ordered reasoning steps.",
        "Return JSON only: {\"steps\":[{\"id\":\"step_1\",\"question\":\"...\","
        "\"depends_on\":[]}]}",
        "Use explicit [ANSWER_1], [ANSWER_2] placeholders when a later step needs an earlier answer.",
        "",
    ]
    for example in training_examples:
        lines.append("Example question: " + str(example["question"]))
        lines.append("Example JSON: " + _json_compact({"steps": example["steps"]}))
    lines.extend(["", "Question: " + question, "JSON:"])
    return "\n".join(lines)


def _json_compact(value: object) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
