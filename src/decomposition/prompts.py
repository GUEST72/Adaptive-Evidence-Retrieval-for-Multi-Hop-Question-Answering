"""Prompt construction with runtime-derived few-shot examples."""
from __future__ import annotations

import random
from collections.abc import Sequence

from src.data.musique_loader import MuSiQueRecord


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
    return [
        {
            "question": record.question,
            "steps": [
                {
                    "id": f"step_{i + 1}",
                    "question": step.question,
                    "depends_on": [f"step_{i}"] if i else [],
                }
                for i, step in enumerate(record.question_decomposition)
            ],
        }
        for record in selected
    ]


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
