"""Intrinsic and answer/evidence-only extrinsic metrics."""
from __future__ import annotations

import re
from collections.abc import Sequence, Mapping

from .models import Decomposition


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    if not pred or not gold:
        return float(pred == gold)
    overlap = sum((__import__("collections").Counter(pred) & __import__("collections").Counter(gold)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def _rouge1(pred: str, gold: str) -> float:
    return _f1(_tokens(pred), _tokens(gold))


def _rouge_l(pred: str, gold: str) -> float:
    a, b = _tokens(pred), _tokens(gold)
    row = [0] * (len(b) + 1)
    for x in a:
        previous = 0
        for j, y in enumerate(b, 1):
            saved = row[j]
            row[j] = previous + 1 if x == y else max(row[j], row[j - 1])
            previous = saved
    lcs = row[-1]
    return 0.0 if not a and not b else (1.0 if not a or not b else 2 * lcs / (len(a) + len(b)))


def intrinsic_metrics(predicted: Decomposition, gold_questions: Sequence[str]) -> dict[str, object]:
    pairs = list(zip(predicted.steps, gold_questions))
    r1 = [_rouge1(step.question, gold) for step, gold in pairs]
    rl = [_rouge_l(step.question, gold) for step, gold in pairs]
    tf1 = [_f1(_tokens(step.question), _tokens(gold)) for step, gold in pairs]
    per_step = [
        {"rouge_1": rouge_1, "rouge_l": rouge_l, "token_f1": token_f1}
        for rouge_1, rouge_l, token_f1 in zip(r1, rl, tf1)
    ]
    return {
        "position_aware": {"rouge_1": sum(r1) / len(r1) if r1 else 0.0,
                           "rouge_l": sum(rl) / len(rl) if rl else 0.0,
                           "token_f1": sum(tf1) / len(tf1) if tf1 else 0.0,
                           "matched_steps": len(pairs), "per_step": per_step},
        "complexity": {
            "predicted_hops": predicted.hop_count,
            "gold_hops": len(gold_questions),
            "hop_delta": predicted.hop_count - len(gold_questions),
            "dependency_edges": sum(len(step.depends_on) for step in predicted.steps),
            "question_tokens": sum(len(_tokens(step.question)) for step in predicted.steps),
        },
    }


def aggregate_intrinsic(
    rows: Sequence[tuple[Decomposition, Sequence[str], int]]
) -> dict[str, object]:
    """Aggregate overall, position-level, hop-count, and 2/3/4-hop metrics."""
    def average(items: list[float]) -> float:
        return sum(items) / len(items) if items else 0.0
    all_metrics = [intrinsic_metrics(pred, gold) for pred, gold, _ in rows]
    position = {}
    position_breakdown: dict[str, dict[str, float]] = {}
    for metric in ("rouge_1", "rouge_l", "token_f1"):
        position[metric] = average([
            float(item["position_aware"][metric]) for item in all_metrics
        ])
    for hop_index in range(1, 5):
        position_breakdown[str(hop_index)] = {}
        for metric in ("rouge_1", "rouge_l", "token_f1"):
            scores = []
            for item in all_metrics:
                per_step = item["position_aware"].get("per_step", [])
                if len(per_step) >= hop_index:
                    scores.append(float(per_step[hop_index - 1][metric]))
            position_breakdown[str(hop_index)][metric] = average(scores)
    breakdown: dict[str, Mapping[str, float]] = {}
    for hop in (2, 3, 4):
        subset = [item for item, _, gold_hops in zip(all_metrics, [r[1] for r in rows], [r[2] for r in rows]) if gold_hops == hop]
        breakdown[str(hop)] = {
            "count": len(subset),
            "rouge_1": average([float(x["position_aware"]["rouge_1"]) for x in subset]),
            "rouge_l": average([float(x["position_aware"]["rouge_l"]) for x in subset]),
            "token_f1": average([float(x["position_aware"]["token_f1"]) for x in subset]),
        }
    accuracy = average([
        float(pred.hop_count == len(gold)) for pred, gold, _ in rows
    ])
    return {"overall": position, "by_position": position_breakdown,
            "hop_count_accuracy": accuracy,
            "complexity_by_gold_hops": breakdown, "count": len(rows)}


def extrinsic_metrics(
    predicted_answer: str,
    gold_answer: str,
    *,
    answer_aliases: Sequence[str] = (),
    predicted_support_indices: Sequence[int] = (),
    gold_support_indices: Sequence[int] = (),
) -> dict[str, float]:
    answers = [gold_answer, *answer_aliases]
    scores = [{"exact": float(predicted_answer.strip().lower() == a.strip().lower()),
               "token_f1": _f1(_tokens(predicted_answer), _tokens(a))} for a in answers]
    support = set(predicted_support_indices)
    gold = set(gold_support_indices)
    return {
        "answer_exact": max(score["exact"] for score in scores),
        "answer_token_f1": max(score["token_f1"] for score in scores),
        "support_precision": len(support & gold) / len(support) if support else 0.0,
        "support_recall": len(support & gold) / len(gold) if gold else 0.0,
        "support_full_coverage": float(bool(gold) and gold <= support),
    }
