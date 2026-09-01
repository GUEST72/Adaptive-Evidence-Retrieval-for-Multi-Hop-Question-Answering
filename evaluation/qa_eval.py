"""EM/F1 scoring matching the standard MuSiQue/HotpotQA-style evaluation,
with a hop-wise breakdown (2/3/4-hop) required by the project plan."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from baseline.answer_extraction import normalize_answer
from baseline.qa_pipeline import QAResult


def exact_match(prediction: str, gold_candidates: list[str]) -> int:
    normalized_pred = normalize_answer(prediction)
    return int(any(normalized_pred == normalize_answer(g) for g in gold_candidates))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1(prediction: str, gold_candidates: list[str]) -> float:
    return max((f1_score(prediction, g) for g in gold_candidates), default=0.0)


@dataclass
class HopMetrics:
    count: int
    em: float
    f1: float


@dataclass
class EvalReport:
    overall: HopMetrics
    by_hop: dict[int, HopMetrics]


def evaluate(results: list[QAResult]) -> EvalReport:
    per_hop_scores: dict[int, list[tuple[int, float]]] = defaultdict(list)
    all_scores: list[tuple[int, float]] = []

    for result in results:
        gold_candidates = [result.gold_answer, *result.gold_aliases]
        em = exact_match(result.predicted_answer, gold_candidates)
        f1 = best_f1(result.predicted_answer, gold_candidates)
        per_hop_scores[result.hop_count].append((em, f1))
        all_scores.append((em, f1))

    def _aggregate(scores: list[tuple[int, float]]) -> HopMetrics:
        if not scores:
            return HopMetrics(count=0, em=0.0, f1=0.0)
        n = len(scores)
        return HopMetrics(
            count=n,
            em=sum(s[0] for s in scores) / n,
            f1=sum(s[1] for s in scores) / n,
        )

    return EvalReport(
        overall=_aggregate(all_scores),
        by_hop={hop: _aggregate(scores) for hop, scores in sorted(per_hop_scores.items())},
    )
