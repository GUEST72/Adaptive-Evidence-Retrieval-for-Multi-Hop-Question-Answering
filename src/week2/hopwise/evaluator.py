"""Budget-matched comparison of hop-wise against one-shot retrieval.

Scoring reuses `score_one` from `evaluation/retrieval_eval.py` rather than
restating the metric definitions: it takes an index set, which is exactly what
accumulated hop-wise evidence produces, so both Week 1 and Week 2 numbers come
from one implementation (spec section 15).

Both conditions receive the same number of retrieval slots per question:
hop-wise spends `k_hop` at each of `hop_count` hops, and one-shot is given
`hop_count * k_hop` in a single call. Mean unique paragraphs is reported
alongside, because deduplication across hops means hop-wise usually ends up with
*fewer* distinct paragraphs than its slot budget allows — without that column a
reader cannot tell the comparison is conservative rather than generous.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from src.data.musique_loader import MuSiQueRecord, supporting_paragraphs
from baseline.retriever_interface import Retriever
from evaluation.retrieval_eval import score_one
from src.week2.evidence import accumulated_indices
from src.week2.hopwise.retriever import (
    hopwise_retrieve,
    one_shot_retrieve,
    slot_budget,
)

__all__ = ["ConditionMetrics", "ComparisonReport", "evaluate_comparison", "report_to_json"]

CONDITIONS = ("one_shot", "hopwise")


@dataclass
class ConditionMetrics:
    count: int = 0
    recall: float = 0.0
    all_gold: float = 0.0
    mrr: float = 0.0
    mean_unique_paragraphs: float = 0.0
    mean_slots: float = 0.0


@dataclass
class _Accumulator:
    scores: list[tuple[float, float, float]] = field(default_factory=list)
    unique: list[int] = field(default_factory=list)
    slots: list[int] = field(default_factory=list)

    def add(self, score: tuple[float, float, float], unique: int, slots: int) -> None:
        self.scores.append(score)
        self.unique.append(unique)
        self.slots.append(slots)

    def finalize(self) -> ConditionMetrics:
        n = len(self.scores)
        if n == 0:
            return ConditionMetrics()
        return ConditionMetrics(
            count=n,
            recall=sum(s[0] for s in self.scores) / n,
            all_gold=sum(s[1] for s in self.scores) / n,
            mrr=sum(s[2] for s in self.scores) / n,
            mean_unique_paragraphs=sum(self.unique) / n,
            mean_slots=sum(self.slots) / n,
        )


@dataclass
class ComparisonReport:
    k_hop: int
    overall: dict[str, ConditionMetrics]
    by_hop: dict[int, dict[str, ConditionMetrics]]


def evaluate_comparison(
    records: Sequence[MuSiQueRecord],
    retrieve: Retriever,
    k_hop: int,
) -> ComparisonReport:
    """Score both conditions over `records` at a matched per-question budget."""
    overall = {condition: _Accumulator() for condition in CONDITIONS}
    by_hop: dict[int, dict[str, _Accumulator]] = defaultdict(
        lambda: {condition: _Accumulator() for condition in CONDITIONS}
    )

    for record in records:
        gold = {paragraph.idx for paragraph in supporting_paragraphs(record)}
        budget = slot_budget(record, k_hop)

        traces = {
            "hopwise": hopwise_retrieve(record, k_hop=k_hop, retrieve=retrieve),
            "one_shot": one_shot_retrieve(record, budget=budget, retrieve=retrieve),
        }

        for condition, trace in traces.items():
            indices = accumulated_indices(trace)
            entry = (score_one(indices, gold), len(indices), budget)
            overall[condition].add(*entry)
            by_hop[record.hop_count][condition].add(*entry)

    return ComparisonReport(
        k_hop=k_hop,
        overall={c: acc.finalize() for c, acc in overall.items()},
        by_hop={
            hop: {c: acc.finalize() for c, acc in conditions.items()}
            for hop, conditions in sorted(by_hop.items())
        },
    )


def report_to_json(report: ComparisonReport) -> dict[str, Any]:
    """Machine-readable results, including the deltas the write-up quotes."""

    def block(conditions: dict[str, ConditionMetrics]) -> dict[str, Any]:
        one_shot, hopwise = conditions["one_shot"], conditions["hopwise"]
        return {
            "one_shot": asdict(one_shot),
            "hopwise": asdict(hopwise),
            "delta": {
                "recall": hopwise.recall - one_shot.recall,
                "all_gold": hopwise.all_gold - one_shot.all_gold,
                "mrr": hopwise.mrr - one_shot.mrr,
            },
        }

    return {
        "k_hop": report.k_hop,
        "overall": block(report.overall),
        "by_hop": {str(hop): block(conditions) for hop, conditions in report.by_hop.items()},
    }
