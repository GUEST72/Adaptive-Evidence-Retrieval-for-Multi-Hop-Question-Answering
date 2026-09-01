"""Retrieval quality metrics, measured without calling an LLM.

Retrieval is the measured bottleneck of the baseline: at k=5 only ~12% of
questions get all of their supporting paragraphs, which caps EM regardless of
how good the reader is. Scoring retrieval on its own runs in about a second for
the whole sample and costs no API budget, so retriever changes can be iterated
freely and the expensive QA evaluation kept for milestones.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from src.data.musique_loader import MuSiQueRecord, supporting_paragraphs
from baseline.retriever_interface import Retriever


@dataclass
class RetrievalMetrics:
    count: int
    recall: float
    all_gold: float
    mrr: float


@dataclass
class RetrievalReport:
    k: int
    overall: RetrievalMetrics
    by_hop: dict[int, RetrievalMetrics]


def _reciprocal_rank(retrieved_indices: Sequence[int], gold: set[int]) -> float:
    for rank, idx in enumerate(retrieved_indices, start=1):
        if idx in gold:
            return 1.0 / rank
    return 0.0


def score_one(retrieved_indices: Sequence[int], gold: set[int]) -> tuple[float, float, float]:
    """Recall, all-gold-retrieved, and reciprocal rank for a single question."""
    if not gold:
        return 0.0, 0.0, 0.0

    found = gold.intersection(retrieved_indices)
    return (
        len(found) / len(gold),
        float(gold.issubset(retrieved_indices)),
        _reciprocal_rank(retrieved_indices, gold),
    )


def _aggregate(scores: list[tuple[float, float, float]]) -> RetrievalMetrics:
    if not scores:
        return RetrievalMetrics(count=0, recall=0.0, all_gold=0.0, mrr=0.0)

    n = len(scores)
    return RetrievalMetrics(
        count=n,
        recall=sum(s[0] for s in scores) / n,
        all_gold=sum(s[1] for s in scores) / n,
        mrr=sum(s[2] for s in scores) / n,
    )


def evaluate_retrieval(
    records: Sequence[MuSiQueRecord],
    retrieve: Retriever,
    k: int,
) -> RetrievalReport:
    per_hop: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    all_scores: list[tuple[float, float, float]] = []

    for record in records:
        gold = {paragraph.idx for paragraph in supporting_paragraphs(record)}
        retrieved = [item["idx"] for item in retrieve(record.question, record.id, k)]

        scores = score_one(retrieved, gold)
        per_hop[record.hop_count].append(scores)
        all_scores.append(scores)

    return RetrievalReport(
        k=k,
        overall=_aggregate(all_scores),
        by_hop={hop: _aggregate(scores) for hop, scores in sorted(per_hop.items())},
    )
