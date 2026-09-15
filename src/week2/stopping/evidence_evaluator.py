"""Set-based Supporting-Evidence precision, recall, and F1.

This matches the official MuSiQue paragraph-support metric used by
``evaluate_v1.0.py`` (the HotpotQA ``update_sp`` contract): predicted and gold
paragraph *indices* are compared as sets, then instance scores are
macro-averaged. It is not hop-wise evidence F1.

Reference behaviour (HotpotQA / MuSiQue):

- true positives = |predicted ∩ gold|
- precision = TP / |predicted| if the prediction is non-empty, else 0
- recall = TP / |gold| if gold is non-empty, else 0
- F1 = harmonic mean, or 0 if precision + recall is 0
- exact match = 1 iff the two sets are identical
- both sets empty → precision, recall, F1, and EM are 1.0

The spec's worked example is the ordinary non-empty case:

    gold={2, 7, 13}, predicted={2, 7, 18} → P=R=F1=2/3
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SupportScores:
    precision: float
    recall: float
    f1: float
    em: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _as_index_set(indices: Iterable[int]) -> set[int]:
    values: set[int] = set()
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"Paragraph idx must be an int; got {index!r}.")
        values.add(index)
    return values


def supporting_evidence_scores(
    predicted_indices: Iterable[int],
    gold_indices: Iterable[int],
) -> SupportScores:
    """Instance-level Supporting-Evidence scores on MuSiQue paragraph ``idx`` values."""

    predicted = _as_index_set(predicted_indices)
    gold = _as_index_set(gold_indices)

    if not predicted and not gold:
        return SupportScores(precision=1.0, recall=1.0, f1=1.0, em=1.0)

    true_positives = len(predicted & gold)
    false_positives = len(predicted - gold)
    false_negatives = len(gold - predicted)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    em = 1.0 if false_positives + false_negatives == 0 else 0.0
    return SupportScores(precision=precision, recall=recall, f1=f1, em=em)


def aggregate_support_scores(scores: Sequence[SupportScores]) -> SupportScores:
    """Macro-average instance scores, matching MuSiQue's ``SupportMetric.get_metric``."""

    if not scores:
        return SupportScores(precision=0.0, recall=0.0, f1=0.0, em=0.0)
    n = len(scores)
    return SupportScores(
        precision=sum(item.precision for item in scores) / n,
        recall=sum(item.recall for item in scores) / n,
        f1=sum(item.f1 for item in scores) / n,
        em=sum(item.em for item in scores) / n,
    )


def scores_from_prediction_row(
    predicted_indices: Iterable[int],
    gold_indices: Iterable[int],
) -> SupportScores:
    return supporting_evidence_scores(predicted_indices, gold_indices)


def report_support_metrics(scores: Mapping[str, SupportScores]) -> dict[str, dict[str, float]]:
    return {name: value.as_dict() for name, value in scores.items()}


def evaluate_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    records_by_id: Mapping[str, Any],
) -> dict[str, Any]:
    """Score prediction objects that expose ``question_id`` and ``retrieved_indices``."""

    overall_scores: list[SupportScores] = []
    by_hop_scores: dict[int, list[SupportScores]] = defaultdict(list)
    missing = 0
    for row in rows:
        record = records_by_id.get(row["question_id"])
        if record is None:
            missing += 1
            continue
        gold = [
            paragraph.idx
            for paragraph in record.paragraphs
            if paragraph.is_supporting
        ]
        scores = supporting_evidence_scores(row["retrieved_indices"], gold)
        overall_scores.append(scores)
        hop_count = int(row.get("hop_count", record.hop_count))
        by_hop_scores[hop_count].append(scores)
    return {
        "n": len(overall_scores),
        "missing_records": missing,
        "overall": aggregate_support_scores(overall_scores).as_dict(),
        "by_hop": {
            str(hop): aggregate_support_scores(group).as_dict()
            for hop, group in sorted(by_hop_scores.items())
        },
    }
