"""Early / correct / late stopping on synthetic gold-prefix traces."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from src.data.musique_loader import MuSiQueRecord
from src.week2.evidence import accumulated_paragraphs

from .stopping_rule import AdaptiveStoppingRule, StoppingDecision
from .synthetic_traces import synthetic_gold_trace

EARLY = "early"
CORRECT = "correct"
LATE = "late"


def classify_stopping(stop_hop: int | None, gold_hops: int) -> str:
    """Label a first-stop hop against the hop at which all gold evidence exists.

    ``stop_hop`` is the first hop where the rule said stop, or ``None`` if it
    never stopped. On synthetic traces the correct hop is ``gold_hops``.
    """

    if gold_hops < 1:
        raise ValueError("gold_hops must be at least 1.")
    if stop_hop is None:
        return LATE
    if stop_hop < 1:
        raise ValueError(f"stop_hop must be >= 1; got {stop_hop}.")
    if stop_hop < gold_hops:
        return EARLY
    if stop_hop == gold_hops:
        return CORRECT
    return LATE


@dataclass(frozen=True)
class StoppingExampleResult:
    question_id: str
    hop_count: int
    stop_hop: int | None
    outcome: str
    decisions: tuple[StoppingDecision, ...]


@dataclass(frozen=True)
class OutcomeRates:
    count: int
    early: float
    correct: float
    late: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "early": self.early,
            "correct": self.correct,
            "late": self.late,
        }


def _rates(outcomes: Sequence[str]) -> OutcomeRates:
    n = len(outcomes)
    if n == 0:
        return OutcomeRates(count=0, early=0.0, correct=0.0, late=0.0)
    counts = Counter(outcomes)
    return OutcomeRates(
        count=n,
        early=counts[EARLY] / n,
        correct=counts[CORRECT] / n,
        late=counts[LATE] / n,
    )


def evaluate_stopping(
    records: Sequence[MuSiQueRecord],
    rule: AdaptiveStoppingRule,
) -> tuple[list[StoppingExampleResult], OutcomeRates, dict[int, OutcomeRates]]:
    """Walk each synthetic prefix and stop at the first ``stop=true`` decision."""

    results: list[StoppingExampleResult] = []
    for record in records:
        trace = synthetic_gold_trace(record)
        decisions: list[StoppingDecision] = []
        stop_hop: int | None = None
        for step in trace:
            evidence = accumulated_paragraphs(trace, through_hop=step.hop)
            decision = rule.decide(
                record.question,
                evidence,
                hop=step.hop,
                context=step.query,
            )
            decisions.append(decision)
            if decision.stop:
                stop_hop = step.hop
                break
        outcome = classify_stopping(stop_hop, record.hop_count)
        results.append(
            StoppingExampleResult(
                question_id=record.id,
                hop_count=record.hop_count,
                stop_hop=stop_hop,
                outcome=outcome,
                decisions=tuple(decisions),
            )
        )

    overall = _rates([item.outcome for item in results])
    by_hop_groups: dict[int, list[str]] = defaultdict(list)
    for item in results:
        by_hop_groups[item.hop_count].append(item.outcome)
    by_hop = {hop: _rates(labels) for hop, labels in sorted(by_hop_groups.items())}
    return results, overall, by_hop
