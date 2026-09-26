"""Score stopping quality on a *live* trace.

Week 2's classifier (`src/week2/stopping/evaluator.py`) compares the stop hop
against the **gold hop count**. That is correct for synthetic traces, which are
built one gold paragraph per hop by construction, and wrong here for two
independent reasons:

1. A generated decomposition chooses its own number of hops. It may plan three
   where the gold has two, so "hop 3 of 3" and "hop 3 of 2" are not comparable.
2. One hop can retrieve several gold paragraphs at once. Week 2 measured this:
   for *"Who is the spouse of the Green performer?"*, hop 2's query recovered
   **both** gold paragraphs, including the one hop 1 failed to find. A trace can
   therefore be complete at hop 1 of a 3-hop question, which the gold-hop-count
   rule would misreport as stopping far too early.

So this module classifies against what the trace actually accumulated: find the
first evidence prefix that contains every gold supporting paragraph, and judge
the stop relative to *that* hop.

This module shares no code with the Week 2 evaluator by design. Gold indices are
read here — that is legitimate, this is scoring — but never during execution.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from src.data.musique_loader import MuSiQueRecord, supporting_paragraphs
from src.week2.evidence import EvidenceStep, accumulated_indices
from src.week3.adaptive_loop import AdaptiveRunResult

__all__ = [
    "NEVER_SUFFICIENT",
    "STOPPED_AT",
    "STOPPED_EARLY",
    "STOPPED_LATE",
    "OUTCOMES",
    "LiveStoppingOutcome",
    "LiveStoppingReport",
    "classify_live_trace",
    "evaluate_live_traces",
    "first_sufficient_hop",
]

# Deliberately distinct strings from Week 2's EARLY / CORRECT / LATE so the two
# classifications can never be silently mixed in one results file.
# Stopped while the evidence was still incomplete: the stop caused the failure.
STOPPED_EARLY = "stopped_early"
# Stopped exactly at the hop that completed the evidence.
STOPPED_AT = "stopped_at"
# Kept retrieving after the evidence was already complete: wasted budget.
STOPPED_LATE = "stopped_late"
# Ran the entire generated plan and still never gathered full support: this is
# a retrieval or decomposition failure, not a stopping failure.
NEVER_SUFFICIENT = "never_sufficient"

OUTCOMES = (STOPPED_EARLY, STOPPED_AT, STOPPED_LATE, NEVER_SUFFICIENT)


def first_sufficient_hop(
    trace: Sequence[EvidenceStep], gold_indices: Iterable[int]
) -> int | None:
    """Earliest hop whose accumulated evidence contains every gold paragraph.

    Returns `None` when no prefix ever does — including when `gold_indices` is
    empty, since "sufficient" is not meaningful without a gold set to satisfy.
    """
    gold = set(gold_indices)
    if not gold:
        return None

    for hop in range(1, len(trace) + 1):
        if gold.issubset(accumulated_indices(trace, through_hop=hop)):
            return hop
    return None


@dataclass(frozen=True)
class LiveStoppingOutcome:
    question_id: str
    outcome: str
    gold_hop_count: int
    planned_hops: int
    executed_hops: int
    stop_hop: int | None
    first_sufficient: int | None
    hops_wasted: int
    hops_abandoned: int


def classify_live_trace(
    record: MuSiQueRecord, result: AdaptiveRunResult
) -> LiveStoppingOutcome:
    """Judge one run's stop against the evidence the trace actually gathered."""
    gold = {paragraph.idx for paragraph in supporting_paragraphs(record)}
    first = first_sufficient_hop(result.trace, gold)

    # A run that never stopped effectively "stopped" at its last executed hop:
    # it spent that much budget either way.
    effective_stop = result.stop_hop if result.stop_hop is not None else result.executed_hops

    # The trace ends at the stop, so there is never a *later* prefix to compare
    # against: `effective_stop < first` cannot occur. An incomplete trace is
    # therefore classified by what ended it — stopping, or the plan running out.
    if first is None:
        outcome = STOPPED_EARLY if result.stop_hop is not None else NEVER_SUFFICIENT
        wasted = 0
    elif effective_stop == first:
        outcome, wasted = STOPPED_AT, 0
    else:
        outcome, wasted = STOPPED_LATE, effective_stop - first

    return LiveStoppingOutcome(
        question_id=record.id,
        outcome=outcome,
        gold_hop_count=record.hop_count,
        planned_hops=result.planned_hops,
        executed_hops=result.executed_hops,
        stop_hop=result.stop_hop,
        first_sufficient=first,
        hops_wasted=wasted,
        hops_abandoned=max(result.planned_hops - result.executed_hops, 0),
    )


@dataclass(frozen=True)
class LiveStoppingReport:
    count: int
    counts: dict[str, int]
    rates: dict[str, float]
    mean_hops_wasted: float
    mean_executed_hops: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _summarise(outcomes: Sequence[LiveStoppingOutcome]) -> LiveStoppingReport:
    n = len(outcomes)
    counts = {outcome: 0 for outcome in OUTCOMES}
    for entry in outcomes:
        counts[entry.outcome] += 1

    if n == 0:
        return LiveStoppingReport(0, counts, {o: 0.0 for o in OUTCOMES}, 0.0, 0.0)

    return LiveStoppingReport(
        count=n,
        counts=counts,
        rates={outcome: counts[outcome] / n for outcome in OUTCOMES},
        mean_hops_wasted=sum(e.hops_wasted for e in outcomes) / n,
        mean_executed_hops=sum(e.executed_hops for e in outcomes) / n,
    )


def evaluate_live_traces(
    pairs: Sequence[tuple[MuSiQueRecord, AdaptiveRunResult]]
) -> dict[str, Any]:
    """Aggregate live stopping outcomes overall and by gold hop count.

    `pairs` associates each record with the run produced for it. Hop count is
    used only to break the report down, never to classify.
    """
    outcomes = [classify_live_trace(record, result) for record, result in pairs]

    by_hop: dict[int, list[LiveStoppingOutcome]] = defaultdict(list)
    for entry in outcomes:
        by_hop[entry.gold_hop_count].append(entry)

    return {
        "overall": _summarise(outcomes).as_dict(),
        "by_hop": {
            str(hop): _summarise(group).as_dict() for hop, group in sorted(by_hop.items())
        },
        "outcomes": [asdict(entry) for entry in outcomes],
    }
