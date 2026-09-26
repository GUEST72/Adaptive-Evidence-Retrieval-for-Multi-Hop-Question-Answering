"""Tests for live-trace stopping classification.

The central case is `test_a_single_hop_gathering_all_gold_is_stopped_at`, which
pins why this module exists at all: Week 2's gold-hop-count classifier gets that
case wrong, and the two must disagree.
"""

from __future__ import annotations

import pytest

from src.week2.evidence import EvidenceStep
from src.week2.stopping.evaluator import classify_stopping
from src.week3.adaptive_loop import AdaptiveRunResult
from src.week3.live_trace_eval import (
    NEVER_SUFFICIENT,
    STOPPED_AT,
    STOPPED_EARLY,
    STOPPED_LATE,
    classify_live_trace,
    evaluate_live_traces,
    first_sufficient_hop,
)


def step(hop: int, indices: tuple[int, ...]) -> EvidenceStep:
    return EvidenceStep(
        hop=hop,
        query=f"query-{hop}",
        retrieved_paragraphs=tuple(
            {"idx": i, "title": f"T{i}", "text": f"text {i}", "score": 1.0} for i in indices
        ),
    )


def run(steps: list[EvidenceStep], stop_hop: int | None, planned: int | None = None):
    return AdaptiveRunResult(
        trace=steps,
        decisions=[],
        intermediate_answers={},
        stop_hop=stop_hop,
        planned_hops=planned if planned is not None else len(steps),
    )


# ------------------------------------------------------- first_sufficient_hop


def test_first_sufficient_hop_finds_the_earliest_complete_prefix() -> None:
    trace = [step(1, (2,)), step(2, (8,)), step(3, (13,))]

    assert first_sufficient_hop(trace, {2, 8}) == 2


def test_first_sufficient_hop_is_none_when_gold_never_completes() -> None:
    trace = [step(1, (2,)), step(2, (8,))]

    assert first_sufficient_hop(trace, {2, 99}) is None


def test_first_sufficient_hop_is_none_for_an_empty_gold_set() -> None:
    assert first_sufficient_hop([step(1, (2,))], set()) is None


# ------------------------------------------------------------ classification


def test_stopping_exactly_when_evidence_completes_is_stopped_at(make_record) -> None:
    record = make_record("q1", (0, 1))
    result = run([step(1, (0,)), step(2, (1,))], stop_hop=2)

    assert classify_live_trace(record, result).outcome == STOPPED_AT


def test_stopping_on_incomplete_evidence_is_early(make_record) -> None:
    """The trace ends at the stop, so 'early' means the stop left it incomplete."""
    record = make_record("q1", (0, 1))
    # Stopped at hop 1 with paragraph 1 still missing, abandoning a planned hop.
    result = run([step(1, (0,))], stop_hop=1, planned=2)

    outcome = classify_live_trace(record, result)

    assert outcome.outcome == STOPPED_EARLY
    assert outcome.first_sufficient is None
    assert outcome.hops_abandoned == 1


def test_running_the_whole_plan_without_full_support_is_not_blamed_on_stopping(
    make_record,
) -> None:
    """Distinguishes a stopping failure from a retrieval/decomposition failure."""
    record = make_record("q1", (0, 1))
    result = run([step(1, (5,)), step(2, (7,))], stop_hop=None, planned=2)

    outcome = classify_live_trace(record, result)

    assert outcome.outcome == NEVER_SUFFICIENT
    assert outcome.hops_abandoned == 0


def test_stopping_after_evidence_completes_is_late(make_record) -> None:
    record = make_record("q1", (0, 1))
    result = run([step(1, (0, 1)), step(2, (5,)), step(3, (7,))], stop_hop=3)

    outcome = classify_live_trace(record, result)

    assert outcome.outcome == STOPPED_LATE
    assert outcome.first_sufficient == 1
    assert outcome.hops_wasted == 2


def test_a_trace_that_never_gathers_all_gold_is_never_sufficient(make_record) -> None:
    record = make_record("q1", (0, 1))
    result = run([step(1, (5,)), step(2, (7,))], stop_hop=None)

    assert classify_live_trace(record, result).outcome == NEVER_SUFFICIENT


def test_a_run_that_never_stopped_is_judged_at_its_last_hop(make_record) -> None:
    record = make_record("q1", (0, 1))
    # Complete at hop 1, but the loop ran all three: budget was still spent.
    result = run([step(1, (0, 1)), step(2, (5,)), step(3, (7,))], stop_hop=None)

    outcome = classify_live_trace(record, result)

    assert outcome.stop_hop is None
    assert outcome.outcome == STOPPED_LATE
    assert outcome.hops_wasted == 2


# --------------------------------------------- the reason this module exists


def test_a_single_hop_gathering_all_gold_is_stopped_at(make_record) -> None:
    """One hop can retrieve several gold paragraphs, so gold hop count misleads.

    Measured in Week 2: for "Who is the spouse of the Green performer?", hop 2
    recovered both gold paragraphs at once. Here hop 1 of a 3-hop question does
    the same. Stopping immediately is optimal, and the two classifiers disagree.
    """
    record = make_record("q1", (0, 1, 2))  # a 3-hop question
    result = run([step(1, (0, 1, 2))], stop_hop=1, planned=3)

    live = classify_live_trace(record, result)

    assert live.outcome == STOPPED_AT
    assert live.first_sufficient == 1
    assert live.gold_hop_count == 3

    # Week 2's synthetic classifier, which compares against gold hop count,
    # calls the same run premature. That disagreement is the point.
    assert classify_stopping(stop_hop=1, gold_hops=3) != STOPPED_AT
    assert classify_stopping(stop_hop=1, gold_hops=3) == "early"


def test_a_generated_plan_longer_than_gold_is_still_classified_live(make_record) -> None:
    record = make_record("q1", (0, 1))  # gold is 2 hops
    # The generated decomposition planned four; evidence completed at hop 2.
    result = run([step(1, (0,)), step(2, (1,)), step(3, (9,))], stop_hop=3, planned=4)

    outcome = classify_live_trace(record, result)

    assert outcome.planned_hops == 4
    assert outcome.first_sufficient == 2
    assert outcome.outcome == STOPPED_LATE


# ---------------------------------------------------------------- aggregation


def test_aggregate_reports_counts_rates_and_hop_breakdown(make_record) -> None:
    at = (make_record("a", (0, 1)), run([step(1, (0,)), step(2, (1,))], stop_hop=2))
    late = (make_record("b", (0, 1)), run([step(1, (0, 1)), step(2, (9,))], stop_hop=2))
    early = (make_record("c", (0, 1)), run([step(1, (0,))], stop_hop=1, planned=2))
    # Ran its whole plan and still never gathered full support.
    never = (make_record("d", (0, 1, 2)), run([step(1, (9,))], stop_hop=None, planned=1))

    report = evaluate_live_traces([at, late, early, never])

    assert report["overall"]["count"] == 4
    assert report["overall"]["counts"][STOPPED_AT] == 1
    assert report["overall"]["counts"][STOPPED_LATE] == 1
    assert report["overall"]["counts"][STOPPED_EARLY] == 1
    assert report["overall"]["counts"][NEVER_SUFFICIENT] == 1
    assert report["overall"]["rates"][STOPPED_AT] == pytest.approx(0.25)
    assert set(report["by_hop"]) == {"2", "3"}
    assert report["by_hop"]["2"]["count"] == 3
    assert report["by_hop"]["3"]["count"] == 1
    assert len(report["outcomes"]) == 4


def test_aggregate_of_nothing_is_zeroed() -> None:
    report = evaluate_live_traces([])

    assert report["overall"]["count"] == 0
    assert report["overall"]["rates"][STOPPED_AT] == 0.0
    assert report["by_hop"] == {}
