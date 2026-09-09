"""Tests for oracle hop-wise retrieval.

The retriever is a recording stub rather than BM25, so these assert the hop
loop, substitution and budget behaviour directly instead of measuring ranking
quality. No dataset and no network required.
"""

from __future__ import annotations

import pytest

from src.week2.evidence import accumulated_indices, validate_trace
from src.week2.hopwise.retriever import (
    build_hop_queries,
    hopwise_retrieve,
    one_shot_retrieve,
    slot_budget,
    substitution_stats,
)


class RecordingRetriever:
    """Returns a fixed result per call and records every query it was asked."""

    def __init__(self, results: list[list[int]] | None = None) -> None:
        self.results = results
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, query: str, question_id: str, k: int) -> list[dict]:
        self.calls.append((query, question_id, k))
        if self.results is not None:
            indices = self.results[len(self.calls) - 1]
        else:
            indices = list(range(k))
        return [
            {"idx": i, "title": f"T{i}", "text": f"text {i}", "score": 1.0 / (rank + 1)}
            for rank, i in enumerate(indices)
        ]

    @property
    def queries(self) -> list[str]:
        return [call[0] for call in self.calls]


# --------------------------------------------------------------- query building


def test_first_hop_query_is_the_sub_question_verbatim(make_record) -> None:
    record = make_record("q1", (0, 1), sub_questions=("Green >> performer", "#1 >> spouse"))

    queries = build_hop_queries(record)

    assert queries[0].query == "Green >> performer"
    assert queries[0].is_conditioned is False


def test_reference_is_replaced_with_the_previous_gold_answer(make_record) -> None:
    record = make_record(
        "q1",
        (0, 1),
        sub_questions=("Green >> performer", "#1 >> spouse"),
        sub_answers=("Steve Hillage", "Miquette Giraudy"),
    )

    queries = build_hop_queries(record)

    assert queries[1].query == "Steve Hillage >> spouse"
    assert queries[1].substitutions == ((1, "Steve Hillage"),)
    assert queries[1].is_conditioned is True


def test_multiple_references_in_one_step_are_all_substituted(make_record) -> None:
    record = make_record(
        "q1",
        (0, 1, 2),
        sub_questions=("who?", "where?", "#1 met #2 when?"),
        sub_answers=("Ada", "Paris", "1840"),
    )

    assert build_hop_queries(record)[2].query == "Ada met Paris when?"


def test_two_digit_references_are_not_truncated(make_record) -> None:
    # "#1" must not match the leading character of "#12".
    subs = tuple(f"step {n}" for n in range(11)) + ("about #11 and #1",)
    answers = tuple(f"A{n}" for n in range(12))
    record = make_record("q1", tuple(range(12)), sub_questions=subs, sub_answers=answers)

    assert build_hop_queries(record)[11].query == "about A10 and A0"


def test_step_without_a_reference_is_kept_verbatim(make_record) -> None:
    # MuSiQue has genuinely independent later hops; they must not be skipped.
    record = make_record(
        "q1",
        (0, 1),
        sub_questions=("who?", "Duane Courtney >> member of sports team"),
    )

    queries = build_hop_queries(record)

    assert queries[1].query == "Duane Courtney >> member of sports team"
    assert queries[1].is_conditioned is False


def test_unresolvable_reference_is_left_in_place(make_record) -> None:
    # Not reachable on the released data, but must not emit an empty query.
    record = make_record("q1", (0, 1), sub_questions=("who?", "#9 >> spouse"))

    assert build_hop_queries(record)[1].query == "#9 >> spouse"


# ------------------------------------------------------------------- hop loop


def test_hops_are_executed_in_order(make_record) -> None:
    record = make_record(
        "q1",
        (0, 1, 2),
        sub_questions=("first", "#1 second", "#2 third"),
        sub_answers=("A1", "A2", "A3"),
    )
    retriever = RecordingRetriever()

    trace = hopwise_retrieve(record, k_hop=2, retrieve=retriever)

    assert retriever.queries == ["first", "A1 second", "A2 third"]
    assert [step.hop for step in trace] == [1, 2, 3]
    validate_trace(trace)


def test_evidence_accumulates_across_hops(make_record) -> None:
    record = make_record("q1", (0, 1, 2))
    retriever = RecordingRetriever(results=[[2, 8], [5, 7], [13, 18]])

    trace = hopwise_retrieve(record, k_hop=2, retrieve=retriever)

    assert accumulated_indices(trace) == [2, 8, 5, 7, 13, 18]


def test_duplicates_across_hops_do_not_corrupt_the_evidence_set(make_record) -> None:
    record = make_record("q1", (0, 1, 2))
    retriever = RecordingRetriever(results=[[2, 8, 11], [5, 7, 11], [13, 2, 18]])

    accumulated = accumulated_indices(hopwise_retrieve(record, k_hop=3, retrieve=retriever))

    assert accumulated == [2, 8, 11, 5, 7, 13, 18]
    assert len(accumulated) == len(set(accumulated))


# --------------------------------------------------------------------- budget


@pytest.mark.parametrize("hops,k_hop,expected", [(2, 3, 6), (3, 3, 9), (4, 2, 8), (2, 1, 2)])
def test_slot_budget_is_hops_times_k(make_record, hops, k_hop, expected) -> None:
    record = make_record("q1", tuple(range(hops)))

    assert slot_budget(record, k_hop) == expected


def test_hopwise_and_one_shot_spend_the_same_slots(make_record) -> None:
    record = make_record("q1", (0, 1, 2))
    k_hop = 3
    hopwise_retriever = RecordingRetriever()
    one_shot_retriever = RecordingRetriever()

    hopwise_retrieve(record, k_hop=k_hop, retrieve=hopwise_retriever)
    one_shot_retrieve(record, budget=slot_budget(record, k_hop), retrieve=one_shot_retriever)

    hopwise_slots = sum(call[2] for call in hopwise_retriever.calls)
    one_shot_slots = sum(call[2] for call in one_shot_retriever.calls)

    assert hopwise_slots == one_shot_slots == 9


def test_one_shot_queries_the_original_question_once(make_record) -> None:
    record = make_record("q1", (0, 1, 2))
    retriever = RecordingRetriever()

    trace = one_shot_retrieve(record, budget=9, retrieve=retriever)

    assert retriever.queries == [record.question]
    assert len(trace) == 1
    assert retriever.calls[0][2] == 9


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_budgets_are_rejected(make_record, bad) -> None:
    record = make_record("q1", (0, 1))

    with pytest.raises(ValueError):
        hopwise_retrieve(record, k_hop=bad, retrieve=RecordingRetriever())
    with pytest.raises(ValueError):
        one_shot_retrieve(record, budget=bad, retrieve=RecordingRetriever())


# ---------------------------------------------------------------------- stats


def test_substitution_stats_separate_conditioned_from_independent_hops(make_record) -> None:
    conditioned = make_record("q1", (0, 1), sub_questions=("who?", "#1 >> spouse"))
    independent = make_record("q2", (0, 1), sub_questions=("who?", "Someone >> team"))

    stats = substitution_stats([conditioned, independent])

    assert stats["records"] == 2
    assert stats["steps"] == 4
    assert stats["first_steps"] == 2
    assert stats["steps_with_reference"] == 1
    assert stats["later_steps_without_reference"] == 1
    assert stats["records_with_independent_hop"] == 1
