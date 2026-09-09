"""Tests for the shared Week 2 evidence trace.

Task 3 builds synthetic traces against this module, so its guarantees —
ordering, prefix accumulation, deduplication — are tested here rather than
inside the hop-wise retrieval tests.
"""

from __future__ import annotations

import pytest

from src.week2.evidence import (
    EvidenceStep,
    accumulated_indices,
    accumulated_paragraphs,
    trace_from_json,
    trace_to_json,
    validate_trace,
)


def paragraph(idx: int, score: float = 1.0) -> dict:
    return {"idx": idx, "title": f"T{idx}", "text": f"text {idx}", "score": score}


def step(hop: int, indices: tuple[int, ...], query: str = "q") -> EvidenceStep:
    return EvidenceStep(
        hop=hop, query=query, retrieved_paragraphs=tuple(paragraph(i) for i in indices)
    )


def test_step_exposes_its_indices_in_rank_order() -> None:
    assert step(1, (7, 2, 13)).indices == (7, 2, 13)


def test_accumulation_unions_across_hops() -> None:
    trace = [step(1, (2, 8)), step(2, (5, 7)), step(3, (13,))]

    assert set(accumulated_indices(trace)) == {2, 5, 7, 8, 13}


def test_duplicate_paragraphs_across_hops_appear_once() -> None:
    # The same paragraph retrieved at two hops must not inflate the evidence set.
    trace = [step(1, (2, 8, 11)), step(2, (5, 7, 11)), step(3, (13, 2, 18))]

    accumulated = accumulated_indices(trace)

    assert accumulated == [2, 8, 11, 5, 7, 13, 18]
    assert len(accumulated) == len(set(accumulated))


def test_first_occurrence_wins_so_earliest_hop_and_rank_are_kept() -> None:
    trace = [step(1, (4,)), step(2, (4,))]
    trace[0].retrieved_paragraphs[0]["score"] = 9.0
    trace[1].retrieved_paragraphs[0]["score"] = 0.1

    kept = accumulated_paragraphs(trace)

    assert len(kept) == 1
    assert kept[0]["score"] == 9.0


def test_prefix_accumulation_gives_evidence_available_at_each_hop() -> None:
    # This is what a stopping rule sees when deciding whether to continue.
    trace = [step(1, (2,)), step(2, (5,)), step(3, (13,))]

    assert accumulated_indices(trace, through_hop=1) == [2]
    assert accumulated_indices(trace, through_hop=2) == [2, 5]
    assert accumulated_indices(trace, through_hop=3) == [2, 5, 13]
    assert accumulated_indices(trace) == [2, 5, 13]


def test_validate_accepts_contiguous_ascending_hops() -> None:
    validate_trace([step(1, (1,)), step(2, (2,)), step(3, (3,))])


@pytest.mark.parametrize(
    "hops",
    [(1, 3), (2, 1), (1, 1), (0, 1)],
    ids=["gap", "descending", "repeated", "zero-based"],
)
def test_validate_rejects_malformed_hop_numbering(hops: tuple[int, ...]) -> None:
    trace = [step(hop, (hop,)) for hop in hops]

    with pytest.raises(ValueError, match="hops must be"):
        validate_trace(trace)


def test_trace_round_trips_through_json() -> None:
    trace = [step(1, (2, 8), query="Green >> performer"), step(2, (5,), query="Steve Hillage >> spouse")]

    restored = trace_from_json(trace_to_json(trace))

    assert [s.hop for s in restored] == [1, 2]
    assert [s.query for s in restored] == ["Green >> performer", "Steve Hillage >> spouse"]
    assert accumulated_indices(restored) == accumulated_indices(trace)


def test_empty_trace_accumulates_to_nothing() -> None:
    assert accumulated_indices([]) == []
    validate_trace([])
