"""Tests for the live adaptive hop loop.

Executor, retriever and stopping rule are all stubs: these assert control flow —
ordering, answer propagation, and that a stop actually prevents work — not
retrieval or reader quality. No dataset, no network, no provider.
"""

from __future__ import annotations

import pytest

from src.decomposition.models import Decomposition, DecompositionStep
from src.week2.evidence import EvidenceStep, accumulated_indices
from src.week2.stopping.stopping_rule import StoppingDecision
from src.week3.adaptive_loop import run_adaptive_hop_loop
from src.week3.contracts import HopResult


def decomposition(*questions: str) -> Decomposition:
    return Decomposition(
        steps=tuple(
            DecompositionStep(id=str(n), question=q) for n, q in enumerate(questions, start=1)
        )
    )


def paragraph(idx: int) -> dict:
    return {"idx": idx, "title": f"T{idx}", "text": f"text {idx}", "score": 1.0}


class RecordingExecutor:
    """Records every hop it was asked for and what answers it was handed."""

    def __init__(self, answers: list[str] | None = None, indices: list[list[int]] | None = None):
        self.answers = answers
        self.indices = indices
        self.calls: list[dict] = []

    def __call__(self, record, decomposition, hop, prior_answers, k_hop, retrieve):
        self.calls.append(
            {"hop": hop, "prior_answers": dict(prior_answers), "k_hop": k_hop}
        )
        n = len(self.calls) - 1
        retrieved = tuple(paragraph(i) for i in (self.indices[n] if self.indices else [hop]))
        return HopResult(
            step=EvidenceStep(hop=hop, query=f"query-{hop}", retrieved_paragraphs=retrieved),
            intermediate_answer=(self.answers[n] if self.answers else f"A{hop}"),
        )

    @property
    def hops_called(self) -> list[int]:
        return [call["hop"] for call in self.calls]


class ScriptedRule:
    """Stops at a chosen hop; records what evidence it was shown."""

    def __init__(self, stop_at: int | None = None):
        self.stop_at = stop_at
        self.seen: list[dict] = []

    def decide(self, question, evidence, *, hop=None, context=None):
        self.seen.append(
            {"question": question, "hop": hop, "context": context,
             "evidence": [p["idx"] for p in evidence]}
        )
        stop = self.stop_at is not None and hop >= self.stop_at
        return StoppingDecision(stop=stop, reason="scripted")


def retriever(query, question_id, k):  # never reached: the executor is stubbed
    raise AssertionError("the loop must not retrieve directly; the executor does")


# ------------------------------------------------------------------- stopping


def test_stop_prevents_every_later_executor_call(make_record) -> None:
    record = make_record("q1", (0, 1, 2))
    executor = RecordingExecutor()
    rule = ScriptedRule(stop_at=1)

    result = run_adaptive_hop_loop(
        record, decomposition("h1", "h2", "h3"), 3, retriever, executor, rule
    )

    # The break must happen before the next iteration, not after it.
    assert len(executor.calls) == 1
    assert executor.hops_called == [1]
    assert result.stop_hop == 1
    assert result.executed_hops == 1
    assert result.planned_hops == 3
    assert result.stopped_early is True


def test_stopping_at_the_final_hop_is_not_early(make_record) -> None:
    record = make_record("q1", (0, 1))
    executor = RecordingExecutor()

    result = run_adaptive_hop_loop(
        record, decomposition("h1", "h2"), 3, retriever, executor, ScriptedRule(stop_at=2)
    )

    assert result.stop_hop == 2
    assert result.stopped_early is False


def test_never_stopping_completes_the_whole_plan(make_record) -> None:
    record = make_record("q1", (0, 1, 2))
    executor = RecordingExecutor()

    result = run_adaptive_hop_loop(
        record, decomposition("h1", "h2", "h3"), 2, retriever, executor, ScriptedRule(stop_at=None)
    )

    assert executor.hops_called == [1, 2, 3]
    assert result.stop_hop is None
    assert result.executed_hops == 3
    assert result.stopped_early is False


# ---------------------------------------------------------------- propagation


def test_each_hop_answer_reaches_the_next_hop(make_record) -> None:
    record = make_record("q1", (0, 1, 2))
    executor = RecordingExecutor(answers=["Steve Hillage", "Miquette Giraudy", "third"])

    result = run_adaptive_hop_loop(
        record, decomposition("h1", "h2", "h3"), 2, retriever, executor, ScriptedRule()
    )

    assert executor.calls[0]["prior_answers"] == {}
    assert executor.calls[1]["prior_answers"] == {1: "Steve Hillage"}
    assert executor.calls[2]["prior_answers"] == {1: "Steve Hillage", 2: "Miquette Giraudy"}
    assert result.intermediate_answers == {
        1: "Steve Hillage", 2: "Miquette Giraudy", 3: "third"
    }


def test_hops_execute_in_order_and_the_trace_matches(make_record) -> None:
    record = make_record("q1", (0, 1, 2))

    result = run_adaptive_hop_loop(
        record, decomposition("h1", "h2", "h3"), 2, retriever, RecordingExecutor(), ScriptedRule()
    )

    assert [step.hop for step in result.trace] == [1, 2, 3]
    assert [step.query for step in result.trace] == ["query-1", "query-2", "query-3"]


def test_k_hop_is_passed_through_to_the_executor(make_record) -> None:
    record = make_record("q1", (0, 1))
    executor = RecordingExecutor()

    run_adaptive_hop_loop(
        record, decomposition("h1", "h2"), 7, retriever, executor, ScriptedRule()
    )

    assert {call["k_hop"] for call in executor.calls} == {7}


# ------------------------------------------------------------------ decisions


def test_the_rule_sees_accumulated_evidence_and_the_hop_query(make_record) -> None:
    record = make_record("q1", (0, 1, 2))
    executor = RecordingExecutor(indices=[[2, 8], [5], [13]])
    rule = ScriptedRule()

    result = run_adaptive_hop_loop(
        record, decomposition("h1", "h2", "h3"), 2, retriever, executor, rule
    )

    # Evidence grows and deduplicates as hops accumulate.
    assert [seen["evidence"] for seen in rule.seen] == [[2, 8], [2, 8, 5], [2, 8, 5, 13]]
    assert [seen["hop"] for seen in rule.seen] == [1, 2, 3]
    assert [seen["context"] for seen in rule.seen] == ["query-1", "query-2", "query-3"]
    assert all(seen["question"] == record.question for seen in rule.seen)
    assert accumulated_indices(result.trace) == [2, 8, 5, 13]


def test_one_decision_is_recorded_per_executed_hop(make_record) -> None:
    record = make_record("q1", (0, 1, 2))

    result = run_adaptive_hop_loop(
        record, decomposition("h1", "h2", "h3"), 2, retriever, RecordingExecutor(),
        ScriptedRule(stop_at=2),
    )

    assert len(result.decisions) == 2 == result.executed_hops
    assert [d.stop for d in result.decisions] == [False, True]


# ----------------------------------------------------------------- edge cases


def test_an_empty_decomposition_executes_nothing(make_record) -> None:
    record = make_record("q1", (0, 1))
    executor = RecordingExecutor()

    result = run_adaptive_hop_loop(record, decomposition(), 2, retriever, executor, ScriptedRule())

    assert executor.calls == []
    assert result.trace == []
    assert result.stop_hop is None
    assert result.planned_hops == 0


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_k_hop_is_rejected(make_record, bad) -> None:
    record = make_record("q1", (0, 1))

    with pytest.raises(ValueError, match="k_hop must be positive"):
        run_adaptive_hop_loop(
            record, decomposition("h1"), bad, retriever, RecordingExecutor(), ScriptedRule()
        )


def test_the_loop_reads_no_gold_decomposition(make_record, monkeypatch) -> None:
    """Execution must not consult gold fields; they are evaluation-only."""
    record = make_record("q1", (0, 1, 2))
    executor = RecordingExecutor()

    # Any access to the gold decomposition during the run is a failure.
    class Exploding:
        def __getattr__(self, name):
            raise AssertionError("execution touched record.question_decomposition")
        def __len__(self):
            raise AssertionError("execution touched record.question_decomposition")
        def __iter__(self):
            raise AssertionError("execution touched record.question_decomposition")

    object.__setattr__(record, "question_decomposition", Exploding())

    run_adaptive_hop_loop(
        record, decomposition("h1", "h2"), 2, retriever, executor, ScriptedRule()
    )
