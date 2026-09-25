"""Offline tests for generated-hop retrieval and answering."""

from __future__ import annotations

import re

import pytest

from src.decomposition.models import Decomposition, DecompositionStep
from src.week3.hop_executor import GeneratedHopExecutor


def _decomposition(*questions: str) -> Decomposition:
    return Decomposition(
        tuple(
            DecompositionStep(
                id=f"step_{number}",
                question=question,
                depends_on=tuple(
                    f"step_{reference}"
                    for reference in dict.fromkeys(
                        re.findall(r"\[ANSWER_(\d+)\]", question)
                    )
                ),
            )
            for number, question in enumerate(questions, start=1)
        )
    )


class RecordingRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def __call__(self, query: str, question_id: str, k: int) -> list[dict]:
        self.calls.append((query, question_id, k))
        return [
            {
                "idx": 7,
                "title": "Retrieved title",
                "text": "Retrieved evidence text",
                "score": 1.0,
            }
        ]


def _stub_reader(monkeypatch, response: str = "Final answer: reader answer"):
    calls: list[dict[str, str]] = []

    def call_llm(prompt: str, *, model: str, provider: str) -> str:
        calls.append({"prompt": prompt, "model": model, "provider": provider})
        return response

    monkeypatch.setattr("src.week3.hop_executor.llm_client.call_llm", call_llm)
    return calls


def test_substitutes_one_placeholder_and_reads_only_retrieved_evidence(
    make_record, monkeypatch
) -> None:
    record = make_record("executor-one", (0,))
    retriever = RecordingRetriever()
    reader_calls = _stub_reader(monkeypatch)
    executor = GeneratedHopExecutor(model="reader-test", provider="stub")

    result = executor(
        record,
        _decomposition("Who?", "Which person did [ANSWER_1] work with?"),
        2,
        {1: "Ada Lovelace"},
        3,
        retriever,
    )

    expected_query = "Which person did Ada Lovelace work with?"
    assert retriever.calls == [(expected_query, record.id, 3)]
    assert result.step.query == expected_query
    assert result.step.hop == 2
    assert result.step.indices == (7,)
    assert result.intermediate_answer == "reader answer"
    assert "Retrieved evidence text" in reader_calls[0]["prompt"]
    assert "text 0" not in reader_calls[0]["prompt"]
    assert reader_calls[0]["model"] == "reader-test"
    assert reader_calls[0]["provider"] == "stub"


def test_substitutes_multiple_placeholders(monkeypatch, make_record) -> None:
    record = make_record("executor-multiple", (0,))
    retriever = RecordingRetriever()
    _stub_reader(monkeypatch)
    executor = GeneratedHopExecutor(model="reader-test")

    result = executor(
        record,
        _decomposition("first", "second", "When did [ANSWER_1] meet [ANSWER_2]?"),
        3,
        {1: "Ada", 2: "Grace"},
        2,
        retriever,
    )

    assert result.step.query == "When did Ada meet Grace?"
    assert retriever.calls == [("When did Ada meet Grace?", record.id, 2)]


def test_independent_hop_uses_its_generated_question_verbatim(
    monkeypatch, make_record
) -> None:
    record = make_record("executor-independent", (0,))
    retriever = RecordingRetriever()
    _stub_reader(monkeypatch)
    executor = GeneratedHopExecutor(model="reader-test")
    query = "Which team did Duane Courtney join?"

    result = executor(record, _decomposition(query), 1, {}, 4, retriever)

    assert result.step.query == query
    assert retriever.calls == [(query, record.id, 4)]


def test_sequential_answer_is_propagated_to_the_next_hop(
    monkeypatch, make_record
) -> None:
    record = make_record("executor-sequential", (0,))
    retriever = RecordingRetriever()
    reader_calls = _stub_reader(monkeypatch, "First hop answer")
    executor = GeneratedHopExecutor(model="reader-test")
    decomposition = _decomposition("Who is the person?", "Where was [ANSWER_1] born?")

    first = executor(record, decomposition, 1, {}, 5, retriever)
    second = executor(
        record,
        decomposition,
        2,
        {1: first.intermediate_answer},
        5,
        retriever,
    )

    assert first.intermediate_answer == "First hop answer"
    assert second.step.query == "Where was First hop answer born?"
    assert [call[0] for call in retriever.calls] == [
        "Who is the person?",
        "Where was First hop answer born?",
    ]
    assert len(reader_calls) == 2


def test_missing_prior_answer_fails_before_retrieval_or_reader_call(
    monkeypatch, make_record
) -> None:
    record = make_record("executor-missing", (0,))
    retriever = RecordingRetriever()
    reader_calls = _stub_reader(monkeypatch)
    executor = GeneratedHopExecutor(model="reader-test")

    with pytest.raises(ValueError, match=r"Missing prior answer.*\[ANSWER_1\]"):
        executor(
            record,
            _decomposition("Who?", "Where was [ANSWER_1] born?"),
            2,
            {},
            3,
            retriever,
        )

    assert retriever.calls == []
    assert reader_calls == []
