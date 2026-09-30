"""Offline tests for the Week 3 adaptive pipeline.

Decomposition, stopping, retrieval, and the final reader are stubs. No dataset
and no provider calls.
"""

from __future__ import annotations

import json

import pytest

from baseline.providers import DailyTokenLimitExceeded
from src.decomposition.models import Decomposition, DecompositionStep, ParseError
from src.week2.evidence import EvidenceStep
from src.week2.stopping.stopping_rule import StoppingDecision
from src.week3.contracts import HopResult
from src.week3.pipeline import (
    STATUS_PARSE_ERROR,
    execute_adaptive_record,
    run_adaptive_pipeline,
)


def _plan(*questions: str) -> Decomposition:
    return Decomposition(
        steps=tuple(
            DecompositionStep(id=f"step_{n}", question=question)
            for n, question in enumerate(questions, start=1)
        )
    )


def _paragraph(idx: int) -> dict:
    return {"idx": idx, "title": f"T{idx}", "text": f"text {idx}", "score": 1.0}


class ScriptedGenerator:
    def __init__(self, plans: dict[str, Decomposition] | None = None, errors: set[str] | None = None):
        self.plans = plans or {}
        self.errors = errors or set()
        self.calls: list[str] = []

    def __call__(self, record):
        self.calls.append(record.id)
        if record.id in self.errors:
            raise ParseError("malformed decomposition")
        return self.plans.get(record.id, _plan("h1", "h2"))


class RecordingExecutor:
    def __init__(self, indices: list[list[int]] | None = None):
        self.indices = indices
        self.calls: list[int] = []

    def __call__(self, record, decomposition, hop, prior_answers, k_hop, retrieve):
        self.calls.append(hop)
        retrieved = tuple(
            _paragraph(i)
            for i in (self.indices[hop - 1] if self.indices else [hop])
        )
        return HopResult(
            step=EvidenceStep(hop=hop, query=f"query-{hop}", retrieved_paragraphs=retrieved),
            intermediate_answer=f"A{hop}",
        )


class ScriptedRule:
    def __init__(self, stop_at: int | None = None):
        self.stop_at = stop_at

    def decide(self, question, evidence, *, hop=None, context=None):
        stop = self.stop_at is not None and hop >= self.stop_at
        return StoppingDecision(stop=stop, reason="scripted")


def _retriever(query, question_id, k):
    raise AssertionError("pipeline must not retrieve; the executor does")


def _reader_factory(log: list):
    def final_reader(question, evidence):
        log.append(
            {
                "question": question,
                "indices": [paragraph["idx"] for paragraph in evidence],
            }
        )
        return "an answer"

    return final_reader


def test_parse_failures_remain_in_the_attempted_denominator(
    tmp_path, make_record, monkeypatch
) -> None:
    records = [make_record("q-parse", (0, 1)), make_record("q-ok", (0, 1))]
    reader_log: list = []
    generate = ScriptedGenerator(errors={"q-parse"})

    outcome = run_adaptive_pipeline(
        records,
        k_hop=2,
        retrieve=_retriever,
        hop_executor=RecordingExecutor(),
        stopping_rule=ScriptedRule(),
        generate_decomposition=generate,
        final_reader=_reader_factory(reader_log),
        results_dir=tmp_path,
        report_path=tmp_path / "report.json",
    )

    assert [result.question_id for result in outcome.results] == ["q-parse", "q-ok"]
    assert outcome.results[0].status == STATUS_PARSE_ERROR
    assert outcome.results[0].predicted_answer == ""
    assert outcome.results[0].retrieved_indices == ()
    assert outcome.parse_failures == 1
    assert outcome.report.overall.count == 2
    # Gold is "an answer"; parse_error scores as a miss, the stub reader as a hit.
    assert outcome.report.overall.em == pytest.approx(0.5)
    rows = [json.loads(line) for line in outcome.predictions_path.read_text().splitlines()]
    assert rows[0]["status"] == STATUS_PARSE_ERROR
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["parse_failures"] == 1
    assert payload["questions_attempted"] == 2
    assert generate.calls == ["q-parse", "q-ok"]
    assert [entry["question"] for entry in reader_log] == [records[1].question]


def test_execution_does_not_read_gold_decomposition(tmp_path, make_record) -> None:
    record = make_record("q-gold", (0, 1, 2))

    class Exploding:
        def __getattr__(self, name):
            raise AssertionError("execution touched record.question_decomposition")

        def __len__(self):
            raise AssertionError("execution touched record.question_decomposition")

        def __iter__(self):
            raise AssertionError("execution touched record.question_decomposition")

    object.__setattr__(record, "question_decomposition", Exploding())
    reader_log: list = []

    result, run = execute_adaptive_record(
        record,
        k_hop=2,
        retrieve=_retriever,
        hop_executor=RecordingExecutor(),
        stopping_rule=ScriptedRule(stop_at=1),
        generate_decomposition=ScriptedGenerator(),
        final_reader=_reader_factory(reader_log),
    )

    assert result.status == "ok"
    assert run is not None
    assert result.hop_count == 0
    assert reader_log[0]["question"] == record.question


def test_stop_point_evidence_is_what_the_final_reader_sees(tmp_path, make_record) -> None:
    record = make_record("q-stop", (0, 1, 2))
    reader_log: list = []
    executor = RecordingExecutor(indices=[[2, 8], [5], [13]])

    outcome = run_adaptive_pipeline(
        [record],
        k_hop=3,
        retrieve=_retriever,
        hop_executor=executor,
        stopping_rule=ScriptedRule(stop_at=1),
        generate_decomposition=ScriptedGenerator(plans={"q-stop": _plan("h1", "h2", "h3")}),
        final_reader=_reader_factory(reader_log),
        results_dir=tmp_path,
        report_path=tmp_path / "report.json",
    )

    assert executor.calls == [1]
    assert reader_log == [{"question": record.question, "indices": [2, 8]}]
    assert outcome.results[0].retrieved_indices == (2, 8)
    assert outcome.results[0].stop_hop == 1
    assert outcome.results[0].generated_hop_count == 1
    assert outcome.results[0].planned_hops == 3


def test_answer_question_is_never_called(tmp_path, make_record, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("pipeline must not call answer_question()")

    monkeypatch.setattr("baseline.qa_pipeline.answer_question", forbidden)
    record = make_record("q-no-answer-question", (0, 1))

    run_adaptive_pipeline(
        [record],
        k_hop=2,
        retrieve=_retriever,
        hop_executor=RecordingExecutor(),
        stopping_rule=ScriptedRule(),
        generate_decomposition=ScriptedGenerator(),
        final_reader=_reader_factory([]),
        results_dir=tmp_path,
        report_path=tmp_path / "report.json",
    )


def test_exhaustion_omits_unattempted_records_from_scores(tmp_path, make_record) -> None:
    records = [make_record(f"q{i}", (0, 1)) for i in range(5)]
    seen = {"n": 0}

    def generate(record):
        seen["n"] += 1
        if seen["n"] > 2:
            raise DailyTokenLimitExceeded("tokens per day (TPD): Limit 200000")
        return _plan("h1", "h2")

    outcome = run_adaptive_pipeline(
        records,
        k_hop=2,
        retrieve=_retriever,
        hop_executor=RecordingExecutor(),
        stopping_rule=ScriptedRule(),
        generate_decomposition=generate,
        final_reader=_reader_factory([]),
        results_dir=tmp_path,
        report_path=tmp_path / "report.json",
    )

    assert outcome.exhausted is True
    assert [result.question_id for result in outcome.results] == ["q0", "q1"]
    assert outcome.report.overall.count == 2
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["complete"] is False
    assert payload["questions_attempted"] == 2
    meta = json.loads(outcome.metadata_path.read_text())
    assert meta["complete"] is False
    assert outcome.predictions_path.name == "predictions_adaptive_k2.jsonl"


def test_live_stopping_report_is_written_for_executed_traces(tmp_path, make_record) -> None:
    record = make_record("q-live", (0, 1))
    outcome = run_adaptive_pipeline(
        [record],
        k_hop=2,
        retrieve=_retriever,
        hop_executor=RecordingExecutor(indices=[[0], [1]]),
        stopping_rule=ScriptedRule(),
        generate_decomposition=ScriptedGenerator(plans={"q-live": _plan("h1", "h2")}),
        final_reader=_reader_factory([]),
        results_dir=tmp_path,
        report_path=tmp_path / "report.json",
    )

    payload = json.loads(outcome.report_path.read_text())
    assert payload["live_stopping"]["overall"]["count"] == 1
    assert "stopped_at" in payload["live_stopping"]["overall"]["counts"]
    assert "qa" in payload
    assert "supporting_evidence" in payload
