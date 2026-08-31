"""Tests for the in-process run loop. No network and no dataset: the retriever
and provider are stubs, but the streaming, sampling, and stop-on-exhaustion
logic under test is the real implementation.
"""

from __future__ import annotations

import json

import pytest

from baseline import providers
from baseline.providers import DailyTokenLimitExceeded
from baseline.runner import run_baseline


def stub_retriever(query, question_id, k):
    return [{"idx": i, "title": f"T{i}", "text": f"text {i}", "score": 1.0} for i in range(k)]


BASE_CONFIG = {"k": 2, "split": "dev", "model": "stub-model", "provider": "stub"}


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    from baseline import llm_cache

    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path / "cache"))
    llm_cache.close()
    llm_cache.reset_stats()
    yield
    llm_cache.close()


def test_run_writes_one_prediction_per_question(tmp_path, monkeypatch, make_record) -> None:
    monkeypatch.setitem(providers.PROVIDERS, "stub", lambda p, m, t, temp: "Paris")
    records = [make_record("q1", (0, 1)), make_record("q2", (0, 1))]

    outcome = run_baseline(BASE_CONFIG, stub_retriever, records=records, results_dir=tmp_path)

    assert len(outcome.results) == 2
    assert outcome.exhausted is False
    lines = [json.loads(l) for l in outcome.predictions_path.read_text().splitlines() if l.strip()]
    assert [row["question_id"] for row in lines] == ["q1", "q2"]
    assert lines[0]["predicted_answer"] == "Paris"


def test_exhaustion_keeps_answers_already_paid_for(tmp_path, monkeypatch, make_record) -> None:
    calls = {"n": 0}

    def failing(prompt, model, max_tokens, temperature):
        calls["n"] += 1
        if calls["n"] > 2:
            raise DailyTokenLimitExceeded("tokens per day (TPD): Limit 200000")
        return "Paris"

    monkeypatch.setitem(providers.PROVIDERS, "stub", failing)
    records = [make_record(f"q{i}", (0, 1)) for i in range(5)]

    outcome = run_baseline(BASE_CONFIG, stub_retriever, records=records, results_dir=tmp_path)

    assert outcome.exhausted is True
    assert len(outcome.results) == 2  # the two answered before the limit hit
    written = [l for l in outcome.predictions_path.read_text().splitlines() if l.strip()]
    assert len(written) == 2


def test_predictions_are_flushed_as_the_run_proceeds(tmp_path, monkeypatch, make_record) -> None:
    seen_midway = {}

    def provider(prompt, model, max_tokens, temperature):
        # By the second question the first must already be durable on disk.
        path = tmp_path / "predictions_k2.jsonl"
        seen_midway.setdefault("lines", []).append(
            len([l for l in path.read_text().splitlines() if l.strip()]) if path.exists() else 0
        )
        return "Paris"

    monkeypatch.setitem(providers.PROVIDERS, "stub", provider)
    records = [make_record(f"q{i}", (0, 1)) for i in range(3)]

    run_baseline(BASE_CONFIG, stub_retriever, records=records, results_dir=tmp_path)

    assert seen_midway["lines"] == [0, 1, 2]


def test_scoring_matches_the_written_predictions(tmp_path, monkeypatch, make_record) -> None:
    monkeypatch.setitem(providers.PROVIDERS, "stub", lambda p, m, t, temp: "an answer")
    records = [make_record("q1", (0, 1))]

    outcome = run_baseline(BASE_CONFIG, stub_retriever, records=records, results_dir=tmp_path)

    # make_record's gold answer is "an answer", so this is an exact match.
    assert outcome.report.overall.count == 1
    assert outcome.report.overall.em == pytest.approx(1.0)
