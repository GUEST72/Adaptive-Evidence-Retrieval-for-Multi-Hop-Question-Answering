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


# retriever "stub" is deliberately not in the registry, so passing a custom
# function is allowed; naming a registered one and passing something else is not.
BASE_CONFIG = {"k": 2, "split": "dev", "model": "stub-model", "provider": "stub", "retriever": "stub"}


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
        path = tmp_path / "predictions_stub_k2.jsonl"
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


def test_run_writes_provenance_beside_the_predictions(tmp_path, monkeypatch, make_record) -> None:
    monkeypatch.setitem(providers.PROVIDERS, "stub", lambda p, m, t, temp: "Paris")
    config = {**BASE_CONFIG, "sample_size": 2, "seed": 13}

    outcome = run_baseline(config, stub_retriever, records=[make_record("q1", (0, 1))], results_dir=tmp_path)

    meta = json.loads(outcome.metadata_path.read_text())
    assert meta["model"] == "stub-model"
    assert meta["provider"] == "stub"
    assert meta["k"] == 2
    assert meta["seed"] == 13
    assert meta["questions_answered"] == 1
    assert meta["complete"] is True
    assert len(meta["prompt_sha256"]) == 16


def test_provenance_marks_an_interrupted_run_incomplete(tmp_path, monkeypatch, make_record) -> None:
    def failing(prompt, model, max_tokens, temperature):
        raise DailyTokenLimitExceeded("tokens per day (TPD): Limit 200000")

    monkeypatch.setitem(providers.PROVIDERS, "stub", failing)

    outcome = run_baseline(BASE_CONFIG, stub_retriever, records=[make_record("q1", (0, 1))], results_dir=tmp_path)

    meta = json.loads(outcome.metadata_path.read_text())
    assert meta["complete"] is False
    assert meta["questions_answered"] == 0


def test_prompt_digest_changes_when_the_template_changes(tmp_path, monkeypatch, make_record) -> None:
    monkeypatch.setitem(providers.PROVIDERS, "stub", lambda p, m, t, temp: "Paris")
    records = [make_record("q1", (0, 1))]

    default = run_baseline(BASE_CONFIG, stub_retriever, records=records, results_dir=tmp_path)
    digest_default = json.loads(default.metadata_path.read_text())["prompt_sha256"]

    variant = tmp_path / "variant.txt"
    variant.write_text("VARIANT {evidence} {question}", encoding="utf-8")
    changed = run_baseline(
        {**BASE_CONFIG, "prompt_path": str(variant)}, stub_retriever, records=records, results_dir=tmp_path
    )
    digest_changed = json.loads(changed.metadata_path.read_text())["prompt_sha256"]

    assert digest_default != digest_changed


def test_config_retriever_and_passed_function_must_agree(tmp_path, monkeypatch, make_record) -> None:
    """The failure that produced a set of 'bm25' results from the placeholder.

    The filename and provenance both come from the config, so a caller passing a
    different function than the config names yields results labelled as a
    retriever that never ran.
    """
    monkeypatch.setitem(providers.PROVIDERS, "stub", lambda p, m, t, temp: "Paris")
    config = {**BASE_CONFIG, "retriever": "bm25"}

    with pytest.raises(ValueError, match="different function was passed"):
        run_baseline(config, stub_retriever, records=[make_record("q1", (0, 1))], results_dir=tmp_path)


def test_retriever_is_resolved_from_config_when_not_passed(tmp_path, monkeypatch, make_record) -> None:
    from baseline import retrievers

    seen = {}

    def fake_bm25(query, question_id, k):
        seen["called"] = True
        return [{"idx": 0, "title": "T", "text": "body", "score": 1.0}]

    monkeypatch.setitem(retrievers.RETRIEVERS, "bm25", fake_bm25)
    monkeypatch.setitem(providers.PROVIDERS, "stub", lambda p, m, t, temp: "Paris")

    outcome = run_baseline(
        {**BASE_CONFIG, "retriever": "bm25"}, records=[make_record("q1", (0, 1))], results_dir=tmp_path
    )

    assert seen.get("called") is True
    assert outcome.predictions_path.name == "predictions_bm25_k2.jsonl"
