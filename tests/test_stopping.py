"""Tests for Week 2 Task 3: adaptive stopping and Supporting-Evidence F1.

These cover parser behaviour, early/correct/late labels, the official
set-based support metric (including the MuSiQue empty-set contract), and
synthetic gold traces. They do not call an LLM.
"""

from __future__ import annotations

import pytest

from src.llm.client import LLMRequest, LLMResponse
from src.week2.evidence import accumulated_indices
from src.week2.stopping.evidence_evaluator import (
    aggregate_support_scores,
    supporting_evidence_scores,
)
from src.week2.stopping.evaluator import CORRECT, EARLY, LATE, classify_stopping, evaluate_stopping
from src.week2.stopping.parser import StoppingParseError, parse_stopping_output
from src.week2.stopping.stopping_rule import LexicalCoverageStoppingRule, LLMStoppingRule
from src.week2.stopping.synthetic_traces import synthetic_gold_trace
from tests.conftest import build_record


def official_hotpot_support_f1(predicted: list[int], gold: list[int]) -> tuple[float, float, float]:
    """Reference implementation of HotpotQA ``update_sp`` / MuSiQue SupportMetric."""

    predicted_set = set(map(int, predicted))
    gold_set = set(map(int, gold))
    true_positives = false_positives = false_negatives = 0
    for item in predicted_set:
        if item in gold_set:
            true_positives += 1
        else:
            false_positives += 1
    for item in gold_set:
        if item not in predicted_set:
            false_negatives += 1
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    if not predicted_set and not gold_set:
        f1 = 1.0
        precision = 1.0
        recall = 1.0
    return precision, recall, f1


class ScriptedClient:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return LLMResponse(text=text, model=request.model, provider="scripted")


def test_supporting_f1_matches_spec_partial_overlap() -> None:
    scores = supporting_evidence_scores([2, 7, 18], [2, 7, 13])
    assert scores.precision == pytest.approx(2 / 3)
    assert scores.recall == pytest.approx(2 / 3)
    assert scores.f1 == pytest.approx(2 / 3)
    assert scores.em == 0.0


def test_supporting_f1_perfect_prediction() -> None:
    scores = supporting_evidence_scores([2, 7, 13], [13, 2, 7])
    assert scores.precision == 1.0
    assert scores.recall == 1.0
    assert scores.f1 == 1.0
    assert scores.em == 1.0


def test_supporting_f1_no_overlap() -> None:
    scores = supporting_evidence_scores([1, 2], [8, 9])
    assert scores.precision == 0.0
    assert scores.recall == 0.0
    assert scores.f1 == 0.0


def test_supporting_f1_extra_predictions() -> None:
    scores = supporting_evidence_scores([2, 7, 13, 18], [2, 7, 13])
    assert scores.precision == pytest.approx(0.75)
    assert scores.recall == 1.0
    assert scores.f1 == pytest.approx(2 * 0.75 * 1.0 / 1.75)


def test_supporting_f1_missing_predictions() -> None:
    scores = supporting_evidence_scores([2], [2, 7, 13])
    assert scores.precision == 1.0
    assert scores.recall == pytest.approx(1 / 3)


def test_supporting_f1_empty_prediction_against_gold() -> None:
    scores = supporting_evidence_scores([], [2, 7])
    assert scores.precision == 0.0
    assert scores.recall == 0.0
    assert scores.f1 == 0.0
    assert scores.em == 0.0


def test_supporting_f1_both_empty_is_perfect() -> None:
    scores = supporting_evidence_scores([], [])
    assert scores == supporting_evidence_scores([], [])
    assert scores.f1 == 1.0
    assert scores.em == 1.0


def test_supporting_f1_matches_official_reference_cases() -> None:
    cases = [
        ([2, 7, 18], [2, 7, 13]),
        ([2, 7, 13], [2, 7, 13]),
        ([], [1]),
        ([], []),
        ([4], []),
        ([0, 1, 2], [2]),
    ]
    for predicted, gold in cases:
        ours = supporting_evidence_scores(predicted, gold)
        precision, recall, f1 = official_hotpot_support_f1(predicted, gold)
        assert ours.precision == pytest.approx(precision)
        assert ours.recall == pytest.approx(recall)
        assert ours.f1 == pytest.approx(f1)


def test_aggregate_support_is_macro_average() -> None:
    scores = [
        supporting_evidence_scores([1], [1]),
        supporting_evidence_scores([1], [2]),
    ]
    aggregated = aggregate_support_scores(scores)
    assert aggregated.f1 == pytest.approx(0.5)
    assert aggregated.precision == pytest.approx(0.5)


def test_parse_stopping_json_object() -> None:
    parsed = parse_stopping_output(
        '{"stop": true, "reason": "All required facts are present."}'
    )
    assert parsed["stop"] is True
    assert "facts" in parsed["reason"]


def test_parse_stopping_accepts_markdown_fence() -> None:
    parsed = parse_stopping_output("```json\n{\"stop\": false, \"reason\": \"missing hop\"}\n```")
    assert parsed["stop"] is False


def test_parse_stopping_rejects_prose() -> None:
    with pytest.raises(StoppingParseError):
        parse_stopping_output("I think we should stop now.")


def test_parse_stopping_rejects_missing_stop_field() -> None:
    with pytest.raises(StoppingParseError, match="stop"):
        parse_stopping_output('{"reason": "no decision"}')


def test_classify_early_correct_late() -> None:
    assert classify_stopping(1, 3) == EARLY
    assert classify_stopping(3, 3) == CORRECT
    assert classify_stopping(None, 3) == LATE
    assert classify_stopping(4, 3) == LATE


def test_synthetic_trace_is_gold_prefixes() -> None:
    record = build_record("q", supporting=(4, 9, 1))
    trace = synthetic_gold_trace(record)
    assert [step.hop for step in trace] == [1, 2, 3]
    assert accumulated_indices(trace, through_hop=1) == [1]
    assert accumulated_indices(trace, through_hop=2) == [1, 4]
    assert set(accumulated_indices(trace)) == {1, 4, 9}


def test_llm_rule_stops_at_first_true() -> None:
    record = build_record("q-llm", supporting=(2, 8, 11))
    client = ScriptedClient(
        [
            '{"stop": false, "reason": "need hop 2"}',
            '{"stop": true, "reason": "enough"}',
            '{"stop": true, "reason": "should not be called"}',
        ]
    )
    results, overall, by_hop = evaluate_stopping(
        [record], LLMStoppingRule(client, model="stub")
    )
    assert client.calls == 2
    assert results[0].stop_hop == 2
    assert results[0].outcome == EARLY
    assert overall.early == 1.0
    assert by_hop[3].early == 1.0


def test_llm_rule_is_correct_when_it_waits_for_full_gold() -> None:
    record = build_record("q-ok", supporting=(3, 5))
    client = ScriptedClient(
        [
            '{"stop": false, "reason": "incomplete"}',
            '{"stop": true, "reason": "complete"}',
        ]
    )
    results, overall, _ = evaluate_stopping([record], LLMStoppingRule(client, model="stub"))
    assert results[0].outcome == CORRECT
    assert overall.correct == 1.0


def test_never_stopping_is_late() -> None:
    record = build_record("q-late", supporting=(3, 5))
    client = ScriptedClient(['{"stop": false, "reason": "keep going"}'])
    results, overall, _ = evaluate_stopping([record], LLMStoppingRule(client, model="stub"))
    assert results[0].stop_hop is None
    assert results[0].outcome == LATE
    assert overall.late == 1.0


def test_lexical_rule_does_not_use_gold_labels() -> None:
    record = build_record(
        "lex",
        supporting=(0, 1),
        sub_questions=("tokenalpha uniqueone?", "tokenbeta uniquetwo?"),
    )
    # Force question text that is only fully covered after both paragraphs exist.
    object.__setattr__(record, "question", "tokenalpha tokenbeta")
    object.__setattr__(record.paragraphs[0], "paragraph_text", "tokenalpha only")
    object.__setattr__(record.paragraphs[1], "paragraph_text", "tokenbeta only")

    results, _, _ = evaluate_stopping([record], LexicalCoverageStoppingRule())
    assert results[0].stop_hop == 2
    assert results[0].outcome == CORRECT


def test_baseline_prediction_file_is_scorable() -> None:
    from src.week2.stopping.evidence_evaluator import evaluate_prediction_rows

    record = build_record("2hop__demo", supporting=(2, 7, 13))
    report = evaluate_prediction_rows(
        [{"question_id": "2hop__demo", "hop_count": 3, "retrieved_indices": [2, 7, 18]}],
        {record.id: record},
    )
    assert report["n"] == 1
    assert report["overall"]["f1"] == pytest.approx(2 / 3)
