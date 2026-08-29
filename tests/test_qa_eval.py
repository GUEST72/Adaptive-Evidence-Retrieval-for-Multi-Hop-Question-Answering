"""Tests for EM/F1 scoring and the hop-wise breakdown.

These run entirely offline — no dataset and no LLM calls — against the real
scoring functions.
"""

from __future__ import annotations

import pytest

from baseline.qa_pipeline import QAResult
from evaluation.qa_eval import best_f1, evaluate, exact_match, f1_score


def make_result(
    question_id: str,
    hop_count: int,
    predicted: str,
    gold: str,
    aliases: tuple[str, ...] = (),
) -> QAResult:
    return QAResult(
        question_id=question_id,
        hop_count=hop_count,
        predicted_answer=predicted,
        gold_answer=gold,
        gold_aliases=aliases,
        retrieved_indices=(0, 1),
    )


def test_exact_match_on_identical_string() -> None:
    assert exact_match("Marie Curie", ["Marie Curie"]) == 1


def test_exact_match_ignores_case_articles_and_punctuation() -> None:
    assert exact_match("the Beatles.", ["Beatles"]) == 1


def test_exact_match_against_an_alias() -> None:
    assert exact_match("USA", ["United States", "USA", "U.S."]) == 1


def test_exact_match_is_zero_for_a_different_answer() -> None:
    assert exact_match("Marie Curie", ["Albert Einstein"]) == 0


def test_f1_gives_partial_credit_on_partial_overlap() -> None:
    # 2 shared tokens; precision 2/2, recall 2/3 -> F1 = 0.8
    score = f1_score("Marie Curie", "Marie Sklodowska Curie")
    assert score == pytest.approx(0.8)
    assert 0.0 < score < 1.0


def test_f1_penalizes_both_missing_and_extra_tokens() -> None:
    # 1 shared token; precision 1/2, recall 1/2 -> F1 = 0.5
    assert f1_score("Curie Institute", "Marie Curie") == pytest.approx(0.5)


def test_f1_is_zero_for_disjoint_answers() -> None:
    assert f1_score("Marie Curie", "Albert Einstein") == 0.0


def test_f1_is_one_for_a_normalized_exact_match() -> None:
    assert f1_score("The Beatles", "beatles") == pytest.approx(1.0)


def test_best_f1_picks_the_highest_scoring_alias() -> None:
    assert best_f1("USA", ["United States of America", "USA"]) == pytest.approx(1.0)


def test_evaluate_aggregates_overall_scores() -> None:
    results = [
        make_result("q1", 2, "Marie Curie", "Marie Curie"),
        make_result("q2", 2, "Albert Einstein", "Isaac Newton"),
    ]

    report = evaluate(results)

    assert report.overall.count == 2
    assert report.overall.em == pytest.approx(0.5)
    assert report.overall.f1 == pytest.approx(0.5)


def test_evaluate_groups_results_by_hop_count() -> None:
    results = [
        make_result("q1", 2, "Paris", "Paris"),
        make_result("q2", 2, "Berlin", "Rome"),
        make_result("q3", 3, "Tokyo", "Tokyo"),
        make_result("q4", 4, "Oslo", "Lisbon"),
    ]

    report = evaluate(results)

    assert set(report.by_hop) == {2, 3, 4}
    assert report.by_hop[2].count == 2
    assert report.by_hop[2].em == pytest.approx(0.5)
    assert report.by_hop[3].count == 1
    assert report.by_hop[3].em == pytest.approx(1.0)
    assert report.by_hop[4].count == 1
    assert report.by_hop[4].em == pytest.approx(0.0)
    assert report.overall.count == 4


def test_evaluate_uses_aliases_when_scoring() -> None:
    results = [make_result("q1", 2, "USA", "United States", aliases=("USA", "U.S."))]

    report = evaluate(results)

    assert report.overall.em == pytest.approx(1.0)
    assert report.overall.f1 == pytest.approx(1.0)


def test_evaluate_on_empty_results_is_zeroed() -> None:
    report = evaluate([])

    assert report.overall.count == 0
    assert report.overall.em == 0.0
    assert report.by_hop == {}
