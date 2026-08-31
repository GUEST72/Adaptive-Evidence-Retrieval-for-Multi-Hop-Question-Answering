"""Tests for the offline retrieval metrics.

These run against the real metric functions with hand-built records and a stub
retriever — the retriever is an input here, not the thing under test.
"""

from __future__ import annotations

import pytest

from src.data.musique_loader import DecompositionStep, MuSiQueRecord, Paragraph
from evaluation.retrieval_eval import evaluate_retrieval, score_one


def make_record(question_id: str, supporting: tuple[int, ...]) -> MuSiQueRecord:
    paragraphs = tuple(
        Paragraph(idx=i, title=f"T{i}", paragraph_text=f"text {i}", is_supporting=i in supporting)
        for i in range(20)
    )
    decomposition = tuple(
        DecompositionStep(id=n + 1, question=f"q{n}", answer=f"a{n}", paragraph_support_idx=idx)
        for n, idx in enumerate(sorted(supporting))
    )
    return MuSiQueRecord(
        id=question_id,
        question="a question?",
        answer="an answer",
        answer_aliases=(),
        paragraphs=paragraphs,
        question_decomposition=decomposition,
        raw={},
    )


def retriever_returning(order: list[int]):
    """Stub retriever that always returns `order`, truncated to k."""

    def retrieve(query: str, question_id: str, k: int):
        return [{"idx": i, "title": f"T{i}", "text": f"text {i}", "score": 1.0} for i in order[:k]]

    return retrieve


def test_score_one_perfect_retrieval() -> None:
    assert score_one([3, 7], {3, 7}) == (1.0, 1.0, 1.0)


def test_score_one_no_gold_retrieved() -> None:
    assert score_one([1, 2], {3, 7}) == (0.0, 0.0, 0.0)


def test_score_one_partial_recall_does_not_count_as_all_gold() -> None:
    recall, all_gold, _ = score_one([3, 1], {3, 7})
    assert recall == pytest.approx(0.5)
    assert all_gold == 0.0


def test_score_one_reciprocal_rank_uses_first_gold_position() -> None:
    # gold 7 sits third in the returned order
    assert score_one([1, 2, 7], {7})[2] == pytest.approx(1 / 3)


def test_score_one_empty_gold_is_zeroed() -> None:
    assert score_one([1, 2], set()) == (0.0, 0.0, 0.0)


def test_evaluate_retrieval_perfect_case() -> None:
    records = [make_record("q1", (0, 1))]
    report = evaluate_retrieval(records, retriever_returning([0, 1]), k=2)

    assert report.k == 2
    assert report.overall.count == 1
    assert report.overall.recall == pytest.approx(1.0)
    assert report.overall.all_gold == pytest.approx(1.0)


def test_k_smaller_than_gold_count_cannot_reach_all_gold() -> None:
    # A 4-hop question needs 4 paragraphs; k=3 makes that structurally impossible.
    records = [make_record("q1", (0, 1, 2, 3))]
    report = evaluate_retrieval(records, retriever_returning([0, 1, 2, 3]), k=3)

    assert report.overall.recall == pytest.approx(0.75)
    assert report.overall.all_gold == pytest.approx(0.0)


def test_evaluate_retrieval_groups_by_hop_count() -> None:
    records = [
        make_record("q1", (0, 1)),          # 2-hop
        make_record("q2", (0, 1, 2)),       # 3-hop
        make_record("q3", (0, 1, 2, 3)),    # 4-hop
    ]
    report = evaluate_retrieval(records, retriever_returning([0, 1, 2, 3]), k=4)

    assert set(report.by_hop) == {2, 3, 4}
    assert report.by_hop[2].count == 1
    assert all(metrics.all_gold == pytest.approx(1.0) for metrics in report.by_hop.values())
    assert report.overall.count == 3


def test_evaluate_retrieval_on_no_records_is_zeroed() -> None:
    report = evaluate_retrieval([], retriever_returning([0]), k=5)

    assert report.overall.count == 0
    assert report.by_hop == {}
