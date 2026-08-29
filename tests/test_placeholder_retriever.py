"""Tests for the placeholder lexical retriever.

The scoring tests exercise the real ranking function directly and need no
dataset. The end-to-end ``retrieve`` tests go through the Task 1 loader, so
they skip when the MuSiQue-Ans JSONL files are not present locally.
"""

from __future__ import annotations

from collections import Counter

import pytest

from baseline.placeholder_retriever import _overlap_score, _tokenize, retrieve
from src.data.musique_loader import load_split


@pytest.fixture(scope="module")
def dev_records():
    try:
        return load_split("dev")
    except FileNotFoundError as error:
        pytest.skip(f"MuSiQue-Ans dev split not available locally: {error}")


def test_paragraph_with_query_term_outscores_one_without() -> None:
    query_tokens = Counter(_tokenize("Who founded the Bauhaus school?"))

    with_term = _overlap_score(query_tokens, _tokenize("The Bauhaus was an art school."))
    without_term = _overlap_score(query_tokens, _tokenize("Rainfall totals were unusually low."))

    assert with_term > without_term
    assert without_term == 0.0


def test_empty_paragraph_scores_zero() -> None:
    assert _overlap_score(Counter(_tokenize("anything at all")), []) == 0.0


def test_longer_paragraph_does_not_win_on_size_alone() -> None:
    query_tokens = Counter(_tokenize("Bauhaus"))
    concise = _tokenize("Bauhaus design.")
    padded = _tokenize("Bauhaus " + "filler words here ")

    assert _overlap_score(query_tokens, concise) > _overlap_score(query_tokens, padded)


def test_returns_exactly_k_items(dev_records) -> None:
    record = dev_records[0]

    for k in (1, 3, 5, 10):
        retrieved = retrieve(record.question, record.id, k)
        assert len(retrieved) == min(k, len(record.paragraphs))


def test_k_larger_than_paragraph_count_returns_all(dev_records) -> None:
    record = dev_records[0]
    retrieved = retrieve(record.question, record.id, len(record.paragraphs) + 5)

    assert len(retrieved) == len(record.paragraphs)


def test_results_are_sorted_by_score_descending(dev_records) -> None:
    record = dev_records[0]
    scores = [item["score"] for item in retrieve(record.question, record.id, 10)]

    assert scores == sorted(scores, reverse=True)


def test_results_come_only_from_the_questions_own_paragraphs(dev_records) -> None:
    record = dev_records[0]
    own_indices = {paragraph.idx for paragraph in record.paragraphs}
    retrieved = retrieve(record.question, record.id, 5)

    assert {item["idx"] for item in retrieved} <= own_indices
    assert all(set(item) == {"idx", "title", "text", "score"} for item in retrieved)
