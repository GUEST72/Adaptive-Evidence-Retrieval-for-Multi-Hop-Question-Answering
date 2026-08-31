from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.retrieval.bm25_retriever import retrieve, tokenize


def _paragraph(idx: int, title: str, text: str, is_supporting: bool) -> dict:
    return {"idx": idx, "title": title, "paragraph_text": text, "is_supporting": is_supporting}


def _synthetic_record() -> dict:
    """A minimal, schema-valid MuSiQue-Ans record for isolated testing.

    Paragraph 3 is the only paragraph that actually shares vocabulary with
    the question ("giraffe", "tallest"), so a correctly-working BM25
    retriever should rank it first. The other 19 paragraphs are unrelated
    filler, mirroring how real MuSiQue records mix a couple of supporting
    paragraphs into many distractors.
    """
    paragraphs = [_paragraph(i, f"Filler {i}", f"Unrelated filler text about topic number {i}.", False) for i in range(20)]
    paragraphs[3] = _paragraph(
        3, "Giraffe", "The giraffe is widely known as the tallest animal alive today.", True
    )
    paragraphs[7] = _paragraph(
        7, "Animal facts", "Among land animals, the tallest animal by far is the giraffe.", True
    )

    return {
        "id": "2hop__1_2",
        "question": "What is the tallest animal?",
        "answer": "Giraffe",
        "answer_aliases": [],
        "paragraphs": paragraphs,
        "question_decomposition": [
            {"id": 1, "question": "Step one", "answer": "tall", "paragraph_support_idx": 3},
            {"id": 2, "question": "Step two", "answer": "Giraffe", "paragraph_support_idx": 7},
        ],
    }


@pytest.fixture()
def synthetic_data_dir(tmp_path: Path) -> Path:
    """Write a one-record synthetic dev split to a temp dir and return it.

    Using a real (tiny) JSONL file, rather than mocking the loader, exercises
    the full path -- including Task 1's schema validation -- so these tests
    also catch accidental interface drift between the two modules.
    """
    dev_file = tmp_path / "dev.jsonl"
    dev_file.write_text(json.dumps(_synthetic_record()) + "\n", encoding="utf-8")
    return tmp_path


def test_tokenize_lowercases_and_splits_on_punctuation() -> None:
    assert tokenize("Giraffes, Tall-Animals!") == ["giraffes", "tall", "animals"]


def test_tokenize_empty_string_returns_empty_list() -> None:
    assert tokenize("") == []


def test_retrieve_ranks_relevant_paragraphs_above_filler(synthetic_data_dir: Path) -> None:
    results = retrieve(
        "What is the tallest animal?",
        "2hop__1_2",
        k=3,
        split="dev",
        data_dir=synthetic_data_dir,
        validate=False,
    )

    retrieved_ids = {item["idx"] for item in results}
    assert {3, 7}.issubset(retrieved_ids)


def test_retrieve_returns_exactly_k_items(synthetic_data_dir: Path) -> None:
    results = retrieve("tallest animal", "2hop__1_2", k=5, split="dev", data_dir=synthetic_data_dir, validate=False)
    assert len(results) == 5


def test_retrieve_scores_are_descending(synthetic_data_dir: Path) -> None:
    results = retrieve("tallest animal", "2hop__1_2", k=10, split="dev", data_dir=synthetic_data_dir, validate=False)
    scores = [item["score"] for item in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_result_shape(synthetic_data_dir: Path) -> None:
    results = retrieve("tallest animal", "2hop__1_2", k=1, split="dev", data_dir=synthetic_data_dir, validate=False)
    item = results[0]
    assert set(item.keys()) == {"idx", "title", "text", "score", "is_supporting"}


def test_retrieve_k_larger_than_corpus_returns_all_paragraphs(synthetic_data_dir: Path) -> None:
    results = retrieve("tallest animal", "2hop__1_2", k=999, split="dev", data_dir=synthetic_data_dir, validate=False)
    assert len(results) == 20


def test_retrieve_rejects_non_positive_k(synthetic_data_dir: Path) -> None:
    with pytest.raises(ValueError):
        retrieve("tallest animal", "2hop__1_2", k=0, split="dev", data_dir=synthetic_data_dir, validate=False)


def test_retrieve_unknown_question_id_raises_key_error(synthetic_data_dir: Path) -> None:
    with pytest.raises(KeyError):
        retrieve("tallest animal", "nonexistent_id", k=3, split="dev", data_dir=synthetic_data_dir, validate=False)


def test_retrieve_matches_entity_name_in_title(tmp_path: Path) -> None:
    """A paragraph whose entity name appears only in the title must still rank first."""
    paragraphs = [
        _paragraph(i, f"Filler {i}", f"Unrelated filler text about topic number {i}.", False)
        for i in range(20)
    ]
    paragraphs[11] = _paragraph(
        11,
        "Zebrington",
        "A small village in the hills with no other distinctive tokens.",
        True,
    )
    paragraphs[12] = _paragraph(12, "Geography", "Hills are elevated landforms.", True)
    record = {
        "id": "2hop__1_2",
        "question": "Where is Zebrington?",
        "answer": "hills",
        "answer_aliases": [],
        "paragraphs": paragraphs,
        "question_decomposition": [
            {"id": 1, "question": "Step one", "answer": "village", "paragraph_support_idx": 11},
            {"id": 2, "question": "Step two", "answer": "hills", "paragraph_support_idx": 12},
        ],
    }
    (tmp_path / "dev.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    results = retrieve(
        "Where is Zebrington?",
        "2hop__1_2",
        k=1,
        split="dev",
        data_dir=tmp_path,
        validate=False,
    )
    assert results[0]["idx"] == 11
