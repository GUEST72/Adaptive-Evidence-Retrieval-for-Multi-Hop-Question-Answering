from __future__ import annotations

from copy import deepcopy

import pytest

from src.data.musique_loader import validate_records


def valid_record() -> dict:
    paragraphs = [
        {"idx": idx, "title": f"Title {idx}", "paragraph_text": f"Text {idx}", "is_supporting": idx < 2}
        for idx in range(20)
    ]
    return {
        "id": "2hop__1_2", "question": "A valid question?", "answer": "Answer", "answer_aliases": [],
        "paragraphs": paragraphs,
        "question_decomposition": [
            {"id": 1, "question": "Step one", "answer": "Intermediate", "paragraph_support_idx": 0},
            {"id": 2, "question": "Step two", "answer": "Answer", "paragraph_support_idx": 1},
        ],
    }


def issue_codes(records: list[dict]) -> set[str]:
    return {issue.code for issue in validate_records(records).issues}


def test_valid_record_and_empty_aliases_pass() -> None:
    report = validate_records([valid_record()])
    assert report.is_valid
    assert report.hop_counts == {2: 1}


def test_missing_question_is_rejected() -> None:
    record = valid_record()
    record["question"] = ""
    assert "missing_question" in issue_codes([record])


def test_incorrect_paragraph_count_is_rejected() -> None:
    record = valid_record()
    record["paragraphs"] = record["paragraphs"][:-1]
    assert "paragraph_count" in issue_codes([record])


def test_invalid_supporting_reference_is_rejected() -> None:
    record = valid_record()
    record["question_decomposition"][0]["paragraph_support_idx"] = 19
    assert "non_supporting_reference" in issue_codes([record])


def test_duplicate_ids_are_rejected() -> None:
    assert "duplicate_id" in issue_codes([valid_record(), deepcopy(valid_record())])
