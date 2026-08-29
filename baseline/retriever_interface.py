"""Shared retrieval interface. Task 2's BM25 retriever must match this
exact signature and return shape so it can replace the placeholder with
no changes to baseline/qa_pipeline.py."""

from __future__ import annotations

from typing import Protocol, TypedDict


class RetrievedParagraph(TypedDict):
    idx: int
    title: str
    text: str
    score: float


class Retriever(Protocol):
    def __call__(self, query: str, question_id: str, k: int) -> list[RetrievedParagraph]:
        """Return the top-k paragraphs for `question_id`, ranked by score
        descending. Must only search within that question's own 20
        paragraphs (closed, per-question retrieval — no cross-question or
        open-domain search)."""
        ...
