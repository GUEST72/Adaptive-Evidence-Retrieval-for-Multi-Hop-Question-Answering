"""Placeholder retriever: simple lexical (word-overlap) ranking over a
question's own 20 paragraphs. Stand-in for BM25 until Task 2 lands. Matches
baseline.retriever_interface.Retriever exactly — do not change the
signature when BM25 replaces this."""

from __future__ import annotations

import re
from collections import Counter

from src.data.musique_loader import get_question
from baseline.retriever_interface import RetrievedParagraph

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _overlap_score(query_tokens: Counter[str], paragraph_tokens: list[str]) -> float:
    if not paragraph_tokens:
        return 0.0
    paragraph_counts = Counter(paragraph_tokens)
    overlap = sum(min(query_tokens[token], count) for token, count in paragraph_counts.items())
    # Normalize by paragraph length so long paragraphs don't win purely on size.
    return overlap / (len(paragraph_tokens) ** 0.5)


def retrieve(query: str, question_id: str, k: int, split: str = "dev") -> list[RetrievedParagraph]:
    record = get_question(question_id, split=split)
    query_tokens = Counter(_tokenize(query))

    scored: list[RetrievedParagraph] = []
    for paragraph in record.paragraphs:
        score = _overlap_score(query_tokens, _tokenize(paragraph.paragraph_text))
        scored.append(
            RetrievedParagraph(
                idx=paragraph.idx,
                title=paragraph.title,
                text=paragraph.paragraph_text,
                score=score,
            )
        )

    scored.sort(key=lambda p: p["score"], reverse=True)
    return scored[:k]
