

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.data.musique_loader import MuSiQueRecord, Paragraph, load_split


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def document_text(paragraph: Paragraph) -> str:
    """Text BM25 indexes: Wikipedia title plus paragraph body.

    Titles often carry the entity name the question mentions, while the body
    may never repeat it. Indexing body-only would miss those matches.
    """
    title = paragraph.title.strip()
    body = paragraph.paragraph_text.strip()
    if not title:
        return body
    return f"{title} {body}"


@functools.lru_cache(maxsize=8)
def _load_split_index(split: str, data_dir: str | None, validate: bool) -> dict[str, MuSiQueRecord]:
    
    records = load_split(split, data_dir, validate=validate)
    return {record.id: record for record in records}


def _get_record(
    question_id: str, split: str, data_dir: str | Path | None, validate: bool
) -> MuSiQueRecord:
    """Look up one record by id, raising a clear KeyError if it's absent."""
    data_dir_key = str(data_dir) if data_dir is not None else None
    index = _load_split_index(split, data_dir_key, validate)

    try:
        return index[question_id]
    except KeyError as error:
        raise KeyError(
            f"Question ID {question_id!r} was not found in the {split!r} split."
        ) from error


def retrieve(
    query: str,
    question_id: str,
    k: int,
    split: str = "dev",
    data_dir: str | Path | None = None,
    validate: bool = False,
) -> list[dict[str, Any]]:
    """Rank one question's own 20 paragraphs with BM25 and return the top-k.

    This is the fixed interface Task 3 (the QA baseline) calls directly, so
    its signature and return shape are not expected to change even as the
    scoring/tokenization internals evolve.

    Args:
        query: Text to score paragraphs against. Week 1 usage is always the
            record's own raw question text -- callers should not pass a
            decomposed sub-question or reformulation yet.
        question_id: MuSiQue record id whose 20 paragraphs form the
            (closed) candidate pool.
        k: Number of paragraphs to return, ranked highest score first.
            Must be a positive integer; if it exceeds 20 (the fixed corpus
            size), all 20 are returned. BM25 indexes `title + paragraph_text`.
        split: Which MuSiQue split `question_id` belongs to ("train" or
            "dev"). Week 1 evaluation only uses "dev".
        data_dir: Optional override for the dataset directory, forwarded to
            Task 1's loader. Defaults to the project's standard
            `data/musique_ans/` location.
        validate: Forwarded to Task 1's `load_split`. Official MuSiQue-Ans
            currently fails the strict 20-paragraph check on a handful of
            records, so this defaults to False so Task 3 can call
            `retrieve(query, question_id, k)` on real data. Pass True if
            you want loading to abort on schema issues.

    Returns:
        A list of up to `k` dicts, ordered by descending BM25 score, each
        shaped:
            {
                "idx": int,            # paragraph index, 0-19
                "title": str,
                "text": str,           # paragraph_text
                "score": float,        # raw BM25 score
                "is_supporting": bool, # ground truth, for debugging only
            }
        `is_supporting` is included purely to make manual/notebook debugging
        easy in this closed-retrieval setting where gold labels are
        available; a genuinely open-domain retriever would not have this
        field to return.

    Raises:
        ValueError: if k is not a positive integer.
        KeyError: if question_id does not exist in the given split.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"k must be a positive integer; got {k!r}.")

    record = _get_record(question_id, split, data_dir, validate)
    paragraphs = record.paragraphs  # tuple[Paragraph, ...], fixed order idx 0..19

    tokenized_corpus = [tokenize(document_text(paragraph)) for paragraph in paragraphs]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(tokenize(query))

    top_k = min(k, len(paragraphs))
    ranked_indices = sorted(range(len(paragraphs)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "idx": paragraphs[i].idx,
            "title": paragraphs[i].title,
            "text": paragraphs[i].paragraph_text,
            "score": float(scores[i]),
            "is_supporting": paragraphs[i].is_supporting,
        }
        for i in ranked_indices
    ]
