"""The single registry of retrievers.

One mapping, used by both entrypoints and the GPU notebook, so a run cannot
name one retriever in its config and quietly execute another.
"""

from __future__ import annotations

from typing import Callable

from baseline.placeholder_retriever import retrieve as placeholder_retrieve
from src.retrieval.bm25_retriever import retrieve as bm25_retrieve

RETRIEVERS: dict[str, Callable] = {
    "placeholder": placeholder_retrieve,
    "bm25": bm25_retrieve,
}


def get_retriever(name: str) -> Callable:
    try:
        return RETRIEVERS[name]
    except KeyError:
        raise ValueError(
            f"Unknown retriever {name!r}; expected one of {sorted(RETRIEVERS)}."
        ) from None
