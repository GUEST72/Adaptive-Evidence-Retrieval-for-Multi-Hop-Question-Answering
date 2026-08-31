"""Shared test helpers.

`tests/` is deliberately not a package (so pytest keeps its rootdir handling),
which means test modules cannot import each other. Shared builders live here.
"""

from __future__ import annotations

import pytest

from src.data.musique_loader import DecompositionStep, MuSiQueRecord, Paragraph


def build_record(question_id: str, supporting: tuple[int, ...]) -> MuSiQueRecord:
    """A synthetic record whose hop count follows from its supporting set."""
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
        # Unique per record: identical questions produce identical prompts,
        # which the response cache would then serve without calling the
        # provider at all — masking whatever the test meant to exercise.
        question=f"a question about {question_id}?",
        answer="an answer",
        answer_aliases=(),
        paragraphs=paragraphs,
        question_decomposition=decomposition,
        raw={},
    )


@pytest.fixture
def make_record():
    return build_record
