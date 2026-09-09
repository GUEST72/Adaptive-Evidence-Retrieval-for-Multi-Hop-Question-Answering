"""Shared test helpers.

`tests/` is deliberately not a package (so pytest keeps its rootdir handling),
which means test modules cannot import each other. Shared builders live here.
"""

from __future__ import annotations

import pytest

from src.data.musique_loader import DecompositionStep, MuSiQueRecord, Paragraph


def build_record(
    question_id: str,
    supporting: tuple[int, ...],
    sub_questions: tuple[str, ...] | None = None,
    sub_answers: tuple[str, ...] | None = None,
) -> MuSiQueRecord:
    """A synthetic record whose hop count follows from its supporting set.

    `sub_questions` / `sub_answers` override the gold decomposition text, which
    hop-wise retrieval tests need in order to exercise MuSiQue's `#N`
    previous-answer references.
    """
    paragraphs = tuple(
        Paragraph(idx=i, title=f"T{i}", paragraph_text=f"text {i}", is_supporting=i in supporting)
        for i in range(20)
    )
    ordered = sorted(supporting)
    if sub_questions is not None and len(sub_questions) != len(ordered):
        raise ValueError("sub_questions must have one entry per supporting paragraph.")
    if sub_answers is not None and len(sub_answers) != len(ordered):
        raise ValueError("sub_answers must have one entry per supporting paragraph.")

    decomposition = tuple(
        DecompositionStep(
            id=n + 1,
            question=sub_questions[n] if sub_questions else f"q{n}",
            answer=sub_answers[n] if sub_answers else f"a{n}",
            paragraph_support_idx=idx,
        )
        for n, idx in enumerate(ordered)
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
