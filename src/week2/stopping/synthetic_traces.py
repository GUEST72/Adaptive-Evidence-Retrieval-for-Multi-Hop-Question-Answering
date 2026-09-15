"""Gold-prefix evidence traces so stopping can be evaluated without Task 2.

For an n-hop MuSiQue record, hop i contains the first i gold supporting
paragraphs in decomposition order. After hop n the accumulated set is exactly
the gold support set.
"""

from __future__ import annotations

from src.data.musique_loader import MuSiQueRecord, Paragraph
from src.week2.evidence import EvidenceStep, EvidenceTrace, validate_trace


def _as_retrieved(paragraph: Paragraph, *, score: float = 1.0) -> dict:
    return {
        "idx": paragraph.idx,
        "title": paragraph.title,
        "text": paragraph.paragraph_text,
        "score": score,
    }


def synthetic_gold_trace(record: MuSiQueRecord) -> EvidenceTrace:
    """One step per gold hop, each retrieving that hop's supporting paragraph."""

    by_idx = {paragraph.idx: paragraph for paragraph in record.paragraphs}
    steps: EvidenceTrace = []
    for hop, gold_step in enumerate(record.question_decomposition, start=1):
        paragraph = by_idx[gold_step.paragraph_support_idx]
        steps.append(
            EvidenceStep(
                hop=hop,
                query=gold_step.question,
                retrieved_paragraphs=(_as_retrieved(paragraph),),
            )
        )
    validate_trace(steps)
    return steps
