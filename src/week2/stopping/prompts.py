"""Prompt for a per-hop evidence-sufficiency check.

The model sees the original question and the evidence accumulated so far. It
does not receive the gold hop count, gold supporting indices, or gold answer.
"""

from __future__ import annotations

from typing import Sequence

from baseline.retriever_interface import RetrievedParagraph

STOPPING_SYSTEM_INSTRUCTIONS = """\
You are checking whether the evidence collected so far is sufficient to answer
a multi-hop question with high confidence.

This is a per-hop sufficiency check over the current evidence set. It is not a
classification of question complexity, and it is not a request to retrieve.

Reply with a single JSON object and no other text:
{"stop": <true or false>, "reason": "<short justification>"}

Set stop=true only if the evidence already contains every fact needed to answer
the original question. Set stop=false if any required hop is still missing.
Do not guess missing facts from world knowledge.
"""


def format_evidence(paragraphs: Sequence[RetrievedParagraph]) -> str:
    if not paragraphs:
        return "(no evidence retrieved yet)"
    blocks = []
    for paragraph in paragraphs:
        title = paragraph.get("title") or f"paragraph {paragraph['idx']}"
        blocks.append(f"[{paragraph['idx']}] {title}\n{paragraph['text']}")
    return "\n\n".join(blocks)


def build_stopping_prompt(
    question: str,
    paragraphs: Sequence[RetrievedParagraph],
    *,
    hop: int | None = None,
    context: str | None = None,
) -> str:
    hop_line = f"Hops completed so far: {hop}\n" if hop is not None else ""
    extra = f"\nReasoning context:\n{context}\n" if context else ""
    return (
        f"{STOPPING_SYSTEM_INSTRUCTIONS}\n"
        f"{hop_line}"
        f"Question:\n{question}\n"
        f"{extra}\n"
        f"Evidence so far:\n{format_evidence(paragraphs)}\n\n"
        'JSON decision:'
    )
