"""End-to-end retrieve-then-answer pipeline. No decomposition, no
iteration, no adaptive stopping — a single retrieve call, a single LLM
call, per question."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.data.musique_loader import MuSiQueRecord
from baseline.answer_extraction import extract_final_answer
from baseline.llm_client import call_llm
from baseline.retriever_interface import Retriever

PROMPT_PATH = Path(__file__).parent / "prompts" / "qa_prompt.txt"


@dataclass(frozen=True)
class QAResult:
    question_id: str
    hop_count: int
    predicted_answer: str
    gold_answer: str
    gold_aliases: tuple[str, ...]
    retrieved_indices: tuple[int, ...]


def _build_prompt(question: str, evidence_paragraphs: list[dict]) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    evidence_text = "\n\n".join(
        f"[{p['title']}]\n{p['text']}" for p in evidence_paragraphs
    )
    return template.format(evidence=evidence_text, question=question)


def answer_question(
    record: MuSiQueRecord,
    retrieve: Retriever,
    k: int,
    model: str,
    split: str = "dev",
) -> QAResult:
    retrieved = retrieve(record.question, record.id, k)
    prompt = _build_prompt(record.question, retrieved)
    raw_output = call_llm(prompt, model=model)
    predicted = extract_final_answer(raw_output)

    return QAResult(
        question_id=record.id,
        hop_count=record.hop_count,
        predicted_answer=predicted,
        gold_answer=record.answer,
        gold_aliases=record.answer_aliases,
        retrieved_indices=tuple(p["idx"] for p in retrieved),
    )
