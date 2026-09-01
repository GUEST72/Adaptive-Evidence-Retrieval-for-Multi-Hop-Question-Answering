"""End-to-end retrieve-then-answer pipeline. No decomposition, no
iteration, no adaptive stopping — a single retrieve call, a single LLM
call, per question."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

from src.data.musique_loader import MuSiQueRecord
from baseline.answer_extraction import extract_final_answer
from baseline.llm_client import call_llm
from baseline.retriever_interface import Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = Path(__file__).parent / "prompts" / "qa_prompt.txt"


def resolve_prompt_path(prompt_path: str | Path | None) -> Path:
    """Resolve a configured `prompt_path`, falling back to the default.

    Configs express it relative to the repository root, so a relative path is
    resolved against that rather than the current working directory.
    """
    if prompt_path is None:
        return PROMPT_PATH

    candidate = Path(prompt_path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return PROJECT_ROOT / candidate


@functools.lru_cache(maxsize=8)
def _load_template(prompt_path: Path) -> str:
    """Read once per path: a sweep would otherwise re-read this 300 times."""
    return prompt_path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class QAResult:
    question_id: str
    hop_count: int
    predicted_answer: str
    gold_answer: str
    gold_aliases: tuple[str, ...]
    retrieved_indices: tuple[int, ...]


def _build_prompt(
    question: str,
    evidence_paragraphs: list[dict],
    prompt_path: str | Path | None = None,
) -> str:
    template = _load_template(resolve_prompt_path(prompt_path))
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
    provider: str = "groq",
    prompt_path: str | Path | None = None,
) -> QAResult:
    retrieved = retrieve(record.question, record.id, k)
    prompt = _build_prompt(record.question, retrieved, prompt_path)
    raw_output = call_llm(prompt, model=model, provider=provider)
    predicted = extract_final_answer(raw_output)

    return QAResult(
        question_id=record.id,
        hop_count=record.hop_count,
        predicted_answer=predicted,
        gold_answer=record.answer,
        gold_aliases=record.answer_aliases,
        retrieved_indices=tuple(p["idx"] for p in retrieved),
    )
