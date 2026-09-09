"""Extrinsic evaluation without retrieval.

The evaluator only compares a supplied answer/support prediction with fields
already present on a MuSiQue record; it never invokes a retriever.
"""
from __future__ import annotations

from src.data.musique_loader import MuSiQueRecord, supporting_paragraphs

from .metrics import extrinsic_metrics
from .models import Decomposition
from src.llm.client import LLMClient, LLMRequest


def evaluate_record(
    record: MuSiQueRecord,
    predicted_answer: str,
    predicted_support_indices: tuple[int, ...] = (),
) -> dict[str, float]:
    return extrinsic_metrics(
        predicted_answer,
        record.answer,
        answer_aliases=record.answer_aliases,
        predicted_support_indices=predicted_support_indices,
        gold_support_indices=tuple(p.idx for p in supporting_paragraphs(record)),
    )


def evaluate_generated_steps(
    record: MuSiQueRecord, generated: Decomposition, client: LLMClient, *, model: str = "default",
    max_tokens: int = 128,
) -> dict[str, object]:
    """Answer generated subquestions using gold evidence, never retrieval.

    Gold previous answers are substituted into the explicit ``[ANSWER_N]``
    contract before each injected client call.
    """
    answers: list[str] = []
    scores = []
    paragraphs = {p.idx: p for p in record.paragraphs}
    gold_steps = record.question_decomposition
    if len(generated.steps) > len(gold_steps):
        raise ValueError("Generated decomposition has more steps than the gold decomposition.")
    for number, step in enumerate(generated.steps, 1):
        gold_step = gold_steps[number - 1]
        question = step.question
        for previous, answer in enumerate(answers, 1):
            question = question.replace(f"[ANSWER_{previous}]", answer)
        paragraph = paragraphs[gold_step.paragraph_support_idx]
        prompt = f"Answer the question using this paragraph.\n{paragraph.paragraph_text}\nQuestion: {question}"
        response = client.complete(LLMRequest(prompt=prompt, model=model, max_tokens=max_tokens))
        predicted = response.text.strip()
        score = extrinsic_metrics(predicted, gold_step.answer)
        score["hop"] = float(number)
        scores.append(score)
        answers.append(gold_step.answer)
    return {"scores": scores, "mean_token_f1": sum(x["answer_token_f1"] for x in scores) / len(scores) if scores else 0.0}
