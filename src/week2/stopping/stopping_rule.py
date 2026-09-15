"""Adaptive stopping: decide whether evidence so far is sufficient."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence

from baseline.retriever_interface import RetrievedParagraph
from src.llm.client import LLMClient, LLMRequest

from .parser import parse_stopping_output
from .prompts import build_stopping_prompt

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class StoppingDecision:
    stop: bool
    reason: str
    answer: str | None = None
    raw_text: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "stop": self.stop,
            "reason": self.reason,
            "answer": self.answer,
        }


class AdaptiveStoppingRule(Protocol):
    """Week 3 integration point: consume a hop prefix and return stop/continue."""

    def decide(
        self,
        question: str,
        evidence: Sequence[RetrievedParagraph],
        *,
        hop: int | None = None,
        context: str | None = None,
    ) -> StoppingDecision:
        ...


class LLMStoppingRule:
    """LLM sufficiency check with structured JSON output."""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def decide(
        self,
        question: str,
        evidence: Sequence[RetrievedParagraph],
        *,
        hop: int | None = None,
        context: str | None = None,
    ) -> StoppingDecision:
        prompt = build_stopping_prompt(question, evidence, hop=hop, context=context)
        response = self.client.complete(
            LLMRequest(
                prompt=prompt,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        )
        parsed = parse_stopping_output(response.text)
        return StoppingDecision(
            stop=parsed["stop"],
            reason=parsed["reason"],
            answer=parsed["answer"],
            raw_text=response.text,
        )


class LexicalCoverageStoppingRule:
    """Offline heuristic: stop when every question token appears in the evidence.

    This is a reproducible no-API baseline, not the primary research method.
    It never sees gold labels or hop count.
    """

    def __init__(self, *, min_coverage: float = 1.0) -> None:
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("min_coverage must be in [0, 1].")
        self.min_coverage = min_coverage

    def decide(
        self,
        question: str,
        evidence: Sequence[RetrievedParagraph],
        *,
        hop: int | None = None,
        context: str | None = None,
    ) -> StoppingDecision:
        question_tokens = _tokens(question)
        if not question_tokens:
            return StoppingDecision(stop=False, reason="Question has no content tokens.")
        evidence_tokens = set()
        for paragraph in evidence:
            evidence_tokens.update(_tokens(paragraph.get("title", "")))
            evidence_tokens.update(_tokens(paragraph.get("text", "")))
        covered = sum(1 for token in question_tokens if token in evidence_tokens)
        coverage = covered / len(question_tokens)
        stop = coverage >= self.min_coverage and bool(evidence)
        reason = (
            f"Question-token coverage {coverage:.2f} "
            f"{'meets' if stop else 'is below'} threshold {self.min_coverage:.2f}."
        )
        return StoppingDecision(stop=stop, reason=reason)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
