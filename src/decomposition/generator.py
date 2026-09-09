"""LLM-backed decomposition generation with injectable clients."""
from __future__ import annotations

from .models import Decomposition
from .parser import parse_decomposition
from .prompts import build_decomposition_prompt
from src.llm.client import LLMClient, LLMRequest


class DecompositionGenerator:
    def __init__(self, client: LLMClient, *, model: str, max_tokens: int = 512,
                 temperature: float = 0.0) -> None:
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, question: str, training_examples: list[dict]) -> Decomposition:
        prompt = build_decomposition_prompt(question, training_examples)
        response = self.client.complete(LLMRequest(
            prompt=prompt, model=self.model, max_tokens=self.max_tokens,
            temperature=self.temperature,
        ))
        return parse_decomposition(response.text)
