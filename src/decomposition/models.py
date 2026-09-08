from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecompositionStep:
    id: str
    question: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decomposition:
    steps: tuple[DecompositionStep, ...]
    raw: dict = field(default_factory=dict, compare=False)

    @property
    def hop_count(self) -> int:
        return len(self.steps)


class ParseError(ValueError):
    """Raised when an LLM response is not a valid decomposition."""
