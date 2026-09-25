"""Shared types for generated Week 3 hop execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from baseline.retriever_interface import Retriever
from src.data.musique_loader import MuSiQueRecord
from src.decomposition.models import Decomposition
from src.week2.evidence import EvidenceStep


@dataclass(frozen=True)
class HopResult:
    """Evidence retrieved and answer generated for one decomposition hop."""

    step: EvidenceStep
    intermediate_answer: str


class HopExecutor(Protocol):
    """Callable interface implemented by generated-hop readers."""

    def __call__(
        self,
        record: MuSiQueRecord,
        decomposition: Decomposition,
        hop: int,
        prior_answers: dict[int, str],
        k_hop: int,
        retrieve: Retriever,
    ) -> HopResult: ...
