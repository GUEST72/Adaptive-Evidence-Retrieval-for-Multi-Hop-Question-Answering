"""Drive a generated decomposition hop by hop, stopping when evidence suffices.

This is the Week 3 control flow that Week 2 measured the parts of: execute one
generated hop, ask the stopping rule whether the evidence gathered so far can
already answer the original question, and abandon the remaining plan if it can.

Everything the loop needs arrives by injection — executor, retriever, stopping
rule — so the loop itself makes no provider calls and can be exercised offline.

**Execution reads no gold data.** Only `record.id`, `record.question`, and the
generated decomposition are consulted. `record.question_decomposition`, gold hop
answers and `is_supporting` are evaluation-only; using them here would make a
generated run silently oracle-assisted. See `live_trace_eval.py`, which is where
gold data legitimately enters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from baseline.retriever_interface import Retriever
from src.data.musique_loader import MuSiQueRecord
from src.decomposition.models import Decomposition
from src.week2.evidence import EvidenceStep, EvidenceTrace, accumulated_paragraphs
from src.week2.stopping.stopping_rule import AdaptiveStoppingRule, StoppingDecision
from src.week3.contracts import HopExecutor

__all__ = ["AdaptiveRunResult", "run_adaptive_hop_loop"]


@dataclass(frozen=True)
class AdaptiveRunResult:
    """Outcome of one adaptive run over a generated decomposition."""

    trace: EvidenceTrace
    decisions: list[StoppingDecision]
    intermediate_answers: dict[int, str]
    stop_hop: int | None
    planned_hops: int = 0

    @property
    def executed_hops(self) -> int:
        return len(self.trace)

    @property
    def stopped_early(self) -> bool:
        """Whether the loop abandoned hops the decomposition had planned.

        Names only what happened to the *plan*. Whether stopping was correct is
        a scoring question, answered in live_trace_eval.py.
        """
        return self.stop_hop is not None and self.stop_hop < self.planned_hops

    def as_dict(self) -> dict[str, object]:
        return {
            "planned_hops": self.planned_hops,
            "executed_hops": self.executed_hops,
            "stop_hop": self.stop_hop,
            "intermediate_answers": dict(self.intermediate_answers),
            "decisions": [decision.as_dict() for decision in self.decisions],
        }


def run_adaptive_hop_loop(
    record: MuSiQueRecord,
    decomposition: Decomposition,
    k_hop: int,
    retrieve: Retriever,
    hop_executor: HopExecutor,
    stopping_rule: AdaptiveStoppingRule,
) -> AdaptiveRunResult:
    """Execute generated hops in order, stopping as soon as evidence suffices.

    Args:
        record: Supplies the question and the closed paragraph pool. Gold fields
            are not read.
        decomposition: The *generated* plan. Its step count sets the upper bound
            on hops; the stopping rule may end the run sooner.
        k_hop: Retrieval slots per hop, passed through to the executor.
        retrieve: Retriever, passed through to the executor.
        hop_executor: Retrieves and answers one hop (`src/week3/contracts.py`).
        stopping_rule: Judges sufficiency after each hop.

    Returns:
        An `AdaptiveRunResult`. `stop_hop` is `None` when every planned hop ran.
    """
    if k_hop <= 0:
        raise ValueError(f"k_hop must be positive; got {k_hop!r}.")

    planned_hops = len(decomposition.steps)
    trace: EvidenceTrace = []
    decisions: list[StoppingDecision] = []
    prior_answers: dict[int, str] = {}
    stop_hop: int | None = None

    for hop in range(1, planned_hops + 1):
        result = hop_executor(
            record=record,
            decomposition=decomposition,
            hop=hop,
            prior_answers=prior_answers,
            k_hop=k_hop,
            retrieve=retrieve,
        )

        trace.append(result.step)
        # Keyed by hop number so the next hop can resolve [ANSWER_<hop>].
        prior_answers[hop] = result.intermediate_answer

        decision = stopping_rule.decide(
            record.question,
            accumulated_paragraphs(trace),
            hop=hop,
            context=result.step.query,
        )
        decisions.append(decision)

        if decision.stop:
            # Break here, before the next iteration, so a stop costs no further
            # executor call and therefore no further retrieval or reader call.
            stop_hop = hop
            break

    return AdaptiveRunResult(
        trace=trace,
        decisions=decisions,
        intermediate_answers=prior_answers,
        stop_hop=stop_hop,
        planned_hops=planned_hops,
    )
