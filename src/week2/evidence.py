"""The shared evidence-trace representation for Week 2.

Task 2 (hop-wise retrieval) produces traces; Task 3 (adaptive stopping) consumes
them. It lives here rather than under `hopwise/` so the stopping work can build
synthetic traces and be reviewed without depending on the retrieval branch.

A trace records what was retrieved *per hop*, not just the final evidence set.
The union is what retrieval metrics score, but the per-hop structure is what
makes "was the evidence sufficient after hop 2?" answerable at all, so it must
not be flattened away at construction time.

Paragraphs use the project's existing `RetrievedParagraph` shape
(`baseline/retriever_interface.py`) rather than a Week 2 duplicate, so a trace
carries exactly what a retriever returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from baseline.retriever_interface import RetrievedParagraph

__all__ = [
    "EvidenceStep",
    "EvidenceTrace",
    "accumulated_indices",
    "accumulated_paragraphs",
    "trace_from_json",
    "trace_to_json",
    "validate_trace",
]


@dataclass(frozen=True)
class EvidenceStep:
    """One reasoning hop: the query issued, and what it retrieved."""

    hop: int
    query: str
    retrieved_paragraphs: tuple[RetrievedParagraph, ...]

    @property
    def indices(self) -> tuple[int, ...]:
        """MuSiQue paragraph indices retrieved at this hop, in rank order."""
        return tuple(paragraph["idx"] for paragraph in self.retrieved_paragraphs)


# The spec defines a trace as an ordered sequence of steps; keeping it a plain
# list means Task 3 can build one literally, with no constructor to learn.
EvidenceTrace = list[EvidenceStep]


def validate_trace(trace: Sequence[EvidenceStep]) -> None:
    """Raise if hops are not numbered 1..n in ascending order.

    A trace with a missing or repeated hop silently breaks any "evidence after
    hop i" question, which is the whole reason the per-hop structure exists.
    """
    expected = list(range(1, len(trace) + 1))
    actual = [step.hop for step in trace]
    if actual != expected:
        raise ValueError(f"Evidence trace hops must be {expected}; got {actual}.")


def _dedup(paragraphs: Iterable[RetrievedParagraph]) -> list[RetrievedParagraph]:
    """First occurrence wins, so a paragraph keeps its earliest hop and best rank."""
    seen: set[int] = set()
    unique: list[RetrievedParagraph] = []
    for paragraph in paragraphs:
        if paragraph["idx"] not in seen:
            seen.add(paragraph["idx"])
            unique.append(paragraph)
    return unique


def accumulated_paragraphs(
    trace: Sequence[EvidenceStep], through_hop: int | None = None
) -> list[RetrievedParagraph]:
    """Deduplicated evidence gathered up to and including `through_hop`.

    `through_hop=None` means the whole trace. Passing a hop number gives the
    prefix a stopping rule sees when deciding whether to continue.
    """
    steps = trace if through_hop is None else [s for s in trace if s.hop <= through_hop]
    return _dedup(paragraph for step in steps for paragraph in step.retrieved_paragraphs)


def accumulated_indices(
    trace: Sequence[EvidenceStep], through_hop: int | None = None
) -> list[int]:
    """Paragraph indices of the accumulated evidence, in first-retrieved order."""
    return [paragraph["idx"] for paragraph in accumulated_paragraphs(trace, through_hop)]


def trace_to_json(trace: Sequence[EvidenceStep]) -> list[dict[str, Any]]:
    """Plain structures for a results file; round-trips via `trace_from_json`."""
    return [
        {
            "hop": step.hop,
            "query": step.query,
            "retrieved_paragraphs": [dict(p) for p in step.retrieved_paragraphs],
        }
        for step in trace
    ]


def trace_from_json(data: Sequence[Mapping[str, Any]]) -> EvidenceTrace:
    return [
        EvidenceStep(
            hop=int(entry["hop"]),
            query=entry["query"],
            retrieved_paragraphs=tuple(entry["retrieved_paragraphs"]),
        )
        for entry in data
    ]
