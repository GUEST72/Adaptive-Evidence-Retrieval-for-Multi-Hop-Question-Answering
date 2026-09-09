"""Oracle hop-wise retrieval over a question's own paragraphs.

For each gold decomposition step, build a retrieval query and retrieve
separately, accumulating evidence across hops. Queries are conditioned on
previous hops via MuSiQue's `#N` references, which are replaced with the *gold*
answer of hop N — this is an oracle experiment, so the retrieval contribution is
isolated from decomposition and answering errors alike.

`retrieve` is a required argument everywhere in this module. Defaulting it would
let a caller run one retriever while a config elsewhere names another, which has
already produced a full set of mislabelled results once in this project.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from src.data.musique_loader import MuSiQueRecord
from baseline.retriever_interface import Retriever
from src.week2.evidence import EvidenceStep, EvidenceTrace

__all__ = [
    "HopQuery",
    "build_hop_queries",
    "hopwise_retrieve",
    "one_shot_retrieve",
    "slot_budget",
]

# MuSiQue writes previous-hop dependencies as "#1", "#2", ... inside the
# sub-question text, e.g. "#1 >> spouse". Verified across the full dev split:
# never in a first step, never forward or self-referential, never out of range.
REFERENCE_RE = re.compile(r"#(\d+)")


@dataclass(frozen=True)
class HopQuery:
    """One hop's retrieval query, and how it was derived."""

    hop: int
    sub_question: str
    query: str
    substitutions: tuple[tuple[int, str], ...] = ()

    @property
    def is_conditioned(self) -> bool:
        """Whether this query used a previous hop's answer."""
        return bool(self.substitutions)


def build_hop_queries(record: MuSiQueRecord) -> list[HopQuery]:
    """Gold sub-questions with `#N` replaced by the gold answer of hop N.

    Steps carrying no `#N` are returned verbatim. Those are not "implicit
    dependencies" needing inference: MuSiQue includes genuinely independent hops
    that name their entity outright (e.g. "Duane Courtney >> member of sports
    team"), and they are 14.5% of dev questions. They are retrieved as written
    rather than skipped.

    The `A >> relation` form used by 36% of steps needs no rewriting: BM25
    tokenises on `[a-z0-9]+`, so `>>` is dropped before scoring either way.
    """
    answers: dict[int, str] = {}
    queries: list[HopQuery] = []

    for hop, step in enumerate(record.question_decomposition, start=1):
        applied: list[tuple[int, str]] = []

        def substitute(match: re.Match[str]) -> str:
            reference = int(match.group(1))
            if reference not in answers:
                # Unreachable on the released data, but leaving the marker in
                # place beats silently emitting an empty string into the query.
                return match.group(0)
            applied.append((reference, answers[reference]))
            return answers[reference]

        query = REFERENCE_RE.sub(substitute, step.question)
        queries.append(
            HopQuery(
                hop=hop,
                sub_question=step.question,
                query=query,
                substitutions=tuple(applied),
            )
        )
        answers[hop] = step.answer

    return queries


def slot_budget(record: MuSiQueRecord, k_hop: int) -> int:
    """Total retrieval slots a hop-wise run spends: one `k_hop` per hop.

    The one-shot condition is given exactly this `k`, so neither system is
    compared against a larger retrieval allowance (spec section 14).
    """
    return record.hop_count * k_hop


def hopwise_retrieve(
    record: MuSiQueRecord,
    k_hop: int,
    retrieve: Retriever,
) -> EvidenceTrace:
    """Retrieve `k_hop` paragraphs per hop, in order, conditioned on gold answers."""
    if k_hop <= 0:
        raise ValueError(f"k_hop must be positive; got {k_hop!r}.")

    return [
        EvidenceStep(
            hop=hop_query.hop,
            query=hop_query.query,
            retrieved_paragraphs=tuple(retrieve(hop_query.query, record.id, k_hop)),
        )
        for hop_query in build_hop_queries(record)
    ]


def one_shot_retrieve(
    record: MuSiQueRecord,
    budget: int,
    retrieve: Retriever,
) -> EvidenceTrace:
    """The Week 1 baseline condition, expressed as a single-step trace.

    Returning a trace rather than a bare list lets the evaluator score both
    conditions through identical code.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive; got {budget!r}.")

    return [
        EvidenceStep(
            hop=1,
            query=record.question,
            retrieved_paragraphs=tuple(retrieve(record.question, record.id, budget)),
        )
    ]


def substitution_stats(records: Sequence[MuSiQueRecord]) -> dict[str, int]:
    """Counts backing the documented substitution behaviour."""
    stats = {
        "records": len(records),
        "steps": 0,
        "first_steps": 0,
        "steps_with_reference": 0,
        "later_steps_without_reference": 0,
        "records_with_independent_hop": 0,
    }

    for record in records:
        queries = build_hop_queries(record)
        has_independent = False
        for hop_query in queries:
            stats["steps"] += 1
            if hop_query.hop == 1:
                stats["first_steps"] += 1
            elif hop_query.is_conditioned:
                stats["steps_with_reference"] += 1
            else:
                stats["later_steps_without_reference"] += 1
                has_independent = True
        if has_independent:
            stats["records_with_independent_hop"] += 1

    return stats
