"""Execute one hop from a model-generated decomposition."""

from __future__ import annotations

import re
from pathlib import Path

from baseline import llm_client
from baseline.answer_extraction import extract_final_answer
from baseline.qa_pipeline import _build_prompt
from baseline.retriever_interface import RetrievedParagraph, Retriever
from src.data.musique_loader import MuSiQueRecord
from src.decomposition.models import Decomposition
from src.week2.evidence import EvidenceStep
from src.week3.contracts import HopResult

_ANSWER_PLACEHOLDER = re.compile(r"\[ANSWER_(\d+)\]")


class GeneratedHopExecutor:
    """Retrieve and answer one generated hop using configured reader settings.

    The executor only consumes the generated decomposition and evidence
    returned by the retriever. It never consults the record's gold
    decomposition or support annotations.
    """

    def __init__(
        self,
        *,
        model: str,
        provider: str = llm_client.DEFAULT_PROVIDER,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.prompt_path = prompt_path

    def __call__(
        self,
        record: MuSiQueRecord,
        decomposition: Decomposition,
        hop: int,
        prior_answers: dict[int, str],
        k_hop: int,
        retrieve: Retriever,
    ) -> HopResult:
        if (
            not isinstance(hop, int)
            or isinstance(hop, bool)
            or not 1 <= hop <= len(decomposition.steps)
        ):
            raise ValueError(
                f"hop must be between 1 and {len(decomposition.steps)}; got {hop!r}."
            )

        query_template = decomposition.steps[hop - 1].question

        def substitute(match: re.Match[str]) -> str:
            answer_number = int(match.group(1))
            try:
                return prior_answers[answer_number]
            except KeyError:
                raise ValueError(
                    f"Missing prior answer for required placeholder "
                    f"[ANSWER_{answer_number}] in hop {hop}."
                ) from None

        query = _ANSWER_PLACEHOLDER.sub(substitute, query_template)
        retrieved: list[RetrievedParagraph] = retrieve(query, record.id, k_hop)
        prompt = _build_prompt(query, retrieved, prompt_path=self.prompt_path)
        raw_answer = llm_client.call_llm(
            prompt,
            model=self.model,
            provider=self.provider,
        )
        return HopResult(
            step=EvidenceStep(
                hop=hop,
                query=query,
                retrieved_paragraphs=tuple(retrieved),
            ),
            intermediate_answer=extract_final_answer(raw_answer),
        )
