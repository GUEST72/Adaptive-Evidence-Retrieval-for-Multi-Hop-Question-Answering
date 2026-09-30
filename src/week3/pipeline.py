"""End-to-end Week 3 adaptive pipeline.

Chains generated decomposition, the live hop loop, and a final reader that
answers the *original* question from evidence at the actual stop point.

Execution never consults `record.question_decomposition`, gold hop answers, or
`is_supporting`. Those fields are evaluation-only. A decomposition parse
failure is a scored empty prediction (`status=parse_error`), not a skip and
not a gold-plan substitution.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from baseline.answer_extraction import extract_final_answer
from baseline.llm_client import call_llm
from baseline.providers import DailyTokenLimitExceeded, ProviderUnavailable
from baseline.qa_pipeline import QAResult, _build_prompt
from baseline.retriever_interface import RetrievedParagraph, Retriever
from evaluation.qa_eval import EvalReport, evaluate
from src.data.musique_loader import MuSiQueRecord
from src.decomposition.models import Decomposition, ParseError
from src.llm.client import ProviderError
from src.llm.key_manager import NoUsableKeyError
from src.week2.evidence import accumulated_paragraphs
from src.week2.stopping.evidence_evaluator import evaluate_prediction_rows
from src.week2.stopping.stopping_rule import AdaptiveStoppingRule
from src.week3.adaptive_loop import AdaptiveRunResult, run_adaptive_hop_loop
from src.week3.contracts import HopExecutor
from src.week3.live_trace_eval import evaluate_live_traces

STATUS_OK = "ok"
STATUS_PARSE_ERROR = "parse_error"

RESULTS_DIR = Path("baseline/results")
DEFAULT_REPORT_PATH = Path("reports/week3/adaptive_pipeline_results.json")
PROGRESS_EVERY = 25

GenerateDecomposition = Callable[[MuSiQueRecord], Decomposition]
FinalReader = Callable[[str, Sequence[RetrievedParagraph]], str]


@dataclass(frozen=True)
class AdaptiveQAResult(QAResult):
    """Week 1 QA fields plus adaptive-run provenance."""

    status: str = STATUS_OK
    generated_hop_count: int = 0
    stop_hop: int | None = None
    decisions: tuple[dict[str, object], ...] = ()
    planned_hops: int = 0


@dataclass
class AdaptivePipelineOutcome:
    report: EvalReport
    results: list[AdaptiveQAResult]
    exhausted: bool
    parse_failures: int
    predictions_path: Path
    metadata_path: Path
    report_path: Path
    live_stopping: dict[str, Any]


def make_baseline_final_reader(
    *,
    model: str,
    provider: str,
    prompt_path: str | Path | None = None,
) -> FinalReader:
    """Answer the original question from stop-point evidence. Does not retrieve."""

    def read_final(
        question: str, evidence: Sequence[RetrievedParagraph]
    ) -> str:
        prompt = _build_prompt(question, list(evidence), prompt_path=prompt_path)
        raw = call_llm(prompt, model=model, provider=provider)
        return extract_final_answer(raw)

    return read_final


def result_to_row(result: AdaptiveQAResult) -> dict[str, Any]:
    return {
        "question_id": result.question_id,
        "hop_count": result.hop_count,
        "predicted_answer": result.predicted_answer,
        "gold_answer": result.gold_answer,
        "gold_aliases": list(result.gold_aliases),
        "retrieved_indices": list(result.retrieved_indices),
        "status": result.status,
        "generated_hop_count": result.generated_hop_count,
        "stop_hop": result.stop_hop,
        "decisions": list(result.decisions),
        "planned_hops": result.planned_hops,
    }


def eval_report_to_dict(report: EvalReport) -> dict[str, Any]:
    return {
        "overall": asdict(report.overall),
        "by_hop": {str(hop): asdict(metrics) for hop, metrics in report.by_hop.items()},
    }


def _eval_hop_count(record: MuSiQueRecord) -> int:
    """Gold hop count for metric grouping only.

    Avoids `record.hop_count` when tests replace `question_decomposition` with a
    non-sequence sentinel: reading that property would call `len` on it.
    """
    decomposition = object.__getattribute__(record, "question_decomposition")
    if isinstance(decomposition, tuple):
        return len(decomposition)
    return 0


def _empty_result(record: MuSiQueRecord, *, status: str) -> AdaptiveQAResult:
    return AdaptiveQAResult(
        question_id=record.id,
        hop_count=_eval_hop_count(record),
        predicted_answer="",
        gold_answer=record.answer,
        gold_aliases=record.answer_aliases,
        retrieved_indices=(),
        status=status,
        generated_hop_count=0,
        stop_hop=None,
        decisions=(),
        planned_hops=0,
    )


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def execute_adaptive_record(
    record: MuSiQueRecord,
    *,
    k_hop: int,
    retrieve: Retriever,
    hop_executor: HopExecutor,
    stopping_rule: AdaptiveStoppingRule,
    generate_decomposition: GenerateDecomposition,
    final_reader: FinalReader,
) -> tuple[AdaptiveQAResult, AdaptiveRunResult | None]:
    """Run one record. Reads no gold decomposition or support labels."""
    try:
        decomposition = generate_decomposition(record)
    except ParseError:
        return _empty_result(record, status=STATUS_PARSE_ERROR), None

    run = run_adaptive_hop_loop(
        record,
        decomposition,
        k_hop,
        retrieve,
        hop_executor,
        stopping_rule,
    )
    evidence = accumulated_paragraphs(run.trace)
    predicted = final_reader(record.question, evidence)
    return (
        AdaptiveQAResult(
            question_id=record.id,
            hop_count=_eval_hop_count(record),
            predicted_answer=predicted,
            gold_answer=record.answer,
            gold_aliases=record.answer_aliases,
            retrieved_indices=tuple(paragraph["idx"] for paragraph in evidence),
            status=STATUS_OK,
            generated_hop_count=run.executed_hops,
            stop_hop=run.stop_hop,
            decisions=tuple(decision.as_dict() for decision in run.decisions),
            planned_hops=run.planned_hops,
        ),
        run,
    )


def run_adaptive_pipeline(
    records: Sequence[MuSiQueRecord],
    *,
    k_hop: int,
    retrieve: Retriever,
    hop_executor: HopExecutor,
    stopping_rule: AdaptiveStoppingRule,
    generate_decomposition: GenerateDecomposition,
    final_reader: FinalReader,
    results_dir: Path = RESULTS_DIR,
    report_path: Path = DEFAULT_REPORT_PATH,
    metadata: Mapping[str, Any] | None = None,
) -> AdaptivePipelineOutcome:
    """Attempt each record; provider exhaustion leaves unattempted rows unscored."""
    if k_hop <= 0:
        raise ValueError(f"k_hop must be positive; got {k_hop!r}.")

    results_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path = results_dir / f"predictions_adaptive_k{k_hop}.jsonl"

    results: list[AdaptiveQAResult] = []
    live_pairs: list[tuple[MuSiQueRecord, AdaptiveRunResult]] = []
    exhausted = False
    parse_failures = 0

    with predictions_path.open("w", encoding="utf-8") as handle:
        for position, record in enumerate(records, start=1):
            try:
                result, run = execute_adaptive_record(
                    record,
                    k_hop=k_hop,
                    retrieve=retrieve,
                    hop_executor=hop_executor,
                    stopping_rule=stopping_rule,
                    generate_decomposition=generate_decomposition,
                    final_reader=final_reader,
                )
            except (DailyTokenLimitExceeded, ProviderUnavailable, NoUsableKeyError, ProviderError) as error:
                print(
                    f"\nStopped at {position}/{len(records)}: backend unavailable or out of budget.",
                    file=sys.stderr,
                )
                print(f"  {error}", file=sys.stderr)
                exhausted = True
                break

            if result.status == STATUS_PARSE_ERROR:
                parse_failures += 1
            if run is not None:
                live_pairs.append((record, run))
            results.append(result)
            handle.write(json.dumps(result_to_row(result)) + "\n")
            handle.flush()

            if position % PROGRESS_EVERY == 0 or position == len(records):
                print(f"  ...{position}/{len(records)}", file=sys.stderr, flush=True)

    qa_report = evaluate(results)
    evidence_report = evaluate_prediction_rows(
        [result_to_row(result) for result in results],
        {record.id: record for record in records},
    )
    live_stopping = evaluate_live_traces(live_pairs)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "k_hop": k_hop,
        "sample_size": len(records),
        "questions_attempted": len(results),
        "parse_failures": parse_failures,
        "completed": sum(1 for result in results if result.status == STATUS_OK),
        "complete": not exhausted,
        "qa": eval_report_to_dict(qa_report),
        "supporting_evidence": evidence_report,
        "live_stopping": live_stopping,
        **dict(metadata or {}),
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    metadata_path = predictions_path.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(
            {
                "k_hop": k_hop,
                "sample_size": len(records),
                "questions_attempted": len(results),
                "parse_failures": parse_failures,
                "complete": not exhausted,
                "git_commit": _git_commit(),
                "run_at": payload["generated_at"],
                "predictions_path": str(predictions_path.as_posix()),
                "report_path": str(report_path.as_posix()),
                **dict(metadata or {}),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return AdaptivePipelineOutcome(
        report=qa_report,
        results=results,
        exhausted=exhausted,
        parse_failures=parse_failures,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        report_path=report_path,
        live_stopping=live_stopping,
    )


def print_report(outcome: AdaptivePipelineOutcome) -> None:
    suffix = "  [PARTIAL RUN]" if outcome.exhausted else ""
    print(
        f"attempted={len(outcome.results)}  parse_errors={outcome.parse_failures}{suffix}",
        file=sys.stderr,
    )
    report = outcome.report
    print(
        f"Overall  EM={report.overall.em:.3f}  F1={report.overall.f1:.3f}"
        f"  (n={report.overall.count})"
    )
    for hop, metrics in report.by_hop.items():
        print(f"{hop}-hop  EM={metrics.em:.3f}  F1={metrics.f1:.3f}  (n={metrics.count})")
    evidence = json.loads(outcome.report_path.read_text(encoding="utf-8")).get(
        "supporting_evidence", {}
    )
    overall = evidence.get("overall", {})
    if overall:
        print(
            f"Evidence  P={overall.get('precision', 0):.3f}  "
            f"R={overall.get('recall', 0):.3f}  F1={overall.get('f1', 0):.3f}"
        )
