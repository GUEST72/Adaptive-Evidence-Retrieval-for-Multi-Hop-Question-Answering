"""The baseline run loop, callable in-process.

Kept out of scripts/run_baseline.py so a notebook that registers a provider at
runtime (see notebooks/run_eval_gpu.ipynb) can drive the same code path. A
subprocess would not inherit that registration.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.musique_loader import MuSiQueRecord, load_split
from baseline import llm_cache
from baseline.providers import DailyTokenLimitExceeded, ProviderUnavailable
from baseline.qa_pipeline import QAResult, answer_question
from evaluation.qa_eval import EvalReport, evaluate

RESULTS_DIR = Path("baseline/results")
PROGRESS_EVERY = 25


@dataclass
class RunOutcome:
    report: EvalReport
    results: list[QAResult]
    exhausted: bool
    predictions_path: Path


def select_records(config: Mapping[str, Any]) -> list[MuSiQueRecord]:
    """The seeded evaluation sample.

    The dev split is ordered by hop count, so a plain head slice yields an
    all-2-hop sample and no 3/4-hop breakdown. Sampling with a fixed seed is
    both representative and reproducible.
    """
    records = load_split(config["split"])
    if not config.get("sample_size"):
        return list(records)

    rng = random.Random(config.get("seed", 0))
    return rng.sample(records, min(config["sample_size"], len(records)))


def run_baseline(
    config: Mapping[str, Any],
    retrieve,
    records: Sequence[MuSiQueRecord] | None = None,
    results_dir: Path = RESULTS_DIR,
) -> RunOutcome:
    if records is None:
        records = select_records(config)

    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = results_dir / f"predictions_k{config['k']}.jsonl"

    results: list[QAResult] = []
    exhausted = False

    # Written as we go: a sweep is hundreds of calls against a metered budget,
    # and a crash partway through must not discard answers already paid for.
    with predictions_path.open("w", encoding="utf-8") as handle:
        for position, record in enumerate(records, start=1):
            try:
                result = answer_question(
                    record,
                    retrieve=retrieve,
                    k=config["k"],
                    model=config["model"],
                    split=config["split"],
                    provider=config.get("provider", "groq"),
                )
            except (DailyTokenLimitExceeded, ProviderUnavailable) as error:
                # Not retryable in any useful timeframe; keep what is answered.
                print(
                    f"\nStopped at {position}/{len(records)}: backend unavailable or out of budget.",
                    file=sys.stderr,
                )
                print(f"  {error}", file=sys.stderr)
                exhausted = True
                break

            results.append(result)
            handle.write(json.dumps(result.__dict__) + "\n")
            handle.flush()

            if position % PROGRESS_EVERY == 0 or position == len(records):
                print(f"  ...{position}/{len(records)}", file=sys.stderr, flush=True)

    return RunOutcome(
        report=evaluate(results),
        results=results,
        exhausted=exhausted,
        predictions_path=predictions_path,
    )


def print_report(outcome: RunOutcome) -> None:
    print(
        f"cache: {llm_cache.hits} hit(s), {llm_cache.misses} miss(es)"
        f"{'  [PARTIAL RUN]' if outcome.exhausted else ''}",
        file=sys.stderr,
    )

    report = outcome.report
    print(
        f"Overall  EM={report.overall.em:.3f}  F1={report.overall.f1:.3f}"
        f"  (n={report.overall.count})"
    )
    for hop, metrics in report.by_hop.items():
        print(f"{hop}-hop  EM={metrics.em:.3f}  F1={metrics.f1:.3f}  (n={metrics.count})")
