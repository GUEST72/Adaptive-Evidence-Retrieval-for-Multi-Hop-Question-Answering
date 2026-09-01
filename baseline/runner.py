"""The baseline run loop, callable in-process.

Kept out of scripts/run_baseline.py so a notebook that registers a provider at
runtime (see notebooks/run_eval_gpu.ipynb) can drive the same code path. A
subprocess would not inherit that registration.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.data.musique_loader import MuSiQueRecord, load_split
from baseline import llm_cache
from baseline.providers import DailyTokenLimitExceeded, ProviderUnavailable
from baseline.retrievers import RETRIEVERS, get_retriever
from baseline.qa_pipeline import QAResult, answer_question, resolve_prompt_path
from evaluation.qa_eval import EvalReport, evaluate

RESULTS_DIR = Path("baseline/results")
PROGRESS_EVERY = 25


@dataclass
class RunOutcome:
    report: EvalReport
    results: list[QAResult]
    exhausted: bool
    predictions_path: Path
    metadata_path: Path | None = None


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[1],
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def run_metadata(
    config: Mapping[str, Any], answered: int, exhausted: bool
) -> dict[str, Any]:
    """Provenance for one run.

    A predictions file records only answers, so nothing in it says which reader
    produced them — enough to mix up two models' results. The prompt digest is
    included because a changed template silently invalidates any comparison
    between runs.
    """
    prompt_path = resolve_prompt_path(config.get("prompt_path"))
    prompt_text = prompt_path.read_text(encoding="utf-8")

    return {
        "provider": config.get("provider", "groq"),
        "model": config["model"],
        "k": config["k"],
        "split": config["split"],
        "sample_size": config.get("sample_size"),
        "seed": config.get("seed", 0),
        "retriever": config.get("retriever"),
        "prompt_path": str(prompt_path.name),
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16],
        "questions_answered": answered,
        "complete": not exhausted,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }


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
    retrieve=None,
    records: Sequence[MuSiQueRecord] | None = None,
    results_dir: Path = RESULTS_DIR,
) -> RunOutcome:
    retriever_name = config.get("retriever", "placeholder")

    if retrieve is None:
        retrieve = get_retriever(retriever_name)
    elif retriever_name in RETRIEVERS and retrieve is not RETRIEVERS[retriever_name]:
        # The filename and the provenance record both come from the config, so
        # a caller passing a different function than the config names produces
        # results labelled as a retriever that never ran.
        raise ValueError(
            f"config names retriever {retriever_name!r} but a different function was "
            f"passed. Omit `retrieve` to use the configured one, or set "
            f"`retriever:` to match."
        )

    if records is None:
        records = select_records(config)

    results_dir.mkdir(parents=True, exist_ok=True)
    # The retriever is part of the name: with more than one retriever in the
    # repo, a k-only name lets one sweep silently overwrite another's results.
    predictions_path = results_dir / f"predictions_{retriever_name}_k{config['k']}.jsonl"

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
                    prompt_path=config.get("prompt_path"),
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

    metadata_path = predictions_path.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(run_metadata(config, len(results), exhausted), indent=2) + "\n",
        encoding="utf-8",
    )

    return RunOutcome(
        report=evaluate(results),
        results=results,
        exhausted=exhausted,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
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
