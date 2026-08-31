"""CLI entrypoint. Run:

    python scripts/run_baseline.py --config configs/baseline.yaml

Prints overall and hop-wise EM/F1, and writes per-example predictions to
baseline/results/ for later inspection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Running this file directly puts scripts/ on sys.path, not the repo root, so
# the `src` / `baseline` / `evaluation` imports below need the root added.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data.musique_loader import load_split
from baseline import llm_cache
from baseline.llm_client import DailyTokenLimitExceeded, ProviderUnavailable, set_cache_enabled
from baseline.placeholder_retriever import retrieve as placeholder_retrieve
from baseline.qa_pipeline import answer_question
from evaluation.qa_eval import evaluate

RETRIEVERS = {
    "placeholder": placeholder_retrieve,
    # "bm25": bm25_retrieve,  # wire in once retrieval/bm25_retriever.py exists
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the response cache and re-query the API (costs daily token budget).",
    )
    args = parser.parse_args()

    set_cache_enabled(not args.no_cache)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    records = load_split(config["split"])
    if config.get("sample_size"):
        # The split is ordered by hop count, so a plain head slice yields an
        # all-2-hop sample and no 3/4-hop breakdown. Sample with a fixed seed
        # instead: representative of the split's hop mix, still reproducible.
        rng = random.Random(config.get("seed", 0))
        records = rng.sample(records, min(config["sample_size"], len(records)))

    retriever = RETRIEVERS[config["retriever"]]

    out_dir = Path("baseline/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Written as we go: a sweep is hundreds of API calls, and a crash partway
    # through should not throw away the answers already paid for.
    results = []
    exhausted = False
    with (out_dir / f"predictions_k{config['k']}.jsonl").open("w", encoding="utf-8") as handle:
        for position, record in enumerate(records, start=1):
            try:
                result = answer_question(
                    record,
                    retrieve=retriever,
                    k=config["k"],
                    model=config["model"],
                    split=config["split"],
                    provider=config.get("provider", "groq"),
                )
            except (DailyTokenLimitExceeded, ProviderUnavailable) as error:
                # Not retryable in any useful timeframe; keep what is already
                # answered and report on it rather than discarding the run.
                print(f"\nStopped at {position}/{len(records)}: backend unavailable or out of budget.", file=sys.stderr)
                print(f"  {error}", file=sys.stderr)
                exhausted = True
                break

            results.append(result)
            handle.write(json.dumps(result.__dict__) + "\n")
            handle.flush()

            if position % 25 == 0 or position == len(records):
                print(f"  ...{position}/{len(records)}", file=sys.stderr, flush=True)

    if not results:
        print("No questions were answered; nothing to score.", file=sys.stderr)
        return 1

    print(
        f"cache: {llm_cache.hits} hit(s), {llm_cache.misses} miss(es)"
        f"{'  [PARTIAL RUN]' if exhausted else ''}",
        file=sys.stderr,
    )

    report = evaluate(results)

    print(f"Overall  EM={report.overall.em:.3f}  F1={report.overall.f1:.3f}  (n={report.overall.count})")
    for hop, metrics in report.by_hop.items():
        print(f"{hop}-hop  EM={metrics.em:.3f}  F1={metrics.f1:.3f}  (n={metrics.count})")

    return 1 if exhausted else 0


if __name__ == "__main__":
    raise SystemExit(main())
