"""CLI entrypoint. Run:

    python scripts/run_baseline.py --config configs/baseline.yaml

Prints overall and hop-wise EM/F1, and writes per-example predictions to
baseline/results/ for later inspection. The run loop itself lives in
baseline/runner.py so notebooks can call it in-process.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running this file directly puts scripts/ on sys.path, not the repo root, so
# the `baseline` / `evaluation` imports below need the root added.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from baseline.llm_client import set_cache_enabled
from baseline.placeholder_retriever import retrieve as placeholder_retrieve
from src.retrieval.bm25_retriever import retrieve as bm25_retrieve
from baseline.runner import print_report, run_baseline

RETRIEVERS = {
    "placeholder": placeholder_retrieve,
    "bm25": bm25_retrieve,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the response cache and re-query the backend (spends budget).",
    )
    args = parser.parse_args()

    set_cache_enabled(not args.no_cache)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    outcome = run_baseline(config, retrieve=RETRIEVERS[config["retriever"]])

    if not outcome.results:
        print("No questions were answered; nothing to score.", file=sys.stderr)
        return 1

    print_report(outcome)
    return 1 if outcome.exhausted else 0


if __name__ == "__main__":
    raise SystemExit(main())
