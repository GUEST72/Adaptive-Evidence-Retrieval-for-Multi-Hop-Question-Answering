"""CLI entrypoint. Run:

    python scripts/run_baseline.py --config configs/baseline.yaml

Prints overall and hop-wise EM/F1, and writes per-example predictions to
baseline/results/ for later inspection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Running this file directly puts scripts/ on sys.path, not the repo root, so
# the `src` / `baseline` / `evaluation` imports below need the root added.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data.musique_loader import load_split
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
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    records = load_split(config["split"])
    if config.get("sample_size"):
        records = records[: config["sample_size"]]

    retriever = RETRIEVERS[config["retriever"]]

    results = [
        answer_question(record, retrieve=retriever, k=config["k"], model=config["model"], split=config["split"])
        for record in records
    ]

    report = evaluate(results)

    print(f"Overall  EM={report.overall.em:.3f}  F1={report.overall.f1:.3f}  (n={report.overall.count})")
    for hop, metrics in report.by_hop.items():
        print(f"{hop}-hop  EM={metrics.em:.3f}  F1={metrics.f1:.3f}  (n={metrics.count})")

    out_dir = Path("baseline/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / f"predictions_k{config['k']}.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.__dict__) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
