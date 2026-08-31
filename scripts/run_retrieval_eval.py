"""Offline retrieval sweep. Makes no API calls and needs no key. Run:

    python scripts/run_retrieval_eval.py --config configs/baseline.yaml

Reports recall@k, all-gold-retrieved@k, and MRR for each k, overall and by hop
count, over the same seeded sample the QA baseline uses.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data.musique_loader import load_split
from baseline.placeholder_retriever import retrieve as placeholder_retrieve
from evaluation.retrieval_eval import evaluate_retrieval

RETRIEVERS = {
    "placeholder": placeholder_retrieve,
    # "bm25": bm25_retrieve,  # wire in once retrieval/bm25_retriever.py exists
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 10])
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    records = load_split(config["split"])
    if config.get("sample_size"):
        rng = random.Random(config.get("seed", 0))
        records = rng.sample(records, min(config["sample_size"], len(records)))

    retriever = RETRIEVERS[config["retriever"]]

    print(f"retriever={config['retriever']}  split={config['split']}  n={len(records)}")
    print(f"{'k':>4} {'hops':>5} {'n':>5} {'recall':>8} {'all gold':>9} {'MRR':>7}")

    for k in args.k:
        report = evaluate_retrieval(records, retriever, k)
        overall = report.overall
        print(
            f"{k:>4} {'all':>5} {overall.count:>5} {overall.recall:>7.1%} "
            f"{overall.all_gold:>8.1%} {overall.mrr:>7.3f}"
        )
        for hop, metrics in report.by_hop.items():
            print(
                f"{'':>4} {hop:>5} {metrics.count:>5} {metrics.recall:>7.1%} "
                f"{metrics.all_gold:>8.1%} {metrics.mrr:>7.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
