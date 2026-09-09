"""Week 2 Task 2 entrypoint. Run:

    python scripts/run_hopwise_retrieval.py --config configs/hopwise.yaml

Compares oracle hop-wise retrieval against the one-shot Week 1 condition at a
matched per-question retrieval budget, over the full dev split and over the
seeded sample the Week 1 baseline used. Makes no LLM calls and needs no API key.

Writes machine-readable results to the path named by the config.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Running this file directly puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.data.musique_loader import load_split
from baseline.retrievers import get_retriever
from src.week2.evidence import trace_to_json
from src.week2.hopwise.evaluator import ComparisonReport, evaluate_comparison, report_to_json
from src.week2.hopwise.retriever import hopwise_retrieve, substitution_stats


def print_report(name: str, report: ComparisonReport) -> None:
    header = (
        f"{'hops':>5} {'n':>5} {'one-shot R':>11} {'hop-wise R':>11} {'dR':>7}"
        f" | {'one-shot AG':>12} {'hop-wise AG':>12} {'dAG':>7} | {'uniq os':>8} {'uniq hw':>8}"
    )
    print(f"\n--- {name} | k_hop={report.k_hop} ---")
    print(header)
    print("-" * len(header))

    rows = [(str(hop), conditions) for hop, conditions in report.by_hop.items()]
    rows.append(("all", report.overall))

    for label, conditions in rows:
        one_shot, hopwise = conditions["one_shot"], conditions["hopwise"]
        print(
            f"{label:>5} {hopwise.count:>5} {one_shot.recall:>10.1%} {hopwise.recall:>10.1%}"
            f" {hopwise.recall - one_shot.recall:>+7.1%}"
            f" | {one_shot.all_gold:>11.1%} {hopwise.all_gold:>11.1%}"
            f" {hopwise.all_gold - one_shot.all_gold:>+7.1%}"
            f" | {one_shot.mean_unique_paragraphs:>8.1f} {hopwise.mean_unique_paragraphs:>8.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/hopwise.yaml"))
    parser.add_argument(
        "--sample",
        choices=("full", "seeded", "both"),
        default="both",
        help="Which evaluation sample(s) to score.",
    )
    parser.add_argument(
        "--dump-traces",
        action="store_true",
        help=(
            "Also write per-question evidence traces for the seeded sample, for "
            "Task 3 and Week 3 integration to consume."
        ),
    )
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    retrieve = get_retriever(config["retriever"])

    records = load_split(config["split"])
    seeded = random.Random(config.get("seed", 0)).sample(
        records, min(config["sample_size"], len(records))
    )

    samples: dict[str, list] = {}
    if args.sample in ("full", "both"):
        samples[f"full_{config['split']}"] = records
    if args.sample in ("seeded", "both"):
        samples[f"seed{config.get('seed', 0)}_{len(seeded)}"] = seeded

    output: dict[str, object] = {
        "config": {
            "split": config["split"],
            "retriever": config["retriever"],
            "k_hop": config["k_hop"],
            "seed": config.get("seed", 0),
            "sample_size": config["sample_size"],
        },
        "substitution": substitution_stats(records),
        "samples": {},
    }

    for name, sample in samples.items():
        print(f"\n===== {name}: {len(sample)} questions =====")
        per_k = {}
        for k_hop in config["k_hop"]:
            report = evaluate_comparison(sample, retrieve, k_hop)
            print_report(name, report)
            per_k[str(k_hop)] = report_to_json(report)
        output["samples"][name] = {"n": len(sample), "k_hop": per_k}

    if args.dump_traces:
        # Seeded sample only: the full split would be a large file, and Task 3
        # needs a representative set of real traces rather than all of them.
        for k_hop in config["k_hop"]:
            path = Path(config["output"]).with_name(f"hopwise_traces_k{k_hop}.jsonl")
            with path.open("w", encoding="utf-8") as handle:
                for record in seeded:
                    trace = hopwise_retrieve(record, k_hop=k_hop, retrieve=retrieve)
                    handle.write(
                        json.dumps(
                            {
                                "question_id": record.id,
                                "question": record.question,
                                "hop_count": record.hop_count,
                                "k_hop": k_hop,
                                "gold_supporting": sorted(
                                    p.idx for p in record.paragraphs if p.is_supporting
                                ),
                                "trace": trace_to_json(trace),
                            }
                        )
                        + "\n"
                    )
            print(f"wrote {path}")

    destination = Path(config["output"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {destination}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
