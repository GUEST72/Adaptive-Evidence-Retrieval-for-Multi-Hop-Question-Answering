"""Task 2 -- retrieval-only evaluation: does BM25 find the right evidence?

This script measures the *retriever* in isolation, with no LLM involved.
For every question in a split, it asks: out of the paragraphs BM25 ranked
highest, how many of the true `is_supporting` paragraphs actually show up?

Metrics reported, for k in {1, 2, 3, 5, 10} (per the project brief):

* Recall@k (macro-averaged): for each question, the fraction of its gold
  supporting paragraphs that appear in the top-k retrieved paragraphs,
  averaged evenly across questions (so a 4-hop question with 4 gold
  paragraphs is not weighted more than a 2-hop question with 2).
* Average number of gold paragraphs retrieved (raw count, not normalized
  by how many gold paragraphs the question has).
* Full-support coverage rate: the fraction of questions where *all* of
  that question's gold supporting paragraphs are present in the top-k
  (i.e. Recall@k == 1.0 for that question). This is the metric that
  matters most for later multi-hop experiments, since a downstream LLM
  can only reason correctly if it received every piece of evidence, not
  just some of it.

A hop-wise breakdown (2/3/4-hop) is also reported. Hop-wise breakdown is
formally Task 3's responsibility for the end-to-end QA numbers, but it
costs nothing extra here and is useful context for judging whether harder
(more-hop) questions are already a retrieval bottleneck before any LLM is
involved.

Usage:
    python -m src.evaluation.retrieval_eval
    python -m src.evaluation.retrieval_eval --split dev --k 1 2 3 5 10
    python -m src.evaluation.retrieval_eval --data-dir data/musique_ans
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.data.musique_loader import MuSiQueRecord, load_split, supporting_paragraphs
from src.retrieval.bm25_retriever import retrieve

DEFAULT_K_VALUES = (1, 2, 3, 5, 10)


@dataclass
class QuestionResult:
    """Per-question retrieval outcome at a single k."""

    question_id: str
    hop_count: int
    num_gold: int
    num_gold_retrieved: int

    @property
    def recall(self) -> float:
        """Fraction of this question's gold paragraphs that were retrieved.

        Defined as 1.0 (vacuously fully covered) if a question has zero gold
        supporting paragraphs; this should not occur in valid MuSiQue-Ans
        data (every hop has exactly one supporting paragraph), but is
        handled defensively rather than dividing by zero.
        """
        if self.num_gold == 0:
            return 1.0
        return self.num_gold_retrieved / self.num_gold

    @property
    def fully_covered(self) -> bool:
        return self.num_gold_retrieved == self.num_gold


def evaluate_at_k(
    records: list[MuSiQueRecord],
    k: int,
    split: str,
    data_dir: str | Path | None = None,
    ranked: dict[str, list[dict]] | None = None,
    validate: bool = False,
) -> list[QuestionResult]:
    """Run retrieval at a single k for every record and score each result.

    If `ranked` is provided, it must map question id -> paragraphs already
    ranked by `retrieve` at some k' >= k; this function then slices the
    prefix instead of rebuilding BM25.

    `validate` is forwarded to `retrieve`. Official MuSiQue-Ans contains a
    handful of records with fewer than 20 paragraphs, which Task 1's
    strict validator rejects, so evaluation defaults to False.
    """
    results: list[QuestionResult] = []

    for record in records:
        gold_indices = {paragraph.idx for paragraph in supporting_paragraphs(record)}
        if ranked is not None:
            retrieved = ranked[record.id][:k]
        else:
            retrieved = retrieve(
                record.question,
                record.id,
                k,
                split=split,
                data_dir=data_dir,
                validate=validate,
            )
        retrieved_indices = {item["idx"] for item in retrieved}

        results.append(
            QuestionResult(
                question_id=record.id,
                hop_count=record.hop_count,
                num_gold=len(gold_indices),
                num_gold_retrieved=len(gold_indices & retrieved_indices),
            )
        )

    return results


def summarize(results: list[QuestionResult]) -> dict[str, float]:
    """Aggregate per-question results into the headline metrics."""
    n = len(results)
    if n == 0:
        return {"recall": 0.0, "avg_gold_retrieved": 0.0, "full_coverage_rate": 0.0}

    return {
        "recall": sum(r.recall for r in results) / n,
        "avg_gold_retrieved": sum(r.num_gold_retrieved for r in results) / n,
        "full_coverage_rate": sum(1 for r in results if r.fully_covered) / n,
    }


def summarize_by_hop(results: list[QuestionResult]) -> dict[int, dict[str, float]]:
    """Same aggregation as `summarize`, broken down by hop count."""
    by_hop: dict[int, list[QuestionResult]] = defaultdict(list)
    for r in results:
        by_hop[r.hop_count].append(r)
    return {hop: summarize(group) for hop, group in sorted(by_hop.items())}


def run(
    split: str,
    k_values: tuple[int, ...],
    data_dir: str | Path | None = None,
    validate: bool = False,
) -> None:
    records = load_split(split, data_dir=data_dir, validate=validate)
    print(f"Loaded {len(records)} records from '{split}'.\n")

    print(f"{'k':>4} | {'Recall@k':>9} | {'Avg Gold Retrieved':>19} | {'Full Coverage':>13}")
    print("-" * 55)

    max_k = max(k_values)
    ranked = {
        record.id: retrieve(
            record.question,
            record.id,
            max_k,
            split=split,
            data_dir=data_dir,
            validate=validate,
        )
        for record in records
    }

    per_k_results: dict[int, list[QuestionResult]] = {}
    for k in k_values:
        results = evaluate_at_k(
            records, k, split, data_dir=data_dir, ranked=ranked, validate=validate
        )
        per_k_results[k] = results
        s = summarize(results)
        print(f"{k:>4} | {s['recall']:>9.3f} | {s['avg_gold_retrieved']:>19.3f} | {s['full_coverage_rate']:>13.3f}")

    print("\nHop-wise breakdown (Recall@k), for context:\n")
    for k in k_values:
        print(f"-- k = {k} --")
        by_hop = summarize_by_hop(per_k_results[k])
        for hop, s in by_hop.items():
            print(f"  {hop}-hop: recall={s['recall']:.3f}  full_coverage={s['full_coverage_rate']:.3f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=["train", "dev"])
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_K_VALUES))
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing the MuSiQue JSONL splits. Defaults to data/musique_ans/.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run Task 1's full-split validator before scoring. Official MuSiQue-Ans "
        "currently fails this check because a few records have fewer than 20 paragraphs.",
    )
    args = parser.parse_args()

    run(args.split, tuple(args.k), data_dir=args.data_dir, validate=args.validate)


if __name__ == "__main__":
    main()
