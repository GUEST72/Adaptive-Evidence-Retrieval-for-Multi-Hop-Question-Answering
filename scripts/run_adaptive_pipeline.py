"""Week 3 end-to-end adaptive pipeline.

    python scripts/run_adaptive_pipeline.py --config configs/adaptive.yaml

Writes ``baseline/results/predictions_adaptive_k{k_hop}.jsonl``, matching
metadata, and ``reports/week3/adaptive_pipeline_results.json``.

Run ``python -m pytest -q`` before any live call. A live sample belongs only
after offline tests pass. Provider exhaustion stops the run; unattempted
records are omitted from scores rather than scored as zeros.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from baseline.llm_client import set_cache_enabled
from baseline.qa_pipeline import resolve_prompt_path
from baseline.retrievers import get_retriever
from baseline.runner import select_records
from src.data.musique_loader import load_split
from src.decomposition.generator import DecompositionGenerator
from src.decomposition.prompts import build_training_examples
from src.llm.groq import GroqClient
from src.week2.stopping.stopping_rule import (
    LexicalCoverageStoppingRule,
    LLMStoppingRule,
)
from src.week3.hop_executor import GeneratedHopExecutor
from src.week3.pipeline import (
    DEFAULT_REPORT_PATH,
    RESULTS_DIR,
    make_baseline_final_reader,
    print_report,
    run_adaptive_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adaptive.yaml"))
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the baseline reader cache (spends budget).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    set_cache_enabled(not args.no_cache)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    k_hop = int(config["k_hop"])
    reader_model = config["model"]
    reader_provider = config.get("provider", "groq")
    decomp_model = config.get("decomposition_model", config.get("model"))
    stopping_model = config.get("stopping_model", decomp_model)
    stopping_mode = config.get("stopping", "llm")
    retriever_name = config.get("retriever", "bm25")
    prompt_path = config.get("prompt_path")
    example_count = int(config.get("training_examples", 3))
    seed = int(config.get("seed", 13))

    records = select_records(config)
    try:
        train = load_split(config.get("train_split", "train"), config.get("data_dir"))
    except FileNotFoundError:
        train = records
    examples = build_training_examples(train, count=example_count, seed=seed)

    decomp_client = GroqClient(model=decomp_model)
    if not decomp_client.key_manager.available:
        parser.error(
            "this pipeline needs GROQ_API_KEY or GROQ_API_KEY_2 for decomposition "
            "(and for LLM stopping). A .env file is loaded for the baseline reader "
            "stack only; export Groq keys in the current session as well."
        )

    generator = DecompositionGenerator(decomp_client, model=decomp_model)

    def generate_decomposition(record):
        return generator.generate(record.question, examples)

    if stopping_mode == "lexical":
        stopping_rule = LexicalCoverageStoppingRule(
            min_coverage=float(config.get("min_coverage", 1.0))
        )
        stopping_provider = "lexical"
    elif stopping_mode == "llm":
        stopping_rule = LLMStoppingRule(decomp_client, model=stopping_model)
        stopping_provider = "groq"
    else:
        parser.error("stopping must be 'llm' or 'lexical'")

    retrieve = get_retriever(retriever_name)
    hop_executor = GeneratedHopExecutor(
        model=reader_model,
        provider=reader_provider,
        prompt_path=prompt_path,
    )
    final_reader = make_baseline_final_reader(
        model=reader_model,
        provider=reader_provider,
        prompt_path=prompt_path,
    )

    resolved_prompt = resolve_prompt_path(prompt_path)
    prompt_text = resolved_prompt.read_text(encoding="utf-8")
    metadata = {
        "seed": seed,
        "split": config.get("split", "dev"),
        "retriever": retriever_name,
        "reader_provider": reader_provider,
        "reader_model": reader_model,
        "decomposition_provider": "groq",
        "decomposition_model": decomp_model,
        "stopping_provider": stopping_provider,
        "stopping_model": stopping_model if stopping_mode == "llm" else None,
        "stopping": stopping_mode,
        "training_examples": example_count,
        "prompt_path": str(resolved_prompt.name),
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16],
    }

    try:
        outcome = run_adaptive_pipeline(
            records,
            k_hop=k_hop,
            retrieve=retrieve,
            hop_executor=hop_executor,
            stopping_rule=stopping_rule,
            generate_decomposition=generate_decomposition,
            final_reader=final_reader,
            results_dir=args.results_dir,
            report_path=args.report,
            metadata=metadata,
        )
    finally:
        decomp_client.close()

    if not outcome.results:
        print("No questions were attempted; nothing to score.", file=sys.stderr)
        return 1

    print_report(outcome)
    return 1 if outcome.exhausted else 0


if __name__ == "__main__":
    raise SystemExit(main())
