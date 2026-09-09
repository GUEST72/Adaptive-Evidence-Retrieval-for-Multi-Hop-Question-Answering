"""Run decomposition prompting and write a JSON report."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make direct ``python scripts/run_decomposition.py`` behave like ``python -m``
# without requiring users to install the repository as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.musique_loader import load_split
from src.decomposition.metrics import aggregate_extrinsic, intrinsic_metrics, aggregate_intrinsic
from src.decomposition.parser import parse_decomposition, ParseError
from src.decomposition.prompts import build_decomposition_prompt, build_training_examples
from src.decomposition.generator import DecompositionGenerator
from src.decomposition.evaluator import evaluate_generated_steps
from src.llm.groq import GroqClient
from src.llm.client import ProviderError
from src.llm.key_manager import NoUsableKeyError


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build decomposition prompts and optionally score JSONL responses."
    )
    parser.add_argument("--split", default="dev")
    parser.add_argument("--data-dir")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--report", type=Path, default=Path("reports/decomposition.json"))
    parser.add_argument("--training-examples", type=int, default=3, choices=range(3, 6))
    parser.add_argument("--live", action="store_true", help="Explicitly enable Groq calls.")
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument(
        "--extrinsic",
        action="store_true",
        help="Use the live LLM to answer each generated step with gold evidence.",
    )
    parser.add_argument("--delay-seconds", type=float, default=0.0,
                        help="Pause between live requests to reduce rate limiting.")
    parser.add_argument("--rate-limit-retries", type=int, default=2,
                        help="Retries per record after all keys are rate limited.")
    parser.add_argument("--rate-limit-wait", type=float, default=15.0,
                        help="Seconds to wait before retrying a rate-limited record.")
    parser.add_argument("--parse-retries", type=int, default=1,
                        help="Retries per record after malformed model JSON.")
    parser.add_argument("--parse-retry-wait", type=float, default=1.0,
                        help="Seconds to wait before retrying malformed model JSON.")
    args = parser.parse_args()
    if args.max_examples < 0:
        parser.error("--max-examples must be non-negative")
    if (
        args.delay_seconds < 0
        or args.rate_limit_retries < 0
        or args.rate_limit_wait < 0
        or args.parse_retries < 0
        or args.parse_retry_wait < 0
    ):
        parser.error("delay and retry settings must be non-negative")
    if args.responses and not args.responses.is_file():
        parser.error(
            f"response fixture not found: {args.responses}. "
            "Omit --responses to record prompts only, or provide an existing JSONL file."
        )
    if args.extrinsic and not args.live:
        parser.error("--extrinsic requires --live because it makes additional LLM calls.")
    train = load_split(args.train_split, args.data_dir)
    records = load_split(args.split, args.data_dir)
    chosen = random.Random(args.seed).sample(records, min(args.max_examples, len(records)))
    examples = build_training_examples(train, count=args.training_examples, seed=args.seed)
    responses = []
    if args.responses:
        with args.responses.open(encoding="utf-8") as handle:
            responses = [json.loads(line) for line in handle if line.strip()]
    client = GroqClient(model=args.model) if args.live and not responses else None
    if client is not None and not client.key_manager.available:
        parser.error(
            "live mode requires GROQ_API_KEY or GROQ_API_KEY_2 in the current "
            "PowerShell session. A .env file is not loaded automatically."
        )
    generator = (
        DecompositionGenerator(client, model=args.model)
        if client is not None
        else None
    )
    rows = []
    metric_rows = []
    extrinsic_rows = []
    failure_count = 0
    for i, record in enumerate(chosen):
        prompt = build_decomposition_prompt(record.question, examples)
        row = {"id": record.id, "hop_count": record.hop_count, "prompt": prompt}
        if i < len(responses):
            text = responses[i].get("response", responses[i]) if isinstance(responses[i], dict) else responses[i]
            try:
                parsed = parse_decomposition(str(text))
                gold = [step.question for step in record.question_decomposition]
                row["metrics"] = intrinsic_metrics(parsed, gold)
                row["parsed"] = {"steps": [step.__dict__ for step in parsed.steps]}
            except (ParseError, TypeError, ValueError) as exc:
                row["parse_error"] = str(exc)
                failure_count += 1
        elif generator is not None:
            for attempt in range(max(args.rate_limit_retries, args.parse_retries) + 1):
                try:
                    parsed = generator.generate(record.question, examples)
                    row["parsed"] = {"steps": [step.__dict__ for step in parsed.steps]}
                    row["metrics"] = intrinsic_metrics(
                        parsed, [s.question for s in record.question_decomposition]
                    )
                    if args.extrinsic:
                        row["extrinsic"] = evaluate_generated_steps(
                            record, parsed, client, model=args.model
                        )
                        extrinsic_rows.append(
                            (row["extrinsic"]["scores"], record.hop_count)
                        )
                    break
                except NoUsableKeyError as exc:
                    if attempt >= args.rate_limit_retries:
                        row["generation_error"] = type(exc).__name__
                        row["generation_error_message"] = str(exc)
                        failure_count += 1
                    else:
                        time.sleep(args.rate_limit_wait)
                        client.key_manager.reset()
                except ParseError as exc:
                    if attempt < args.parse_retries:
                        time.sleep(args.parse_retry_wait)
                        continue
                    row["generation_error"] = type(exc).__name__
                    row["generation_error_message"] = str(exc)
                    failure_count += 1
                    break
                except (ProviderError, RuntimeError, ValueError) as exc:
                    row["generation_error"] = type(exc).__name__
                    row["generation_error_message"] = str(exc)
                    failure_count += 1
                    break
                finally:
                    if args.delay_seconds and attempt == 0:
                        time.sleep(args.delay_seconds)
        if "metrics" in row:
            metric_rows.append((parsed, [s.question for s in record.question_decomposition], record.hop_count))
        rows.append(row)
    report = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "split": args.split, "seed": args.seed,
            "max_examples": args.max_examples,
            "training_example_count": len(examples),
            "offline_fixture": bool(args.responses),
            "live_generation": bool(args.live and generator),
            "extrinsic_evaluation": bool(args.extrinsic),
        },
        "count": len(rows), "failure_count": failure_count,
        "metrics": aggregate_intrinsic(metric_rows) if metric_rows else {},
        "extrinsic": aggregate_extrinsic(extrinsic_rows) if extrinsic_rows else {},
        "examples": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
