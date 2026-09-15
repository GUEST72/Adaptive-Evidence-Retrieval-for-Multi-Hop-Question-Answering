"""Week 2 Task 3 entrypoint.

Scores Supporting-Evidence F1 on Week 1 baseline prediction files and evaluates
adaptive stopping on synthetic gold-prefix traces.

    python scripts/run_stopping_eval.py
    python scripts/run_stopping_eval.py --live --model qwen/qwen3.8-27b --max-examples 25

Without ``--live`` the stopping experiment uses the lexical-coverage heuristic
so it needs no API key. ``--live`` uses Groq via ``GROQ_API_KEY``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.musique_loader import load_split
from src.llm.groq import GroqClient
from src.week2.stopping.evidence_evaluator import evaluate_prediction_rows
from src.week2.stopping.evaluator import evaluate_stopping
from src.week2.stopping.stopping_rule import LexicalCoverageStoppingRule, LLMStoppingRule

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_FILES = (
    PROJECT_ROOT / "baseline" / "results" / "predictions_bm25_k3.jsonl",
    PROJECT_ROOT / "baseline" / "results" / "predictions_bm25_k5.jsonl",
    PROJECT_ROOT / "baseline" / "results" / "predictions_bm25_k10.jsonl",
)


def _load_predictions(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate_baseline_predictions(
    path: Path,
    records_by_id: dict,
) -> dict:
    report = evaluate_prediction_rows(_load_predictions(path), records_by_id)
    report["path"] = str(path.as_posix())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--data-dir")
    parser.add_argument("--max-examples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/week2/stopping_results.json"),
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument(
        "--baseline",
        type=Path,
        nargs="*",
        default=list(DEFAULT_BASELINE_FILES),
        help="Week 1 prediction JSONL files to score with Supporting-Evidence F1.",
    )
    parser.add_argument(
        "--skip-stopping",
        action="store_true",
        help="Only score baseline evidence F1.",
    )
    args = parser.parse_args()
    if args.max_examples < 0:
        parser.error("--max-examples must be non-negative")

    records = load_split(args.split, args.data_dir, validate=False)
    records_by_id = {record.id: record for record in records}

    baseline_reports = []
    for path in args.baseline:
        if not path.is_file():
            print(f"Skipping missing baseline file: {path}")
            continue
        report = evaluate_baseline_predictions(path, records_by_id)
        baseline_reports.append(report)
        overall = report["overall"]
        print(
            f"{path.name}: n={report['n']}  "
            f"P={overall['precision']:.3f}  R={overall['recall']:.3f}  "
            f"F1={overall['f1']:.3f}"
        )

    stopping_payload = None
    if not args.skip_stopping:
        sample = random.Random(args.seed).sample(records, min(args.max_examples, len(records)))
        if args.live:
            client = GroqClient(model=args.model)
            if not client.key_manager.available:
                parser.error(
                    "live mode requires GROQ_API_KEY or GROQ_API_KEY_2 in the "
                    "current process. A .env file is not loaded automatically."
                )
            rule = LLMStoppingRule(client, model=args.model)
            method = f"llm:{args.model}"
        else:
            rule = LexicalCoverageStoppingRule(min_coverage=args.min_coverage)
            method = f"lexical_coverage:{args.min_coverage}"

        results, overall, by_hop = evaluate_stopping(sample, rule)
        stopping_payload = {
            "method": method,
            "n": len(results),
            "seed": args.seed,
            "split": args.split,
            "overall": overall.as_dict(),
            "by_hop": {str(hop): rates.as_dict() for hop, rates in by_hop.items()},
        }
        print(
            f"\nStopping ({method}), n={overall.count}: "
            f"early={overall.early:.3f}  correct={overall.correct:.3f}  "
            f"late={overall.late:.3f}"
        )
        for hop, rates in by_hop.items():
            print(
                f"  {hop}-hop n={rates.count}: "
                f"early={rates.early:.3f} correct={rates.correct:.3f} "
                f"late={rates.late:.3f}"
            )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framing": (
            "The Week 2 stopping mechanism is a per-hop evidence sufficiency "
            "check. It is inspired by iterative reasoning-and-retrieval "
            "termination behavior, particularly IRCoT, but it is not the same "
            "as Adaptive-RAG's upfront complexity classifier."
        ),
        "baseline_supporting_evidence": baseline_reports,
        "stopping": stopping_payload,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
