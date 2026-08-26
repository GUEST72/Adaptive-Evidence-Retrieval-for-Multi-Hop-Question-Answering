"""Simple validation script for the local MuSiQue-Ans dataset."""

import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "musique_ans"

EXPECTED_HOP_COUNTS = {
    "train": {2: 14376, 3: 4387, 4: 1175},
    "dev": {2: 1252, 3: 760, 4: 405},
}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def validate_split(split):
    filename = {
    "train": "musique_ans_v1.0_train.jsonl",
    "dev": "musique_ans_v1.0_dev.jsonl",
    }[split]

    path = DATA_DIR / filename

    if not path.exists():
        print(f"[FAIL] Missing file: {path}")
        return False

    records = load_jsonl(path)
    hop_counts = Counter()
    issues = []

    for i, record in enumerate(records, start=1):
        # Basic schema checks
        required_fields = [
            "id",
            "question",
            "answer",
            "answer_aliases",
            "paragraphs",
            "question_decomposition",
        ]

        for field in required_fields:
            if field not in record:
                issues.append(f"line {i}: missing '{field}'")

        if "question_decomposition" not in record:
            continue

        # Hop count
        hop_count = len(record["question_decomposition"])
        hop_counts[hop_count] += 1

        if hop_count not in (2, 3, 4):
            issues.append(
                f"line {i} ({record.get('id', 'unknown')}): "
                f"unexpected hop count = {hop_count}"
            )

        # Paragraph validation
        paragraphs = record.get("paragraphs", [])

        if not isinstance(paragraphs, list) or not paragraphs:
            issues.append(
                f"line {i} ({record.get('id', 'unknown')}): "
                "invalid or empty paragraphs"
            )
            continue

        paragraph_indices = {
            paragraph.get("idx")
            for paragraph in paragraphs
            if isinstance(paragraph, dict)
        }

        # Check decomposition references
        for step in record["question_decomposition"]:
            support_idx = step.get("paragraph_support_idx")

            if support_idx not in paragraph_indices:
                issues.append(
                    f"line {i} ({record.get('id', 'unknown')}): "
                    f"invalid paragraph_support_idx={support_idx}"
                )

    print(f"\n{split.upper()}")
    print("-" * 30)
    print(f"Records: {len(records):,}")
    print(
        "Hops: "
        f"2={hop_counts[2]:,}, "
        f"3={hop_counts[3]:,}, "
        f"4={hop_counts[4]:,}"
    )

    expected = EXPECTED_HOP_COUNTS[split]

    if dict(hop_counts) == expected:
        print("Hop counts: PASS")
    else:
        print("Hop counts: FAIL")
        print(f"Expected: {expected}")
        print(f"Actual:   {dict(hop_counts)}")

    if issues:
        print(f"Validation: FAIL ({len(issues)} issue(s))")

        for issue in issues[:10]:
            print(f"  - {issue}")

        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        return False

    print("Schema / references: PASS")
    return dict(hop_counts) == expected


def main():
    print("MuSiQue-Ans Validation")
    print("=" * 30)

    train_ok = validate_split("train")
    dev_ok = validate_split("dev")

    print("\n" + "=" * 30)

    if train_ok and dev_ok:
        print("Dataset validation completed successfully.")
        return 0

    print("Dataset validation failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())