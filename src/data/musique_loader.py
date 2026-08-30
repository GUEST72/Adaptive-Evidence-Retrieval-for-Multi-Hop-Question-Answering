"""Loading and validating the MuSiQue-Ans JSONL splits.

The module retains the original benchmark fields in typed dataclasses so later
experiments can map retrieved evidence back to MuSiQue's paragraph indices.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "musique_ans"
SPLIT_FILENAMES = {
    "train": ("train.jsonl", "musique_ans_v1.0_train.jsonl"),
    "dev": ("dev.jsonl", "musique_ans_v1.0_dev.jsonl"),
}
EXPECTED_HOP_COUNTS = {
    "train": {2: 14_376, 3: 4_387, 4: 1_175},
    "dev": {2: 1_252, 3: 760, 4: 405},
}


@dataclass(frozen=True)
class Paragraph:
    """A context paragraph, retaining its benchmark-local index."""

    idx: int
    title: str
    paragraph_text: str
    is_supporting: bool


@dataclass(frozen=True)
class DecompositionStep:
    """One gold decomposition step in its source-file order."""

    id: int
    question: str
    answer: str
    paragraph_support_idx: int


@dataclass(frozen=True)
class MuSiQueRecord:
    """A MuSiQue-Ans example with all fields needed by future project tasks."""

    id: str
    question: str
    answer: str
    answer_aliases: tuple[str, ...]
    paragraphs: tuple[Paragraph, ...]
    question_decomposition: tuple[DecompositionStep, ...]
    raw: Mapping[str, Any] = field(repr=False)

    @property
    def hop_count(self) -> int:
        """Reasoning depth, defined as the count of gold decomposition steps."""
        
        return len(self.question_decomposition)


@dataclass(frozen=True)
class ValidationIssue:
    """A single schema or benchmark-invariant violation."""

    code: str
    message: str
    question_id: str | None = None
    line_number: int | None = None


@dataclass
class ValidationReport:
    """Validation findings for one split or an in-memory record collection."""

    split: str | None = None
    record_count: int = 0
    hop_counts: Counter[int] = field(default_factory=Counter)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def add(self, code: str, message: str, record: Mapping[str, Any] | None = None, line_number: int | None = None) -> None:
        
        question_id = record.get("id") if isinstance(record, Mapping) and isinstance(record.get("id"), str) else None
        self.issues.append(ValidationIssue(code, message, question_id, line_number))


class DatasetValidationError(ValueError):
    """Raised when a JSONL split does not satisfy MuSiQue-Ans invariants."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        examples = "; ".join(
            f"{issue.question_id or 'unknown'}: {issue.message}" for issue in report.issues[:5]
        )
        super().__init__(f"MuSiQue validation failed with {len(report.issues)} issue(s). {examples}")


def data_directory(data_dir: str | Path | None = None) -> Path:
    """Resolve a configured data directory without relying on the current directory."""

    configured = data_dir or os.getenv("MUSIQUE_DATA_DIR")
    
    if configured is None:
        return DEFAULT_DATA_DIR
    
    candidate = Path(configured).expanduser()
    
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def split_path(split: str, data_dir: str | Path | None = None) -> Path:
    """Find a split under either the short project name or official release name."""

    normalized = split.lower()

    if normalized not in SPLIT_FILENAMES:
        raise ValueError(
            f"Unknown split {split!r}; expected one of {sorted(SPLIT_FILENAMES)}."
        )

    directory = data_directory(data_dir)
    candidates = [directory / filename for filename in SPLIT_FILENAMES[normalized]]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    names = " or ".join(path.name for path in candidates)

    raise FileNotFoundError(
        f"MuSiQue {normalized!r} split not found in {directory}. Expected {names}."
    )


def read_jsonl(path: str | Path) -> list[Mapping[str, Any]]:
    """Read a JSONL file safely, identifying malformed non-empty lines."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    records: list[Mapping[str, Any]] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Empty line in {file_path} at line {line_number}.")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed JSON in {file_path} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {file_path} at line {line_number}."
                )
            records.append(record)

    return records


def validate_records(records: Sequence[Mapping[str, Any]], split: str | None = None) -> ValidationReport:
    """Validate schema and gold-evidence invariants without altering input records."""

    report = ValidationReport(split=split, record_count=len(records))
    seen_ids: set[str] = set()
    
    for line_number, record in enumerate(records, start=1):
        _validate_record(record, report, seen_ids, line_number)
        
    if split in EXPECTED_HOP_COUNTS:
        expected = EXPECTED_HOP_COUNTS[split]
        actual = {hop: report.hop_counts.get(hop, 0) for hop in expected}
        if actual != expected:
            report.add("unexpected_split_counts", f"{split} hop counts are {actual}; expected {expected}.")
            
    return report


def _validate_record(record: Mapping[str, Any], report: ValidationReport, seen_ids: set[str], line_number: int) -> None:
    question_id = record.get("id")

    if not isinstance(question_id, str) or not question_id.strip():
        report.add(
            "missing_id", "Record ID must be a non-empty string.", record, line_number
        )

    elif question_id in seen_ids:
        report.add(
            "duplicate_id",
            "Record ID is duplicated within this split.",
            record,
            line_number,
        )

    else:
        seen_ids.add(question_id)

    if not isinstance(record.get("question"), str) or not record["question"].strip():
        report.add(
            "missing_question",
            "Question must be a non-empty string.",
            record,
            line_number,
        )

    if not isinstance(record.get("answer"), str):
        report.add(
            "missing_answer", "Answer must exist and be a string.", record, line_number
        )

    aliases = record.get("answer_aliases")

    if not isinstance(aliases, list) or not all(
        isinstance(alias, str) for alias in aliases
    ):
        report.add(
            "invalid_answer_aliases",
            "answer_aliases must be a list of strings.",
            record,
            line_number,
        )

    paragraphs = record.get("paragraphs")

    if not isinstance(paragraphs, list):
        report.add(
            "missing_paragraphs", "paragraphs must be a list.", record, line_number
        )
        paragraphs = []

    _validate_paragraphs(paragraphs, record, report, line_number)

    decomposition = record.get("question_decomposition")

    if not isinstance(decomposition, list):
        report.add(
            "missing_decomposition",
            "question_decomposition must be a list.",
            record,
            line_number,
        )
        return

    hop_count = len(decomposition)

    if hop_count not in {2, 3, 4}:
        report.add(
            "invalid_hop_count",
            f"Expected 2, 3, or 4 decomposition steps; found {hop_count}.",
            record,
            line_number,
        )
    else:
        report.hop_counts[hop_count] += 1

    _validate_decomposition(decomposition, paragraphs, record, report, line_number)


def _validate_paragraphs(paragraphs: list[Any], record: Mapping[str, Any], report: ValidationReport, line_number: int) -> None:

    # Most records carry 20 paragraphs, but the official MuSiQue-Ans release
    # genuinely ships shorter contexts (16 in dev, 21 in train, down to 16
    # paragraphs), so only an empty or oversized context is a real violation.
    if not paragraphs or len(paragraphs) > 20:
        report.add(
            "paragraph_count",
            f"Expected between 1 and 20 paragraphs; found {len(paragraphs)}.",
            record,
            line_number,
        )

    indices: list[int] = []

    for paragraph in paragraphs:

        if not isinstance(paragraph, dict):
            report.add(
                "invalid_paragraph",
                "Each paragraph must be an object.",
                record,
                line_number,
            )
            continue

        idx = paragraph.get("idx")

        if not isinstance(idx, int) or isinstance(idx, bool):
            report.add(
                "invalid_paragraph_index",
                "Paragraph idx must be an integer.",
                record,
                line_number,
            )
        else:
            indices.append(idx)

        if (
            not isinstance(paragraph.get("paragraph_text"), str)
            or not paragraph["paragraph_text"].strip()
        ):
            report.add(
                "empty_paragraph_text",
                "Paragraph text must be non-empty.",
                record,
                line_number,
            )

        if not isinstance(paragraph.get("is_supporting"), bool):
            report.add(
                "invalid_supporting_flag",
                "is_supporting must be boolean.",
                record,
                line_number,
            )

    if len(indices) == len(paragraphs) and set(indices) != set(range(len(paragraphs))):
        report.add(
            "inconsistent_paragraph_indices",
            f"Paragraph indices must contain each value from 0 through {len(paragraphs) - 1} exactly once.",
            record,
            line_number,
        )


def _validate_decomposition(decomposition: list[Any], paragraphs: list[Any], record: Mapping[str, Any], report: ValidationReport, line_number: int) -> None:
    by_index = {
        item.get("idx"): item
        for item in paragraphs
        if isinstance(item, dict) and isinstance(item.get("idx"), int)
    }

    for step in decomposition:

        if not isinstance(step, dict):
            report.add(
                "invalid_decomposition_step",
                "Each decomposition step must be an object.",
                record,
                line_number,
            )
            continue

        for field_name in ("id", "question", "answer", "paragraph_support_idx"):
            if field_name not in step:
                report.add(
                    "missing_decomposition_field",
                    f"Decomposition step is missing {field_name!r}.",
                    record,
                    line_number,
                )

        if not isinstance(step.get("id"), int) or isinstance(step.get("id"), bool):
            report.add(
                "invalid_decomposition_id",
                "Decomposition step id must be an integer.",
                record,
                line_number,
            )

        if (
            not isinstance(step.get("question"), str)
            or not step.get("question", "").strip()
        ):
            report.add(
                "invalid_decomposition_question",
                "Decomposition question must be non-empty.",
                record,
                line_number,
            )

        if not isinstance(step.get("answer"), str):
            report.add(
                "invalid_decomposition_answer",
                "Decomposition answer must be a string.",
                record,
                line_number,
            )

        support_idx = step.get("paragraph_support_idx")

        if not isinstance(support_idx, int) or isinstance(support_idx, bool):
            report.add(
                "invalid_supporting_reference",
                "paragraph_support_idx must be an integer.",
                record,
                line_number,
            )

        elif support_idx not in by_index:
            report.add(
                "invalid_supporting_reference",
                f"paragraph_support_idx {support_idx} does not refer to a paragraph.",
                record,
                line_number,
            )

        elif by_index[support_idx].get("is_supporting") is not True:
            report.add(
                "non_supporting_reference",
                f"paragraph_support_idx {support_idx} is not marked is_supporting=true.",
                record,
                line_number,
            )


def to_record(record: Mapping[str, Any]) -> MuSiQueRecord:
    """Convert a previously validated raw object into an immutable typed record."""

    return MuSiQueRecord(
        id=record["id"], question=record["question"], answer=record["answer"],
        answer_aliases=tuple(record["answer_aliases"]),
        paragraphs=tuple(Paragraph(**paragraph) for paragraph in record["paragraphs"]),
        question_decomposition=tuple(DecompositionStep(**step) for step in record["question_decomposition"]),
        raw=record,
    )


def load_split(split: str, data_dir: str | Path | None = None, validate: bool = True) -> list[MuSiQueRecord]:
    """Load a named MuSiQue split, raising ``DatasetValidationError`` by default."""

    raw_records = read_jsonl(split_path(split, data_dir))
    report = validate_records(raw_records, split=split.lower())
    
    if validate and not report.is_valid:
        raise DatasetValidationError(report)
    
    return [to_record(record) for record in raw_records]


def get_question(question_id: str, split: str = "dev", data_dir: str | Path | None = None) -> MuSiQueRecord:
    """Return one record by benchmark question ID or raise ``KeyError`` if absent."""

    for record in load_split(split, data_dir):
        
        if record.id == question_id:
            return record
        
    raise KeyError(f"Question ID {question_id!r} was not found in the {split!r} split.")


def supporting_paragraphs(record: MuSiQueRecord) -> tuple[Paragraph, ...]:
    """Return the record's gold supporting paragraphs in original paragraph order."""

    return tuple(paragraph for paragraph in record.paragraphs if paragraph.is_supporting)
