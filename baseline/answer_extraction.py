"""Normalization matching the standard SQuAD/HotpotQA/MuSiQue-style
convention, so EM/F1 are computed the same way the official scripts do."""

from __future__ import annotations

import re
import string


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def extract_final_answer(raw_llm_output: str) -> str:
    """The prompt already instructs the model to return only the final
    answer, but strip common leading labels defensively in case it doesn't
    comply exactly."""
    text = raw_llm_output.strip()
    text = re.sub(r"(?i)^final answer:\s*", "", text)
    return text.strip()
