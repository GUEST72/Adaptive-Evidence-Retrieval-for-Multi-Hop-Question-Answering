"""Week 2 Task 3: per-hop adaptive stopping and Supporting-Evidence F1.

The stopping rule is a sufficiency check over evidence gathered so far. It is
inspired by iterative reasoning-and-retrieval termination, particularly IRCoT,
but it is not Adaptive-RAG's upfront complexity classifier.
"""

from .evidence_evaluator import (
    SupportScores,
    aggregate_support_scores,
    supporting_evidence_scores,
)
from .evaluator import classify_stopping, evaluate_stopping
from .stopping_rule import (
    AdaptiveStoppingRule,
    LexicalCoverageStoppingRule,
    LLMStoppingRule,
    StoppingDecision,
)
from .synthetic_traces import synthetic_gold_trace

__all__ = [
    "AdaptiveStoppingRule",
    "LexicalCoverageStoppingRule",
    "LLMStoppingRule",
    "StoppingDecision",
    "SupportScores",
    "aggregate_support_scores",
    "classify_stopping",
    "evaluate_stopping",
    "supporting_evidence_scores",
    "synthetic_gold_trace",
]
