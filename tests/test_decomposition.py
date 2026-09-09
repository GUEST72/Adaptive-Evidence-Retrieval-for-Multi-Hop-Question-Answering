from __future__ import annotations

import json

import pytest

from src.decomposition.metrics import extrinsic_metrics, intrinsic_metrics
from src.decomposition.parser import parse_decomposition
from src.decomposition.prompts import build_decomposition_prompt, build_training_examples
from src.decomposition.models import ParseError
from src.decomposition.generator import DecompositionGenerator
from src.decomposition.evaluator import evaluate_generated_steps
from src.llm.client import LLMResponse


def response() -> str:
    return json.dumps({"steps": [
        {"id": "step_1", "question": "Who founded Acme?", "depends_on": []},
        {"id": "step_2", "question": "When did [ANSWER_1] happen?", "depends_on": ["step_1"]},
    ]})


def test_parser_validates_dependencies_and_placeholders():
    parsed = parse_decomposition(response())
    assert parsed.hop_count == 2
    assert parsed.steps[1].depends_on == ("step_1",)
    assert intrinsic_metrics(parsed, ["Who founded Acme?", "When did the founder happen?"])["position_aware"]["rouge_1"] > 0


def test_parser_accepts_json_with_model_wrapping():
    parsed = parse_decomposition(
        'Here is the decomposition:\n```json\n'
        + response()
        + '\n```'
    )
    assert parsed.hop_count == 2


@pytest.mark.parametrize("bad", [
    '{"steps":[]}',
    '{"steps":[{"id":"a","question":"x","depends_on":["b"]},{"id":"b","question":"y"}]}',
    '{"steps":[{"id":"a","question":"x"},{"id":"a","question":"y"}]}',
])
def test_parser_rejects_invalid_structures(bad):
    with pytest.raises(ParseError):
        parse_decomposition(bad)


@pytest.mark.parametrize("hop_count", [2, 3, 4])
def test_parser_accepts_supported_hop_counts(hop_count):
    steps = [
        {
            "id": f"step_{index}",
            "question": f"Question {index}",
            "depends_on": [f"step_{index - 1}"] if index > 1 else [],
        }
        for index in range(1, hop_count + 1)
    ]
    parsed = parse_decomposition(json.dumps({"steps": steps}))
    assert parsed.hop_count == hop_count


@pytest.mark.parametrize(
    "bad",
    [
        "not json",
        '{"missing_steps":[]}',
        '{"steps":[{"id":"step_1","question":"only one"}]}',
        '{"steps":[{"id":"step_1","question":"q"},'
        '{"id":"step_2","question":"q [ANSWER_1]","depends_on":[]}]}',
    ],
)
def test_parser_rejects_malformed_or_unusable_output(bad):
    with pytest.raises(ParseError):
        parse_decomposition(bad)


def test_examples_are_built_from_records(make_record):
    records = [make_record(f"id-{i}", (0, 1)) for i in range(4)]
    examples = build_training_examples(records, count=3, seed=4)
    assert len(examples) == 3
    assert examples[0]["steps"][1]["depends_on"] == ["step_1"]
    prompt = build_decomposition_prompt("new question", examples)
    assert "new question" in prompt
    assert "Example question:" in prompt


def test_extrinsic_scores_answer_and_support():
    scores = extrinsic_metrics("Paris", "Paris", predicted_support_indices=[1, 2], gold_support_indices=[2])
    assert scores["answer_exact"] == 1
    assert scores["support_recall"] == 1


class FakeLLM:
    provider = "fake"
    def __init__(self, texts):
        self.texts = iter(texts)
        self.prompts = []
    def complete(self, request):
        self.prompts.append(request.prompt)
        return LLMResponse(next(self.texts), request.model, self.provider)


def test_generator_uses_injected_client():
    client = FakeLLM([response()])
    generated = DecompositionGenerator(client, model="test").generate("q", [])
    assert generated.hop_count == 2


def test_extrinsic_generated_steps_uses_gold_evidence_and_substitution(make_record):
    record = make_record("x", (0, 1))
    # Replace synthetic questions/answers with a dependency-bearing chain.
    client = FakeLLM(["a0", "a1"])
    generated = parse_decomposition(response())
    result = evaluate_generated_steps(record, generated, client)
    assert len(result["scores"]) == 2
    assert record.paragraphs[0].paragraph_text in client.prompts[0]
    assert "[ANSWER_1]" not in client.prompts[1]
    assert "a0" in client.prompts[1]


def test_aggregate_extrinsic_breaks_down_by_position_and_complexity():
    from src.decomposition.metrics import aggregate_extrinsic

    result = aggregate_extrinsic([
        ([
            {"hop": 1, "answer_exact": 1.0, "answer_token_f1": 1.0},
            {"hop": 2, "answer_exact": 0.0, "answer_token_f1": 0.5},
        ], 2),
    ])
    assert result["overall"]["answer_exact"] == pytest.approx(0.5)
    assert result["by_position"]["2"]["answer_token_f1"] == pytest.approx(0.5)
    assert result["by_gold_hops"]["2"]["count"] == 2
