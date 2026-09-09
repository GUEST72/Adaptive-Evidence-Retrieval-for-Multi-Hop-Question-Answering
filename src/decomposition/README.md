# Question decomposition guide

`src.decomposition` creates structured multi-hop reasoning plans from
MuSiQue questions. Training examples are selected at runtime from the train
split; development questions are not embedded in source code.

## Output contract

The model must return 2-4 ordered steps:

```json
{
  "steps": [
    {
      "id": "step_1",
      "question": "Who founded Acme?",
      "depends_on": []
    },
    {
      "id": "step_2",
      "question": "When did [ANSWER_1] happen?",
      "depends_on": ["step_1"]
    }
  ]
}
```

The parser validates unique IDs, backward-only dependencies, hop count, and
placeholder/dependency consistency. It accepts plain JSON and common model
wrappers such as a JSON Markdown code fence.

## Offline mode

Use offline mode to build prompts without contacting an LLM:

```powershell
python scripts/run_decomposition.py `
  --max-examples 25 `
  --seed 13 `
  --report reports/decomposition.json
```

To score saved responses, provide an existing JSONL fixture with one response
per line. Each line may be a JSON string or an object with a `response` field:

```powershell
python scripts/run_decomposition.py `
  --responses path\to\responses.jsonl `
  --report reports/decomposition.json
```

## Live mode

Follow the configuration instructions in [`src/llm/README.md`](../llm/README.md),
then run:

```powershell
python scripts/run_decomposition.py `
  --live `
  --model qwen/qwen3.8-27b `
  --max-examples 5 `
  --seed 13 `
  --delay-seconds 6 `
  --rate-limit-retries 3 `
  --rate-limit-wait 20 `
  --parse-retries 1 `
  --report reports/decomposition.json
```

The JSON report records each prompt, parsed decomposition, intrinsic metrics,
and safe error details. It never records API keys. A successful run should have
`failure_count: 0`.

For larger batches, keep the pacing and bounded retry options enabled. They
reduce provider throttling without retrying indefinitely.

To run the required extrinsic answerability evaluation, add `--extrinsic`.
This makes one additional LLM answer call per generated hop using that hop's
gold supporting paragraph and gold previous-hop answers. The report then
contains overall, hop-position, and 2/3/4-hop answerability metrics:

```powershell
python scripts/run_decomposition.py `
  --live `
  --extrinsic `
  --model qwen/qwen3.8-27b `
  --max-examples 50 `
  --seed 13 `
  --delay-seconds 6 `
  --rate-limit-retries 3 `
  --rate-limit-wait 20 `
  --parse-retries 1 `
  --report reports/decomposition-50-extrinsic.json
```

## Metrics and limitations

Intrinsic metrics are position-aware ROUGE-1, ROUGE-L, and token F1. The
report also includes hop-count accuracy, dependency complexity, and 2/3/4-hop
breakdowns. Extrinsic answerability uses gold previous-hop answers and the
corresponding gold support paragraph; it does not perform retrieval.

Model output quality is not guaranteed by syntactic validity. Review parsed
steps and metrics before using decompositions in later experiments. Live runs
are subject to provider quotas, account-specific model availability, and
network conditions.

## Tests

```powershell
python -m pytest tests/test_decomposition.py tests/test_key_manager.py -v
python -m pytest -q
```
