# Week 2 Task 1 integration review

## What was reused

This work did not rebuild the project from scratch:

| Existing component | How decomposition uses it |
|---|---|
| `src.data.musique_loader` | Loads the same typed MuSiQue train/dev records and gold decompositions. |
| MuSiQue dataset placement and validation | Uses the existing `data/musique_ans/` files and split validation. |
| Existing Python/test setup | Adds tests to the existing pytest suite; no replacement test framework. |
| Existing project structure | Adds `src.decomposition`, `src.llm`, and a script without changing baseline package boundaries. |
| Existing Task 2/3 code | Remains available and unchanged; retrieval and baseline tests continue to pass. |

The decomposition prompt builder imports `MuSiQueRecord` directly. The runner
calls the existing `load_split()` for both training examples and evaluation
records. This preserves the original record schema and avoids a second dataset
loader.

## What was added

- Structured 2-4 hop decomposition models and validation.
- Runtime few-shot prompt construction from the train split.
- A provider-neutral `LLMClient` protocol.
- Groq transport with environment-key rotation and classified errors.
- Position-aware intrinsic metrics and retrieval-free extrinsic evaluation.
- An offline/live CLI with deterministic sampling, JSON reports, pacing, and
  bounded retries.

## What is not yet end-to-end

The new decomposition stage is **not yet inserted into the existing
retrieve-then-answer baseline loop**. The current baseline still performs:

```text
original question -> existing retriever -> existing reader -> final answer
```

The new Task 1 runner performs:

```text
MuSiQue record -> decomposition prompt -> LLM -> validated steps -> metrics
```

It does not yet pass generated sub-questions into BM25 retrieval, perform
iterative hop-wise retrieval, substitute answers during live retrieval, or
change the baseline's public retriever interface. This separation was
intentional so the existing baseline remains a stable comparison point. The
next integration task should connect validated steps to retrieval through a
new adapter rather than rewriting `retrieve()` or the baseline runner.

## 50-record live validation

Command used:

```powershell
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*(GROQ_API_KEY(?:_2)?)\s*=\s*(.*)\s*$') {
    [Environment]::SetEnvironmentVariable(
      $matches[1], $matches[2].Trim().Trim('"').Trim("'"), "Process"
    )
  }
}
python scripts/run_decomposition.py `
  --live `
  --model qwen/qwen3.8-27b `
  --max-examples 50 `
  --seed 13 `
  --delay-seconds 6 `
  --rate-limit-retries 3 `
  --rate-limit-wait 20 `
  --parse-retries 1 `
  --parse-retry-wait 2 `
  --report reports/live-verified-50.json
```

The full machine-readable output was written to
`reports/live-verified-50.json` locally. That generated JSON remains ignored;
the tracked summary and interpretation are documented below.

| Metric | Result |
|---|---:|
| Records processed | 50 |
| Valid parsed decompositions | 49 |
| Validation/generation failures | 1 |
| Gold 2/3/4-hop counts | 28 / 16 / 6 |
| Overall ROUGE-1 | 0.630 |
| Overall ROUGE-L | 0.574 |
| Overall token F1 | 0.630 |
| Reported hop-count accuracy | 0.857 |

The one failure was record
`4hop3__719125_132409_223216_35031`. The model referenced
`[ANSWER_2]` in `step_4` without declaring `step_2` in `depends_on`; the
strict validator correctly rejected it. No API key or transport error remained
in the final run.

The runner also supports `--extrinsic`, which evaluates each valid generated
step with its gold supporting paragraph and gold previous-hop answers. This is
separate from the 50-record intrinsic run above because it makes additional
LLM calls and has separate provider cost/quota implications.

The live extrinsic path was smoke-tested on 2 development records using the
same model. It completed 5 per-hop answer checks with zero generation
failures. The smoke metrics were:

| Metric | Result |
|---|---:|
| Per-hop checks | 5 |
| Exact answer accuracy | 0.000 |
| Answer token F1 | 0.087 |

These are smoke-test results, not a representative benchmark. A larger
extrinsic run should be performed when provider quota permits.

The full automated suite passed with **119 tests**:

```text
119 passed
```
