# Week 3 — Generated Adaptive Pipeline

Week 2 measured the three pieces in isolation (generated decomposition, oracle
hop-wise retrieval, synthetic stopping). Week 3 runs a **generated** plan live:
substitute prior answers, retrieve per hop, stop when the original question is
already answerable, then read a final answer from evidence at that stop.

`src/week2/hopwise/` and `src/week2/stopping/` stay the oracle and synthetic
experiments. This package does not reuse them as the live path.

```text
question
  -> generate decomposition (train few-shot; parse failure = scored miss)
  -> hop 1: substitute -> retrieve -> intermediate answer
  -> accumulate evidence -> sufficient? -- no --> hop 2 -> ...
                                 |
                                yes
                                 v
                    final reader on original question
                    + evidence at the actual stop
```

## Shared rules

- Execution never reads `record.question_decomposition`, gold hop answers, or
  `is_supporting`. Those fields are evaluation-only.
- A decomposition parse failure is a failed generated prediction
  (`status=parse_error`, empty answer and evidence). Do not substitute the gold
  plan or drop the record from EM/F1.
- Live traces are **not** labelled `early` / `correct` / `late`. Those Week 2
  labels use gold hop count and apply only to synthetic prefixes.
- Offline tests use stubs only; they do not load MuSiQue or call a provider.
- Every live report records commit, seed, sample size, `k_hop`, all models and
  providers, completion state, and parse-failure count.

## Modules

| file | role |
| --- | --- |
| `contracts.py` | shared `HopResult` / `HopExecutor` |
| `hop_executor.py` | generated-hop retrieval + hop reader |
| `adaptive_loop.py` | `run_adaptive_hop_loop(...)` -> `AdaptiveRunResult` |
| `live_trace_eval.py` | classifies stopping on a live evidence prefix |
| `pipeline.py` | seeded-dev run, predictions, live-trace report |

## Shared contract

The executor is constructed with reader model, provider, and prompt path. Those
settings are not hidden module globals.

```python
@dataclass(frozen=True)
class HopResult:
    step: EvidenceStep
    intermediate_answer: str

class HopExecutor(Protocol):
    def __call__(
        self,
        record: MuSiQueRecord,
        decomposition: Decomposition,
        hop: int,
        prior_answers: dict[int, str],
        k_hop: int,
        retrieve: Retriever,
    ) -> HopResult: ...
```

## Generated hop executor

`GeneratedHopExecutor` consumes only the generated decomposition and retrieved
evidence. For generated hop `n`:

1. Read `decomposition.steps[n - 1].question`.
2. Substitute every validated `[ANSWER_N]` placeholder from `prior_answers`.
   A missing required answer is an error, not an empty-string fallback.
3. Call the existing `Retriever` with the substituted question, record ID, and
   `k_hop`.
4. Return an `EvidenceStep` containing that query and retrieved paragraphs.
5. Build the existing baseline QA prompt from those paragraphs, call cached
   `baseline.llm_client.call_llm`, extract the answer, and return `HopResult`.

```powershell
python -m pytest tests/test_hop_executor.py -v
```

## Adaptive loop

```python
run_adaptive_hop_loop(record, decomposition, k_hop, retrieve,
                      hop_executor, stopping_rule) -> AdaptiveRunResult
```

Per hop: call the executor once, append its `EvidenceStep`, record the
intermediate answer under its hop number so hop *n+1* can resolve
`[ANSWER_n]`, then ask the stopping rule with the deduplicated accumulated
evidence and the hop's query as context.

**The break happens before the next iteration.** A stop therefore costs no
further executor call, and so no further retrieval or reader call. A test
asserts the executor call count is exactly 1 after a stop at hop 1.

`AdaptiveRunResult` carries `trace`, `decisions`, `intermediate_answers`,
`stop_hop` (`None` when every planned hop ran) and `planned_hops`.

Executor, retriever and stopping rule all arrive by injection, so the loop makes
no provider calls and runs fully offline in tests.

A test replaces `record.question_decomposition` with an object that raises on
any access; it was verified non-vacuous by confirming a deliberately cheating
executor trips it.

## Live-trace evaluator

Week 2's `classify_stopping` compares the stop hop against **gold hop count**.
That is correct for synthetic traces, built one gold paragraph per hop, and
wrong here for two reasons:

1. A generated plan chooses its own hop count, so "hop 3 of 3" and "hop 3 of 2"
   are not comparable.
2. One hop can retrieve several gold paragraphs at once. Week 2 measured this:
   for *"Who is the spouse of the Green performer?"*, hop 2 recovered **both**
   gold paragraphs, including the one hop 1 missed.

So classification runs against what the trace actually accumulated — the first
evidence prefix containing every gold paragraph:

| outcome | meaning |
| --- | --- |
| `stopped_at` | stopped exactly at the hop that completed the evidence |
| `stopped_late` | kept retrieving after evidence was already complete |
| `stopped_early` | stopping ended a trace whose evidence was still incomplete |
| `never_sufficient` | ran the whole plan and still never gathered full support |

The trace **ends** at the stop, so there is never a later prefix to compare
against. An incomplete trace is classified by what ended it. That separates a
*stopping* failure (`stopped_early`) from a *retrieval or decomposition*
failure (`never_sufficient`).

Gold indices are read here. That is legitimate — this is scoring — and never
happens during execution.

## End-to-end pipeline

`pipeline.py` plus `scripts/run_adaptive_pipeline.py` run the Week 1 seeded
sample (`split: dev`, `seed: 13`, `sample_size: 300`) end to end:

1. Build train-derived few-shot examples **once** per run, then generate a
   decomposition.
2. On parse failure, emit empty predicted answer/evidence with
   `status=parse_error` and keep the record in the attempted denominator.
3. Run `run_adaptive_hop_loop` with `GeneratedHopExecutor`.
4. Build the baseline QA prompt from the **original question** and
   `accumulated_paragraphs` at the stop. It does not call `answer_question()`,
   which would retrieve again from the raw question.
5. Emit standard `QAResult` fields plus `status`, generated hop count, stop
   hop, decisions, and retrieved indices.

Score answers with `evaluation.qa_eval` and evidence with
`evaluate_prediction_rows`. Provider exhaustion produces an incomplete run:
records never attempted are omitted from scores, not zero-filled.

Outputs:

- `baseline/results/predictions_adaptive_k{k_hop}.jsonl`
- matching `.meta.json`
- `reports/week3/adaptive_pipeline_results.json` — EM/F1, supporting-evidence
  P/R/F1, parse/completion counts, and live stopping-prefix outcomes

Required live metadata: `git_commit`, `seed`, `sample_size`, `k_hop`, reader /
decomposition / stopping models and providers, `complete`, `parse_failures`.

There are **two LLM stacks**: `baseline/llm_client.py` (hop executor and final
reader, with the response cache) and `src/llm/client.py` (decomposition and the
LLM stopping rule). Lexical stopping (`stopping: lexical`) needs no stopping
model.

```powershell
python -m pytest tests/test_pipeline.py tests/test_hop_executor.py tests/test_adaptive_loop.py tests/test_live_trace_eval.py -v
python -m pytest -q
python scripts/run_adaptive_pipeline.py --config configs/adaptive.yaml
```

Run the full offline suite before any live call. A small live sample belongs
only after that.

## Limitations

- The loop is only as good as the plan it is given. A decomposition with too few
  hops caps the evidence regardless of stopping quality.
- `stopped_early` records that stopping ended an incomplete trace. It cannot
  show whether continuing *would* have completed it — the hops were never run.
- Classification needs gold support indices, so it applies to labelled
  evaluation data only.
- Live EM/F1 belongs in the root README only after a complete run whose
  metadata is reproducible.
