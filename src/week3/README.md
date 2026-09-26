# Week 3 — Adaptive Loop (Person 2)

## What it does

Drives a **generated** decomposition hop by hop, asking after each hop whether
the evidence gathered so far already answers the original question, and
abandoning the rest of the plan when it does.

```text
hop 1 -> execute -> accumulate -> sufficient? -- no --> hop 2 -> ...
                                      |
                                     yes
                                      v
                                     stop
```

## Why it exists

Week 2 showed hop-wise retrieval recovers far more evidence than one-shot, but
it always ran *every* hop. Retrieving beyond sufficiency costs budget and
latency, and feeds the reader distractors it does not need. This loop decides
when to stop, and the evaluator measures whether it stops at the right moment.

## Modules

| file | role |
| --- | --- |
| `adaptive_loop.py` | `run_adaptive_hop_loop(...)` -> `AdaptiveRunResult` |
| `live_trace_eval.py` | classifies stopping quality on a live trace |
| `contracts.py` | shared `HopResult` / `HopExecutor` (Person 1) |
| `hop_executor.py` | generated-hop retrieval + reader (Person 1) |

## The loop

```python
run_adaptive_hop_loop(record, decomposition, k_hop, retrieve,
                      hop_executor, stopping_rule) -> AdaptiveRunResult
```

Per hop: call the executor once, append its `EvidenceStep`, record the
intermediate answer under its hop number so hop *n+1* can resolve
`[ANSWER_n]`, then ask the stopping rule with the deduplicated accumulated
evidence and the hop's query as context.

**The break happens before the next iteration.** A stop therefore costs no
further executor call, and so no further retrieval or reader call — the saving
is the mechanism's purpose, not a side effect. A test asserts the executor call
count is exactly 1 after a stop at hop 1.

`AdaptiveRunResult` carries `trace`, `decisions`, `intermediate_answers`,
`stop_hop` (`None` when every planned hop ran) and `planned_hops`.

Executor, retriever and stopping rule all arrive by injection, so the loop makes
no provider calls and runs fully offline in tests.

### No gold data during execution

Execution reads only `record.id`, `record.question`, and the generated
decomposition. `question_decomposition`, gold hop answers and `is_supporting`
are evaluation-only. A test replaces `record.question_decomposition` with an
object that raises on any access; it was verified non-vacuous by confirming a
deliberately cheating executor trips it.

## The live-trace evaluator

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

A design point worth knowing: the trace **ends** at the stop, so there is never
a later prefix to compare against — "stopped before the sufficient prefix"
cannot arise positionally. An incomplete trace is therefore classified by what
ended it. That separates a *stopping* failure (`stopped_early`) from a
*retrieval or decomposition* failure (`never_sufficient`), which a positional
reading would have conflated.

Outcome strings are deliberately distinct from Week 2's `early`/`correct`/`late`
so the two classifications can never be silently mixed in one results file. The
modules share no code; a regression test pins their disagreement on the
one-hop-gathers-everything case.

Gold indices are read here. That is legitimate — this is scoring — and never
happens during execution.

## Running the tests

```bash
python -m pytest tests/test_adaptive_loop.py tests/test_live_trace_eval.py -v
python -m pytest tests/ -q
```

All offline: stubs only, no dataset, no network, no provider.

## Limitations

- The loop is only as good as the plan it is given. A decomposition with too few
  hops caps the evidence regardless of stopping quality.
- `stopped_early` records that stopping ended an incomplete trace. It cannot
  show whether continuing *would* have completed it — the hops were never run.
- Classification needs gold support indices, so it applies to labelled
  evaluation data only.

## For Person 3

`AdaptiveRunResult.as_dict()` serialises the run for a results file.
`evaluate_live_traces([(record, result), ...])` returns overall and per-hop-count
blocks plus per-question outcomes, ready to merge into
`reports/week3/adaptive_pipeline_results.json`.

Note there are now **two LLM stacks**: `baseline/llm_client.py` (hop executor,
with the response cache) and `src/llm/client.py` (stopping rule). The pipeline
will need to configure both, and the report should record models and providers
for each.
