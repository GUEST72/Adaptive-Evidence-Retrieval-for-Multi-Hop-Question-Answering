# Week 2 — Task 3: Adaptive Stopping and Supporting-Evidence F1

## What it does

After each reasoning hop, decide whether the evidence gathered so far is
enough to answer the original question, and score predicted supporting
paragraphs with official-style Supporting-Evidence precision / recall / F1.

```text
Evidence so far
       ↓
Stopping model
       ↓
Is it sufficient?
  /        \
YES        NO
 ↓          ↓
STOP    Continue
```

## Why it exists

Week 2 asks whether retrieval can stop when the evidence is already sufficient,
instead of always spending the full hop budget. That question is independent of
decomposition quality and of hop-wise recall, which is why this module is
evaluated on **synthetic gold-prefix traces** rather than waiting for Task 2.

The Week 2 stopping mechanism is a **per-hop evidence sufficiency check**. It is
inspired by iterative reasoning-and-retrieval termination behaviour,
particularly IRCoT, but it is **not** Adaptive-RAG's upfront complexity
classifier. An Adaptive-RAG-style router can be added later as a comparison.

## How it works

### 3A — Stopping

1. Build a synthetic evidence trace from gold MuSiQue decomposition order:
   hop 1 = gold paragraph 1, hop 2 = gold paragraphs 1–2, and so on.
2. At each hop, call a stopping rule on the original question plus the
   accumulated paragraphs. The rule does **not** receive gold hop count or gold
   indices.
3. Record the first hop where the rule says `stop=true`.
4. Label the outcome against the gold hop count `n`:

| Outcome | Meaning |
| --- | --- |
| early | stopped at hop `< n` (dangerous: missing evidence) |
| correct | stopped at hop `n` (all gold evidence just became available) |
| late | never stopped, even with the full gold set (wasted retrieval) |

The LLM rule asks for structured JSON:

```json
{"stop": true, "reason": "The evidence contains all required facts."}
```

Without an API key, the runner uses a lexical-coverage heuristic (stop when
every question token appears in the evidence). That heuristic is a reproducible
offline baseline, not the claimed research method.

### 3B — Supporting-Evidence F1

Compare predicted MuSiQue paragraph `idx` values to the gold supporting set:

```text
precision = |pred ∩ gold| / |pred|
recall    = |pred ∩ gold| / |gold|
F1        = harmonic mean
```

Empty prediction and empty gold → F1 = 1, matching MuSiQue / HotpotQA
`SupportMetric`. Instance scores are **macro-averaged**. This is the official
whole-set metric, not hop-wise evidence F1.

The Week 1 BM25 baseline prediction files are scored with this evaluator so the
project has an evidence-quality number before Week 3 integration.

## How to run it

From the repository root, after the MuSiQue-Ans files are in `data/musique_ans/`:

```powershell
python scripts/run_stopping_eval.py
```

This scores the checked-in Week 1 BM25 prediction JSONL files and runs the
lexical stopping experiment on a seeded 300-example dev sample (`seed=13`,
same seed as the Week 1 baseline).

Live LLM stopping (Groq):

```powershell
$env:GROQ_API_KEY = "your-groq-key"
python scripts/run_stopping_eval.py --live --model qwen/qwen3.8-27b --max-examples 25
```

Keys are read from the process environment only; `.env` is not loaded
automatically. See [the LLM guide](../../llm/README.md).

Tests:

```powershell
python -m pytest tests/test_stopping.py -v
python -m pytest -q
```

## Inputs and outputs

| Input | Role |
| --- | --- |
| MuSiQue-Ans split | gold paragraphs, hop count, questions |
| `baseline/results/predictions_bm25_k*.jsonl` | Week 1 predicted `retrieved_indices` |
| Groq (`--live`) | LLM sufficiency judgements |

Output: `reports/week2/stopping_results.json` with baseline support metrics,
overall early/correct/late rates, and a 2/3/4-hop breakdown.

The shared evidence-trace type is `src.week2.evidence.EvidenceStep`. Week 3 can
pass a real Task 2 trace into `LLMStoppingRule.decide` unchanged.

## Evaluation

- **Stopping:** rates of early / correct / late, overall and by 2/3/4-hop.
- **Evidence:** Supporting-Evidence P/R/F1 (and EM) on Week 1 BM25 k=3, 5, 10.

Seeded 300-question Week 1 sample (`seed=13`):

| k | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: |
| 3 | 0.349 | 0.412 | 0.371 |
| 5 | 0.273 | 0.534 | 0.355 |
| 10 | 0.179 | 0.691 | 0.281 |

Lexical stopping on synthetic gold prefixes (same sample): early 0.000, correct
0.003, late 0.997. Use `--live` for the LLM sufficiency rule.

## Limitations

- Synthetic traces give the stopper gold paragraphs in hop order. That isolates
  the sufficiency decision from retrieval errors; it is optimistic relative to
  noisy hop-wise retrieval.
- The lexical heuristic under-stops or over-stops whenever question wording is
  a poor proxy for the facts needed. Use `--live` for the LLM rule.
- Answer generation is optional in the JSON schema and is not scored here.
- MuSiQue-Full sufficiency pairs are out of scope for Week 2.
