# Task 3 — Naive Retrieve-then-Answer Baseline

The first end-to-end AdaptiveHop baseline:

```
Question → Retrieve once → Top-k paragraphs → LLM → Final answer → EM/F1
```

One retrieval call and one LLM call per question. No decomposition, no
iterative retrieval, no adaptive stopping, no agent loop — those are later
tasks.

## Model

`claude-sonnet-4-6`, called through `baseline/llm_client.py` with
`max_tokens=64` and `temperature=0.0`. The model name lives in
`configs/baseline.yaml`, never in code. The API key is read from `.env` via
`python-dotenv` (`ANTHROPIC_API_KEY`); `.env` is gitignored.

Because `temperature=0.0`, single runs are effectively deterministic, so the
numbers below are not averaged over repeated runs.

## Retriever

**Placeholder lexical retriever** (`baseline/placeholder_retriever.py`) — a
stand-in until Task 2's BM25 retriever is merged. It scores each of a
question's own 20 paragraphs by query-token overlap, normalized by the square
root of paragraph length so long paragraphs don't win on size alone, and
returns the top *k* by score descending.

Retrieval is **closed per question**: only that question's 20 paragraphs are
searched — no cross-question or open-domain search. Paragraphs come from the
Task 1 loader (`src/data/musique_loader.get_question`); the JSONL is never
parsed here.

### Swapping in BM25 (Task 2)

`baseline/retriever_interface.py` defines the contract both implementations
satisfy:

```python
def __call__(self, query: str, question_id: str, k: int) -> list[RetrievedParagraph]
```

`baseline/qa_pipeline.py` contains no BM25-specific or placeholder-specific
logic. Integration once Task 2 lands is exactly two edits:

1. Add `"bm25": bm25_retrieve` to `RETRIEVERS` in `scripts/run_baseline.py`.
2. Change `retriever: placeholder` → `retriever: bm25` in
   `configs/baseline.yaml`.

The placeholder takes an extra `split: str = "dev"` keyword beyond the three
required arguments. The pipeline only ever passes the three required ones
positionally, so a BM25 retriever that omits `split` is still compatible —
but check its signature before wiring it in.

## Prompt

Fixed across every run being compared — a prompt change is not a valid
"improvement" between *k* or retriever settings. Stored at
`baseline/prompts/qa_prompt.txt`:

```text
Answer the question using only the evidence paragraphs below. If the
evidence does not contain the answer, respond with "unknown".

Respond with the final answer only — a short phrase or entity, no
explanation, no full sentence.

Evidence:
{evidence}

Question: {question}

Final answer:
```

Each retrieved paragraph is rendered as `[title]` followed by its text,
separated by blank lines.

## Evaluation

`evaluation/qa_eval.py` implements the standard MuSiQue/HotpotQA-style
metrics over the normalization in `baseline/answer_extraction.py`
(lowercase, strip punctuation, drop `a`/`an`/`the`, collapse whitespace):

- **EM** — 1 if the normalized prediction equals the normalized gold answer
  *or any of its `answer_aliases`*, else 0.
- **F1** — token-level F1 against the gold answer and each alias, taking the
  best score across candidates.

Both are reported overall and broken down by hop count (2/3/4), where hop
count is the number of gold decomposition steps.

## Running it

Install dependencies and set up the key:

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
```

The MuSiQue-Ans splits are not versioned in this repository — see
`data/musique_ans/README.md` for how to obtain them. They must be present
before the baseline can run.

Run the baseline:

```bash
python scripts/run_baseline.py --config configs/baseline.yaml
```

It prints overall and hop-wise EM/F1 and writes per-example predictions to
`baseline/results/predictions_k{k}.jsonl`.

Run the tests:

```bash
python -m pytest tests/test_placeholder_retriever.py tests/test_qa_eval.py -v
```

The eval tests and the retriever's scoring tests run fully offline. The four
end-to-end `retrieve` tests need the MuSiQue-Ans dev split on disk and skip
cleanly when it is absent.

## Results

> **Not yet measured.** The runs are blocked on two prerequisites that are
> not present in this environment: the MuSiQue-Ans JSONL splits are not
> downloaded, and no `ANTHROPIC_API_KEY` is configured. Once both are in
> place, run the smoke test (`sample_size: 5`), then `sample_size: 300` for
> each of *k* = 3, 5, 10, and fill in the tables below.

Sample: first *N* records of the `dev` split (`sample_size` in
`configs/baseline.yaml`).

### Overall

| k | n | EM | F1 |
| ---: | ---: | ---: | ---: |
| 3 | — | — | — |
| 5 | — | — | — |
| 10 | — | — | — |

### By hop count

| k | hops | n | EM | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 2 | — | — | — |
| 3 | 3 | — | — | — |
| 3 | 4 | — | — | — |
| 5 | 2 | — | — | — |
| 5 | 3 | — | — | — |
| 5 | 4 | — | — | — |
| 10 | 2 | — | — | — |
| 10 | 3 | — | — | — |
| 10 | 4 | — | — | — |

## Known limitations

- The retriever is a lexical-overlap placeholder, not BM25. Retrieval
  quality — and therefore every number above — is expected to move once
  Task 2 is merged.
- The default sample is the **first 300 dev records**, not a random sample,
  so the hop-count mix in the sample may not match the full dev split
  (1,252 / 760 / 405 for 2/3/4-hop).
- Single run per configuration. With `temperature=0.0` this is nearly
  deterministic, but no variance is reported.
- `get_question` re-reads and re-validates the whole split on every call, so
  a 300-question run reloads the dev split 300 times. This is correctness-
  neutral but slow; see the note in the pull request.
