# Task 3 — Naive Retrieve-then-Answer Baseline

The first end-to-end AdaptiveHop baseline:

```
Question → Retrieve once → Top-k paragraphs → LLM → Final answer → EM/F1
```

One retrieval call and one LLM call per question. No decomposition, no
iterative retrieval, no adaptive stopping, no agent loop — those are later
tasks.

## Model and provider

<<<<<<< Updated upstream
The backend is a config choice (`provider:` in `configs/*.yaml`), not a code
change — `qa_pipeline.py` only ever calls `call_llm`. See
`baseline/providers.py`.
=======
`openai/gpt-oss-120b`, served by **Groq** and called through
`baseline/llm_client.py` with `max_tokens=512`, `temperature=0.0`, and
`reasoning_effort="low"`.

> The project plan originally specified `claude-sonnet-4-6` on the Anthropic
> API. The API key available for this work is a Groq key (`gsk_…`), so the
> client targets Groq's OpenAI-compatible chat-completions endpoint instead.
> `call_llm`'s signature is unchanged, so `qa_pipeline.py` never had to know.
> Switching back to Anthropic means rewriting only `baseline/llm_client.py`.

`gpt-oss` reasons before answering, so `reasoning_effort` is pinned low and
`max_tokens` is above the answer's needs — otherwise the budget is spent on
hidden deliberation and the response comes back empty.

The model name lives in `configs/baseline.yaml`, never in code. The API key is
read from `.env` via `python-dotenv` (`GROQ_API_KEY`); `.env` is gitignored.
>>>>>>> Stashed changes

| provider | model | why |
| --- | --- | --- |
| `groq` (default) | `openai/gpt-oss-120b` | fast, but only 200k tokens/day free |
| `gemini` | `gemini-2.5-flash` | much larger free daily budget; fits a full sweep |
| `ollama` | `qwen2.5:7b-instruct` | local, no quota at all |

Called with `max_tokens=512`, `temperature=0.0`. Keys are read from `.env` via
`python-dotenv` (`GROQ_API_KEY` / `GEMINI_API_KEY`); `.env` is gitignored.

> The project plan originally specified `claude-sonnet-4-6` on the Anthropic
> API. The available key was a Groq key, so the baseline targets Groq's
> OpenAI-compatible endpoint instead. `call_llm`'s signature never changed, so
> `qa_pipeline.py` was untouched by the switch.

`gpt-oss` and Gemini 2.5 both reason before answering, so reasoning is minimised
(`reasoning_effort: low`, `thinkingBudget: 0`) — otherwise the output budget is
spent on hidden deliberation and the response comes back empty.

Despite `temperature=0.0`, repeat runs are **not** bit-identical: the same k=3
config scored EM 0.063 one day and 0.073 the next from an identical seed and
sample. Treat differences below about 1 EM point as noise.

### The real budget constraint

Groq's free tier limits **tokens per day to 200,000**, and this is reported
*only* in the 429 error body — no response header exposes it, which is why a
per-minute throttle never sees it coming. Measured cost is ~485 tokens per
question at k=5 (431 prompt + 54 completion):

| run (n=300) | tokens | share of one day's free budget |
| --- | ---: | ---: |
| k=3 | ~101k | 50% |
| k=5 | ~146k | 73% |
| k=10 | ~256k | 128% — impossible in a single day |

All three sweeps at n=300 need ~500k tokens: 2.5 days of Groq's free tier.
`baseline/providers.py` detects this specific 429 and stops immediately rather
than retrying something that cannot be waited out, keeping whatever has already
been answered. Use `provider: gemini` or `provider: ollama` to avoid the wall.

### Response caching

Every response is cached in `.cache/llm/responses.db` (sqlite, gitignored),
keyed on provider, model, prompt, `max_tokens`, and `temperature`. Re-running an
unchanged config costs zero budget and finishes in seconds, and an interrupted
sweep resumes for free. Pass `--no-cache` to force fresh calls.

## Retriever

**Placeholder lexical retriever** (`baseline/placeholder_retriever.py`) — a
<<<<<<< Updated upstream
stand-in until Task 2's BM25 retriever is merged. It scores each of a question's
own paragraphs by query-token overlap, normalised by the square root of
paragraph length so long paragraphs don't win on size alone, and returns the top
*k* by score descending.
=======
stand-in until Task 2's BM25 retriever is merged. It scores each of a
question's own paragraphs by query-token overlap, normalized by the square
root of paragraph length so long paragraphs don't win on size alone, and
returns the top *k* by score descending.
>>>>>>> Stashed changes

Retrieval is **closed per question**: only that question's own paragraphs are
searched — no cross-question or open-domain search. Paragraphs come from the
Task 1 loader (`src.data.musique_loader.load_split`), indexed once per process
<<<<<<< Updated upstream
behind an `lru_cache`, since `get_question` otherwise re-reads *and re-validates*
the whole split on every call (~0.8s), i.e. 300 times per run.

### Retrieval quality

Measured by `evaluation/retrieval_eval.py` over the same seeded 300-question
sample, with **no API calls** (`python scripts/run_retrieval_eval.py`, ~9s).
A MuSiQue question is only answerable if *every* supporting paragraph is
retrieved, so "all gold" is the practical ceiling on EM:

| k | recall | all gold retrieved | MRR |
| ---: | ---: | ---: | ---: |
| 3 | 21.1% | 2.3% | 0.342 |
| 5 | 30.9% | 6.3% | 0.375 |
| 10 | 50.9% | 18.7% | 0.407 |

Broken down by hop count, it degrades sharply with reasoning depth. 4-hop at
k=3 is structurally impossible — four gold paragraphs cannot fit in three slots:

| all gold retrieved | 2-hop | 3-hop | 4-hop |
| ---: | ---: | ---: | ---: |
| k=3 | 4.5% | 0.0% | 0.0% |
| k=5 | 11.5% | 1.0% | 0.0% |
| k=10 | 30.6% | 7.1% | 2.3% |

This is the headline weakness of the baseline and the direct motivation for
Task 2 (BM25) and the adaptive retrieval work after it.
=======
behind an `lru_cache`; the JSONL is never parsed here.

> `get_question` re-reads *and re-validates* the whole split on every call
> (~0.8s for dev), which over a 300-question run would mean reloading the
> split 300 times. The retriever therefore indexes the split once by question
> ID and looks up against that, preserving `get_question`'s behaviour
> including its `KeyError` on an unknown ID.

### Retrieval quality

Measured over the 300-question evaluation sample. A MuSiQue question is only
answerable if **every** supporting paragraph is retrieved, so the rightmost
column is the practical ceiling on EM:

| k | supporting-paragraph recall | all gold paragraphs retrieved |
| ---: | ---: | ---: |
| 3 | 26.5% | 7.7% |
| 5 | 35.8% | 11.7% |
| 10 | 55.5% | 30.3% |

This is the headline weakness of the baseline and the direct motivation for
Task 2 (BM25) and the adaptive retrieval work that follows.
>>>>>>> Stashed changes

### Swapping in BM25 (Task 2)

`baseline/retriever_interface.py` defines the contract both implementations
satisfy:

```python
def __call__(self, query: str, question_id: str, k: int) -> list[RetrievedParagraph]
```

`baseline/qa_pipeline.py` contains no retriever-specific logic. Integration once
Task 2 lands is exactly two edits:

1. Add `"bm25": bm25_retrieve` to `RETRIEVERS` in `scripts/run_baseline.py`
   (and in `scripts/run_retrieval_eval.py` to score it offline).
2. Change `retriever: placeholder` → `retriever: bm25` in the config.

The placeholder takes an extra `split: str = "dev"` keyword beyond the three
required arguments. The pipeline only ever passes the three required ones, so a
BM25 retriever that omits `split` is still compatible — but check its signature
before wiring it in.

## Prompt

Fixed across every run being compared — a prompt change is not a valid
"improvement" between k or retriever settings. Stored at
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

Each retrieved paragraph is rendered as `[title]` followed by its text.

## Evaluation

`evaluation/qa_eval.py` implements the standard MuSiQue/HotpotQA-style metrics
over the normalisation in `baseline/answer_extraction.py` (lowercase, strip
punctuation, drop `a`/`an`/`the`, collapse whitespace):

- **EM** — 1 if the normalised prediction equals the normalised gold answer *or
  any of its `answer_aliases`*, else 0.
- **F1** — token-level F1 against gold and each alias, taking the best score.

Both are reported overall and by hop count (2/3/4), where hop count is the
number of gold decomposition steps.

### Sampling

The dev split is **ordered by hop count**, so the plan's original
`records[:sample_size]` head slice returns an all-2-hop sample and makes the
required hop breakdown impossible. `scripts/run_baseline.py` draws a seeded
random sample instead (`seed` in the config) — reproducible and representative:

| sample | 2-hop | 3-hop | 4-hop |
| --- | ---: | ---: | ---: |
| 300 drawn with `seed: 13` | 157 | 99 | 44 |
| full dev split | 1,252 | 760 | 405 |

### Sampling

The dev split is **ordered by hop count**, so the plan's original
`records[:sample_size]` head slice returns an all-2-hop sample and makes the
required 2/3/4-hop breakdown impossible. `scripts/run_baseline.py` instead
draws a seeded random sample (`seed` in `configs/baseline.yaml`), which is
reproducible and representative:

| sample | 2-hop | 3-hop | 4-hop |
| --- | ---: | ---: | ---: |
| 300 drawn with `seed: 13` | 157 | 99 | 44 |
| full dev split | 1,252 | 760 | 405 |

## Running it

```bash
pip install -r requirements.txt
```

```bash
<<<<<<< Updated upstream
cp .env.example .env   # then fill in the key for your chosen provider
=======
cp .env.example .env   # then fill in GROQ_API_KEY
>>>>>>> Stashed changes
```

The MuSiQue-Ans splits are not versioned here — see `data/musique_ans/README.md`.

Full QA baseline:

```bash
python scripts/run_baseline.py --config configs/baseline.yaml
```

Retrieval only — no API calls, no key, ~9 seconds. This is the loop to use while
iterating on retrievers:

```bash
python scripts/run_retrieval_eval.py --config configs/baseline.yaml
```

<<<<<<< Updated upstream
Tests:

```bash
python -m pytest tests/ -v
```

Available configs: `baseline.yaml` (Groq, n=300, reported numbers),
`baseline_fast.yaml` (n=100, quick checks), `baseline_gemini.yaml` (Gemini,
n=300), `baseline_local.yaml` (Ollama, no quota).

## Results

Sample: 300 dev questions drawn with `seed: 13` (157/99/44 across 2/3/4-hop),
provider `groq`, model `openai/gpt-oss-120b`.

### k = 3 (complete)

| | n | EM | F1 |
| --- | ---: | ---: | ---: |
| **Overall** | 300 | **0.073** | **0.101** |
| 2-hop | 157 | 0.096 | 0.126 |
| 3-hop | 99 | 0.040 | 0.066 |
| 4-hop | 44 | 0.068 | 0.090 |

### k = 5 and k = 10 — not yet measured

Blocked on the 200k tokens/day limit above: k=5 needs 73% and k=10 needs 128% of
a full day's free budget, and the k=3 run had already consumed half of it. They
will be run on `provider: gemini` or `provider: ollama`, which do not have this
ceiling.

Reading the k=3 result: EM 0.073 against a **2.3% all-gold-retrieved ceiling**
means the reader is answering nearly everything it has actually been given the
evidence for, plus a little from parametric knowledge. The bottleneck is
squarely retrieval, not the reader — exactly the gap Task 2 onwards addresses.

## Known limitations

- The retriever is a lexical-overlap placeholder, not BM25. It has no IDF
  weighting (so `the` counts as much as `Bauhaus`) and ignores paragraph titles,
  which in MuSiQue are usually the entity the question names.
- Single-shot retrieval against the original question. A 2-hop question's second
  paragraph usually mentions an entity that appears only in the *first hop's
  answer*, so it is unreachable by lexical matching at any k. That gap is the
  project's actual subject.
- 300 of 2,417 dev questions, one run per configuration, with ~1 EM point of
  run-to-run noise even at `temperature=0`.
- Hop-wise cells are small — 44 questions at 4 hops — so those numbers carry
  meaningfully wider error bars than the overall figure.
=======
## Results

<!-- RESULTS -->

## Known limitations

- The retriever is a lexical-overlap placeholder, not BM25. It retrieves all
  supporting paragraphs for only 8–30% of questions depending on *k*, which
  caps EM well below what the model could otherwise reach. Every number above
  should be expected to move once Task 2 is merged.
- 300 of 2,417 dev questions, one run per configuration. With
  `temperature=0.0` this is nearly deterministic, but no variance across
  repeated runs is reported.
- Hop-wise cells are small — 44 questions at 4 hops — so per-hop numbers carry
  meaningfully wider error bars than the overall figure.
- The baseline answers from a single retrieval pass, so questions whose later
  hops depend on an intermediate answer are largely unreachable by
  construction. That gap is the project's actual subject.
>>>>>>> Stashed changes
