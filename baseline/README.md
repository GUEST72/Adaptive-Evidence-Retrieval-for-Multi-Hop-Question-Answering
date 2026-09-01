# Task 3 — Naive Retrieve-then-Answer Baseline

The first end-to-end AdaptiveHop baseline:

```
Question → Retrieve once → Top-k paragraphs → LLM → Final answer → EM/F1
```

One retrieval call and one LLM call per question. No decomposition, no
iterative retrieval, no adaptive stopping, no agent loop — those are later
tasks.

## Model and provider

The backend is a config choice (`provider:` in `configs/*.yaml`), not a code
change — `qa_pipeline.py` only ever calls `call_llm`. See
`baseline/providers.py`.

| provider | model | why |
| --- | --- | --- |
| `hf_local` | `Qwen/Qwen2.5-7B-Instruct` | **produced the reported results**; free Colab/Kaggle GPU, no per-token quota |
| `groq` | `openai/gpt-oss-120b` | fast and strong, but only 200k tokens/day free |
| `gemini` | `gemini-3.5-flash` | 20 requests/day/model free — spot checks only |
| `ollama` | `qwen2.5:7b-instruct` | local, no quota, needs a capable machine |

`hf_local` is registered at runtime by `notebooks/run_eval_gpu.ipynb` via
`providers.register_provider`, so `torch` never becomes a dependency of the
local install. The other three live in `baseline/providers.py`.

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
been answered. The reported sweep was run instead on a free GPU (see below), which has no
per-token ceiling.

### Response caching

Every response is cached in `.cache/llm/responses.db` (sqlite, gitignored),
keyed on provider, model, prompt, `max_tokens`, and `temperature`. Re-running an
unchanged config costs zero budget and finishes in seconds, and an interrupted
sweep resumes for free. Pass `--no-cache` to force fresh calls.

## Retriever

**BM25** (`src/retrieval/bm25_retriever.py`, Task 2) is the retriever in use.
It indexes `title + paragraph_text` for each of a question's own paragraphs and
ranks them with Okapi BM25.

The **placeholder lexical retriever** (`baseline/placeholder_retriever.py`) is
kept as the pre-Task-2 comparison point: query-token overlap normalised by the
square root of paragraph length, scoring the body only.

Retrieval is **closed per question**: only that question's own paragraphs are
searched — no cross-question or open-domain search. Paragraphs come from the
Task 1 loader (`src.data.musique_loader.load_split`), indexed once per process
behind an `lru_cache`, since `get_question` otherwise re-reads *and re-validates*
the whole split on every call (~0.8s), i.e. 300 times per run.

### Retrieval quality

Measured by `evaluation/retrieval_eval.py` over the same seeded 300-question
sample, with **no API calls** (`python scripts/run_retrieval_eval.py`, ~9s).
A MuSiQue question is only answerable if *every* supporting paragraph is
retrieved, so "all gold" is the practical ceiling on EM:

**BM25** (`retriever: bm25`, Task 2 — the retriever now in use):

| k | recall | all gold retrieved | MRR |
| ---: | ---: | ---: | ---: |
| 3 | 41.2% | 8.7% | 0.656 |
| 5 | 53.4% | 17.3% | 0.681 |
| 10 | 69.1% | 38.0% | 0.689 |

**Placeholder** (lexical overlap, superseded — kept for comparison):

| k | recall | all gold retrieved | MRR |
| ---: | ---: | ---: | ---: |
| 3 | 21.1% | 2.3% | 0.342 |
| 5 | 30.9% | 6.3% | 0.375 |
| 10 | 50.9% | 18.7% | 0.407 |

BM25 roughly doubles recall at every k and lifts the EM ceiling at k=10 from
18.7% to 38.0%. MRR nearly doubles, so gold paragraphs also rank higher, not
just appear more often. Much of the gain comes from BM25 indexing
`title + paragraph_text` where the placeholder scored the body alone — MuSiQue
titles are entity names the question usually mentions.

Broken down by hop count, it degrades sharply with reasoning depth. 4-hop at
k=3 is structurally impossible — four gold paragraphs cannot fit in three slots:

BM25, all gold retrieved by hop count:

| | 2-hop | 3-hop | 4-hop |
| ---: | ---: | ---: | ---: |
| k=3 | 15.3% | 2.0% | 0.0% |
| k=5 | 29.3% | 5.1% | 2.3% |
| k=10 | 49.7% | 25.3% | 25.0% |

This is the headline weakness of the baseline and the direct motivation for
Task 2 (BM25) and the adaptive retrieval work after it.

### Swapping in BM25 (Task 2)

`baseline/retriever_interface.py` defines the contract both implementations
satisfy:

```python
def __call__(self, query: str, question_id: str, k: int) -> list[RetrievedParagraph]
```

`baseline/qa_pipeline.py` contains no retriever-specific logic. Integration once
Task 2 lands is exactly two edits:

1. Add `"bm25": bm25_retrieve` to `RETRIEVERS` in `baseline/retrievers.py`.
2. Change `retriever: placeholder` → `retriever: bm25` in the config.

`baseline/retrievers.py` is the single registry both entrypoints and the GPU
notebook resolve through. `run_baseline` takes the retriever from the config and
refuses a caller that passes a different function than the config names — the
filename and provenance are derived from the config, so a mismatch produces
results labelled as a retriever that never ran. That is not hypothetical: it
happened once, and the cache made every call a hit, so a full "BM25" sweep
returned byte-identical placeholder output in seconds.

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

The template file is chosen by `prompt_path` in the config, resolved relative to
the repository root. Point it at a different file to try a prompt variant —
but keep it fixed across any set of runs being compared.

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

## Running it

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env   # then fill in the key for your chosen provider
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

Tests:

```bash
python -m pytest tests/ -v
```

Available configs: `baseline.yaml` (Groq, n=300, reported numbers),
`baseline_fast.yaml` (n=100, quick checks), `baseline_gpu.yaml` (GPU notebook),
`baseline_gemini.yaml` (Gemini — 20 requests/day, spot checks only),
`baseline_local.yaml` (Ollama).

### Running the sweep on a free GPU

Hosted free tiers cannot finish this sweep: Groq allows 200k tokens/day (k=10
at n=300 alone needs ~299k) and Gemini allows 20 requests/day per model. A free
Colab or Kaggle T4 lifts that ceiling and runs all three k values in about an
hour, repeatable whenever the retriever changes.

Hugging Face acts only as the weights host: the notebook downloads the model
once and generates locally on the GPU, so no inference API is involved and
there is no per-token or per-request quota. The constraint becomes GPU time —
roughly 30 GPU-hours/week on Kaggle against ~1 hour per full sweep, versus
Groq's 200k tokens/day that makes k=10 at n=300 impossible outright. Colab's
free GPU has no published quota but is allocated dynamically and can be
refused, so Kaggle is the more predictable of the two.

Open `notebooks/run_eval_gpu.ipynb` in Colab or Kaggle, enable the GPU, and run
it top to bottom. It loads an open model through `transformers` and registers it
with `providers.register_provider('hf_local', ...)`, so the pipeline is
unchanged and `torch` never becomes a dependency of the local install. The run
loop lives in `baseline/runner.py` precisely so the notebook can call it
in-process — a subprocess would not see a provider registered at runtime.

## Results

Reader held constant across every row: **`Qwen/Qwen2.5-7B-Instruct`** in 4-bit on
a Colab T4, via `notebooks/run_eval_gpu.ipynb`. Same 300 dev questions for every
run, drawn with `seed: 13` (157/99/44 across 2/3/4-hop).

### BM25 vs placeholder

| k | placeholder EM | BM25 EM | ΔEM | placeholder F1 | BM25 F1 | ΔF1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 0.047 | **0.107** | +0.060 | 0.065 | **0.137** | +0.072 |
| 5 | 0.063 | **0.127** | +0.063 | 0.093 | **0.169** | +0.077 |
| 10 | 0.113 | **0.157** | +0.043 | 0.159 | **0.221** | +0.062 |

Swapping the retriever alone — same reader, same sample, same prompt — more than
doubles EM at k=3 and k=5. The gain is largest at small k, where retrieval was
the binding constraint; by k=10 the placeholder had already caught up somewhat.

### BM25 by hop count

| k | hops | n | EM | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 3 | all | 300 | 0.107 | 0.137 |
| | 2 | 157 | 0.134 | 0.167 |
| | 3 | 99 | 0.061 | 0.094 |
| | 4 | 44 | 0.114 | 0.127 |
| 5 | all | 300 | 0.127 | 0.169 |
| | 2 | 157 | 0.166 | 0.212 |
| | 3 | 99 | 0.091 | 0.134 |
| | 4 | 44 | 0.068 | 0.095 |
| 10 | all | 300 | **0.157** | **0.221** |
| | 2 | 157 | 0.217 | 0.282 |
| | 3 | 99 | 0.091 | 0.156 |
| | 4 | 44 | 0.091 | 0.149 |

The model abstains rather than guesses, and does so less as evidence improves:
`unknown` falls 67.3% → 62.0% → 48.3% across k=3/5/10 with BM25, against
83.7% → 74.7% → 65.3% with the placeholder.

### Answers are now better grounded

Counting correct answers that arrived *without* all gold paragraphs retrieved —
i.e. supplied from the model's own knowledge despite the prompt restricting it
to the evidence:

| retriever | k | correct | all gold retrieved | ungrounded |
| --- | ---: | ---: | ---: | ---: |
| placeholder | 3 | 14 | 3 | 11 (79%) |
| placeholder | 5 | 19 | 6 | 13 (68%) |
| placeholder | 10 | 34 | 14 | 20 (59%) |
| bm25 | 3 | 32 | 12 | 20 (62%) |
| bm25 | 5 | 38 | 19 | 19 (50%) |
| bm25 | 10 | 47 | 38 | **9 (19%)** |

This is the most encouraging number here. With the placeholder at k=10, 59% of
right answers were unsupported by the retrieved evidence; with BM25 that falls to
19%. The pipeline is not just scoring higher, it is scoring higher *for the right
reason* — and the remaining EM headroom is now genuinely about retrieval and
reading rather than recall.

### Earlier `gpt-oss-120b` runs (not comparable)

A k=3 placeholder sweep on `provider: groq` with `openai/gpt-oss-120b`, same seed
and sample, scored EM 0.073 / F1 0.101 — above Qwen's 0.047, as expected from a
far larger reasoning model. Different reader, so it is not in the tables above.
That run also scored EM 0.063 on a previous day from an identical config, which
is the noise floor to keep in mind when reading small differences.

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
