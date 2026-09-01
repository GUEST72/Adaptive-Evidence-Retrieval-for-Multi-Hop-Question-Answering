# Task 2 -- BM25 Retrieval

Retrieval-only baseline: given a MuSiQue-Ans question and its own 20
candidate paragraphs, can plain BM25 keyword search surface the correct
supporting evidence? No LLM or answer generation happens here.

## Scope (Week 1)

Deliberately limited, per the project plan:

- **Closed / per-question retrieval**: the candidate pool for a question is
  always exactly its own 20 paragraphs, never an open corpus. We are not
  indexing "all of Wikipedia" this week.
- **Query = raw question only.** No decomposition, no sub-question
  generation, no iterative or hop-wise retrieval, no adaptive stopping.
  Those are explicitly next-stage experiments.
- **Plain `rank-bm25`** (`BM25Okapi`), no Elasticsearch / vector DB.

## How the index is built

For each question, at retrieval time:

1. Load that question's record via Task 1's loader
   (`src.data.musique_loader.load_split` / an internal id-indexed cache --
   see "Performance note" below).
2. Tokenize each of the 20 documents as `title + " " + paragraph_text`.
   Wikipedia titles often hold the entity name the question mentions, while
   the body may never repeat it.
3. Build a fresh `BM25Okapi` index over just those 20 tokenized documents.
4. Tokenize the query the same way and score it against the index.
5. Return the top-k paragraphs by descending BM25 score.

Because each question's corpus is only 20 documents, building a fresh index
per question is cheap and keeps retrieval correctly *scoped* to that
question -- there's no risk of accidentally leaking terms/statistics across
different questions' paragraph pools.

### Tokenization

Deliberately the simplest thing that works, per the project brief
("start with the simplest working implementation"):

```python
_TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())
```

Lowercase, then split into alphanumeric runs. No stemming, no stopword
removal, no lemmatization. The same function is used for both paragraph
text and the query, so scoring is consistent. This is a known limitation
(see Results / Limitations below) and an obvious next lever to pull.

### Performance note

`src.data.musique_loader.get_question` re-reads and re-validates the whole
split on every call, which is fine for one-off lookups but far too slow for
evaluating ~2,400 dev questions one at a time. `bm25_retriever.py` instead
loads a split once per process (`functools.lru_cache`) and indexes it by
question id, so the expensive read/validate step happens exactly once
regardless of how many questions are retrieved.

## Function interface

```python
from src.retrieval.bm25_retriever import retrieve

retrieve(query: str, question_id: str, k: int,
         split: str = "dev", data_dir: str | Path | None = None,
         validate: bool = False) -> list[dict]
```

Returns up to `k` dicts, ordered by descending BM25 score:

```python
{
    "idx": int,             # paragraph index, 0-19
    "title": str,
    "text": str,            # paragraph_text
    "score": float,         # raw BM25 score
    "is_supporting": bool,  # ground truth, included for debugging only
}
```

This is the fixed interface Task 3 imports and calls directly -- no
copy-pasting retrieval logic into the QA pipeline.

## Evaluation

`src/evaluation/retrieval_eval.py` measures the retriever in isolation
(no LLM):

```bash
python -m src.evaluation.retrieval_eval --split dev --k 1 2 3 5 10
python -m src.evaluation.retrieval_eval --data-dir data/musique_ans
```

Metrics, computed per question and macro-averaged (each question counted
equally regardless of how many gold paragraphs it has):

- **Recall@k** -- fraction of a question's gold supporting paragraphs found
  in its top-k retrieved paragraphs, averaged across all questions.
- **Average gold paragraphs retrieved** -- raw count (not normalized),
  averaged across questions.
- **Full-support coverage rate** -- fraction of questions where *all* gold
  supporting paragraphs were retrieved (Recall@k == 1.0). This is the
  number that matters most downstream: an LLM given partial evidence for a
  multi-hop question generally can't reconstruct the missing hop.

A hop-wise (2/3/4-hop) breakdown of the same metrics is also printed, since
it's essentially free to compute here and useful context before Task 3's
own hop-wise EM/F1 breakdown.

### k values

Per the brief, evaluated at `k = 1, 2, 3, 5, 10`.

- Small k -> less noise/context, but higher risk of missing evidence.
- Large k -> better coverage, but more distractor noise and larger prompts
  downstream in Task 3.

## Results

Measured on the official MuSiQue-Ans **dev** split (2,417 questions) with
query = the original question and BM25 over `title + paragraph_text`.

The loader's strict "exactly 20 paragraphs" check currently fails on a
handful of official records (some have 18–19 paragraphs). Evaluation
therefore loads with `validate=False` so the full split is scored; pass
`--validate` if you want the Task 1 check to abort instead.

| k | Recall@k | Avg gold retrieved | Full-support coverage |
|---|----------|---------------------|------------------------|
| 1 | 0.230 | 0.571 | 0.000 |
| 2 | 0.350 | 0.886 | 0.045 |
| 3 | 0.428 | 1.093 | 0.092 |
| 5 | 0.532 | 1.378 | 0.179 |
| 10 | 0.702 | 1.835 | 0.387 |

Hop-wise Recall@k:

| Hop | k=1 | k=2 | k=3 | k=5 | k=10 |
|-----|-----|-----|-----|-----|------|
| 2-hop | 0.273 | 0.393 | 0.468 | 0.561 | 0.723 |
| 3-hop | 0.207 | 0.338 | 0.423 | 0.539 | 0.713 |
| 4-hop | 0.139 | 0.238 | 0.312 | 0.430 | 0.615 |

Hop-wise full-support coverage:

| Hop | k=1 | k=2 | k=3 | k=5 | k=10 |
|-----|-----|-----|-----|-----|------|
| 2-hop | 0.000 | 0.087 | 0.164 | 0.274 | 0.500 |
| 3-hop | 0.000 | 0.000 | 0.024 | 0.108 | 0.330 |
| 4-hop | 0.000 | 0.000 | 0.000 | 0.017 | 0.146 |

Read as: even at k=10, only half of 2-hop questions get *every* gold
paragraph, and 4-hop full coverage is 14.6%. That is the Week 1 retrieval
ceiling Task 3 will inherit, and the gap later hops leave is what
decomposition / iterative retrieval is meant to close.

## Known limitations

- Tokenizer is ASCII-only (`[a-z0-9]+`), so accented names are split or dropped.
- No stemming/lemmatization: a question asking about "invented" won't match
  a paragraph saying "invention" or "inventor" via shared tokens.
- No stopword removal: common words contribute to BM25's IDF weighting
  exactly like content words.
- Later hops are structurally hard for single-shot BM25 against the raw
  question: a hop-2 paragraph is often only relevant *given* hop-1's
  answer, and may share no vocabulary with the original question at all
  (see the "Green -> Steve Hillage -> spouse" example discussed in the
  project write-up). This is an expected, real limitation of Week 1's
  fixed-query design, not a bug -- it's exactly what motivates later
  decomposition/iterative-retrieval experiments.

## Running the tests

```bash
python -m pytest tests/test_bm25_retriever.py -v
```

Tests use a small synthetic in-memory-generated split (written to a temp
directory) rather than the real dataset, so they run without requiring the
MuSiQue download and stay fast.
