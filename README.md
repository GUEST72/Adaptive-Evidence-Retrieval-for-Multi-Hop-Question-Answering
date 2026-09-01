# AdaptiveHop

AdaptiveHop is a research-oriented NLP project on multi-hop question answering.
It investigates whether explicit question decomposition, iterative evidence
retrieval, and adaptive hop selection can improve answer accuracy, evidence
grounding, and retrieval efficiency over a conventional retrieve-then-answer
baseline.

## MuSiQue-Ans dataset foundation

The current implementation covers MuSiQue-Ans loading/validation/EDA (Task 1),
a closed BM25 evidence retriever (Task 2), and an end-to-end retrieve-then-answer
QA baseline with EM/F1 scoring (Task 3). Decomposition, iterative retrieval, and
adaptive hop selection are not implemented yet — they are the point of the
project, and the baseline below is what they have to beat.

### Environment

Use Python 3.11 or later. Create a local environment and install dependencies:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Dataset placement

Obtain MuSiQue-Ans from the [official MuSiQue repository](https://github.com/StonyBrookNLP/musique)
and place the supplied files at:

```text
data/musique_ans/train.jsonl
data/musique_ans/dev.jsonl
```

Dataset files are ignored by Git. See [the dataset README](data/musique_ans/README.md)
for expected split counts, provenance, and licensing guidance.

### Validation and EDA

After the dataset is in place, validate both splits with:

```powershell
python scripts/validate_musique.py
```

The EDA notebook is `notebooks/01_dataset_eda.ipynb`; launch it after validation:

```powershell
jupyter notebook notebooks/01_dataset_eda.ipynb
```

The loader, validator, tests, and notebook are added after the source data is
available so their schema assumptions can be checked against the actual release.

## BM25 retrieval (Task 2)

Task 2 is **retrieval only**: given a question and its own 20 candidate
paragraphs, can BM25 rank the supporting evidence into the top-k? There is
no LLM, no answer generation, and no open-domain Wikipedia index.

The Week 1 design is deliberately closed and single-shot:

```text
Question  →  that question's 20 paragraphs  →  BM25  →  Top-k
```

The query is the **original question text**. There is no decomposition,
sub-question generation, iterative retrieval, hop-wise retrieval, or
adaptive stopping — those are later experiments. This module is the
retrieval baseline Task 3 should call, not copy.

### Index and tokenization

For each question, at retrieval time:

1. Load the record through Task 1's loader (`src.data.musique_loader`).
2. Tokenize each of the 20 documents as `title + " " + paragraph_text`.
   Wikipedia titles often hold the entity name the question mentions,
   while the body may never repeat it.
3. Build a fresh `rank-bm25` `BM25Okapi` index over just those 20
   documents (cheap, and scoped so IDF statistics never leak across
   questions).
4. Tokenize the query the same way and return the top-k paragraphs by
   descending BM25 score.

Tokenization is lowercase alphanumeric splits (`[a-z0-9]+`). There is no
stemming, stopword removal, or lemmatization.

### Interface

```python
from src.retrieval.bm25_retriever import retrieve

retrieve(query, question_id, k)
```

Each returned item includes paragraph `idx`, `title`, `text`, BM25
`score`, and `is_supporting` (gold label, for debugging only). Optional
kwargs: `split` (default `"dev"`), `data_dir`, `validate`. Official
MuSiQue-Ans has a handful of records with 16–19 paragraphs (16 in dev, 21 in
train). The loader originally rejected these, so `validate` defaults to `False`
here; the loader has since been corrected to accept short contexts while still
requiring paragraph indices to be exactly `0..n-1`, so `validate=True` now works
on the real release too.

### Retrieval evaluation

This measures whether the retriever found evidence, not whether an LLM
can answer. Metrics are computed per question and **macro-averaged**:

- **Recall@k** — fraction of gold supporting paragraphs in the top-k
- **Average gold paragraphs retrieved** — raw count, not normalized
- **Full-support coverage** — fraction of questions where *all* gold
  supporting paragraphs appear in the top-k (`Recall@k == 1.0`)

k values: `1, 2, 3, 5, 10`. A hop-wise (2/3/4-hop) breakdown is printed
as well.

```powershell
python -m src.evaluation.retrieval_eval --split dev --k 1 2 3 5 10
python -m pytest tests/test_bm25_retriever.py -v
```

Retriever tests use a small synthetic split, so they do not require the
MuSiQue download.

### Dev results (MuSiQue-Ans, 2,417 questions)

Query = original question. Corpus = `title + paragraph_text`.

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

Even at k=10, only half of 2-hop questions retrieve *every* gold
paragraph, and 4-hop full coverage is 14.6%. That is the Week 1
retrieval ceiling the QA baseline inherits, and the gap later hops leave
is what decomposition and iterative retrieval are meant to close.

More detail (limitations, loader caching, return schema):
[src/retrieval/README.md](src/retrieval/README.md).

## Retrieve-then-answer baseline (Task 3)

The first end-to-end pipeline:

```text
Question -> Retrieve once -> Top-k paragraphs -> LLM -> Final answer -> EM/F1
```

A single retrieval call and a single LLM call per question. No decomposition, no
iteration, no adaptive stopping — deliberately, so later work has an honest
floor to improve on.

```powershell
python scripts/run_baseline.py --config configs/baseline.yaml
python scripts/run_retrieval_eval.py --config configs/baseline.yaml
python -m pytest tests/ -q
```

The retriever is a config choice (`retriever:` in `configs/*.yaml`) resolved
through `baseline/retrievers.py`; the LLM backend is likewise `provider:`.
Swapping either requires no pipeline code changes.

### Results

300 dev questions sampled with `seed: 13` (157/99/44 across 2/3/4-hop), reader
held constant at `Qwen/Qwen2.5-7B-Instruct` (4-bit, Colab T4):

| k | placeholder EM | BM25 EM | placeholder F1 | BM25 F1 |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 0.047 | **0.107** | 0.065 | **0.137** |
| 5 | 0.063 | **0.127** | 0.093 | **0.169** |
| 10 | 0.113 | **0.157** | 0.159 | **0.221** |

Replacing the pre-Task-2 lexical placeholder with BM25 — same reader, same
sample, same prompt — more than doubles EM at k=3 and k=5.

The more informative number is grounding. Counting correct answers that arrived
*without* all gold paragraphs retrieved, i.e. recalled rather than read:

| retriever | k=3 | k=5 | k=10 |
| --- | ---: | ---: | ---: |
| placeholder | 79% | 68% | 59% |
| bm25 | 62% | 50% | **19%** |

BM25 does not just raise the score, it makes the score mean what it should.

Absolute EM stays low because single-shot retrieval caps it: even with BM25,
all gold paragraphs are retrieved for only 38% of questions at k=10, and 3-hop
and 4-hop sit at 0.091 EM against 0.217 for 2-hop. Closing that is what
decomposition and iterative retrieval are for.

Note the two retrieval evaluations are independent implementations —
`src/evaluation/retrieval_eval.py` (Task 2, full 2,417-question split) and
`evaluation/retrieval_eval.py` (Task 3, the seeded 300-question sample). They
agree closely: full-support coverage at k=10 is 0.387 and 0.380 respectively,
which is a useful cross-check rather than a redundancy.

Full detail — prompt, normalisation, provider budgets, response caching, run
provenance, and the GPU notebook: [baseline/README.md](baseline/README.md).
