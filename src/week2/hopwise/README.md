# Week 2 — Task 2: Hop-wise Retrieval

## What it does

Retrieves evidence **once per reasoning hop** instead of once per question,
conditioning each hop's query on the previous hop's answer, and accumulates the
results into a single evidence set.

```text
Hop 1:  Q1                  -> retrieve(Q1)  -> {P2, P8}
Hop 2:  Q2 with A1 filled in -> retrieve(Q2) -> {P5, P7}
Hop 3:  Q3 with A2 filled in -> retrieve(Q3) -> {P13}

Accumulated evidence = {P2, P5, P7, P8, P13}
```

## Why it exists

Week 1 measured the ceiling that motivates the whole project: with one-shot BM25,
*all* gold supporting paragraphs reach the reader for only 38% of questions at
k=10. A single retrieval pass cannot reach evidence that only becomes findable
once an intermediate answer is known — the second hop of *"Who is the spouse of
the Green performer?"* is about a person whose name appears nowhere in the
question.

This task asks one question: **does hop-conditioned retrieval recover more gold
evidence than one-shot retrieval at the same retrieval budget?**

It uses the **gold** decomposition deliberately. That isolates the retrieval
contribution instead of mixing it with generated-decomposition errors, which is
what makes the result attributable (spec §10.1).

## How it works

1. **Build one query per hop.** Take the gold sub-question and replace MuSiQue's
   `#N` references with the gold answer of hop N. `#1 >> spouse` becomes
   `Steve Hillage >> spouse`.
2. **Retrieve per hop** with `k_hop` slots, using the existing BM25 retriever
   unchanged.
3. **Accumulate**, deduplicating while preserving hop structure in the trace.

### Substitution behaviour

Verified across all 2,417 dev records: `#N` references are always well-formed —
never in a first step, never forward or self-referential, never out of range —
so substitution is unambiguous.

| | count |
| --- | ---: |
| decomposition steps | 6,404 |
| first steps (no reference possible) | 2,417 |
| later steps with a `#N` reference | 3,636 |
| later steps **without** a reference | 351 |
| questions containing such a step | 351 (14.5%) |

Those 351 steps are **not implicit dependencies needing inference**. They are
genuinely independent hops that name their entity outright — e.g.
`Duane Courtney >> member of sports team`. They are retrieved verbatim, not
skipped (spec §12).

36% of all steps use the `A >> relation` form rather than natural language. No
rewriting is applied: BM25 tokenises on `[a-z0-9]+`, so `>>` is dropped before
scoring either way.

An unresolvable reference would be left in the query text rather than replaced
with an empty string. This does not occur on the released data.

## Retrieval budget

The comparison is meaningless unless both systems retrieve the same amount
(spec §14). Hop-wise spends `k_hop` slots at each of `hop_count` hops, so
one-shot is given exactly `hop_count × k_hop` in a single call — a budget that
varies per question.

Two things worth noting when reading the tables:

- **Hop-wise is handicapped, not favoured.** Hops overlap, so deduplication
  leaves it with *fewer unique paragraphs* than its slot budget allows (4.6 vs
  5.3 at `k_hop=2`). It wins despite seeing less.
- **No ceiling.** One-shot recall runs 40–64% at these budgets, far from
  saturation, so the gains are not an artefact of large `k` trivially returning
  the whole 20-paragraph context (spec §17).

## Results

Full dev split, 2,417 questions, BM25. `R` = recall, `AG` = all gold retrieved.

| k_hop | one-shot R | hop-wise R | ΔR | one-shot AG | hop-wise AG | ΔAG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40.0% | **68.7%** | +28.7 | 5.3% | **35.4%** | +30.0 |
| 2 | 54.6% | **86.9%** | +32.3 | 17.7% | **69.8%** | +52.1 |
| 3 | 64.4% | **92.2%** | +27.8 | 30.0% | **81.4%** | +51.4 |

MRR also rises (0.696 → 0.812 at `k_hop=2`), so gold paragraphs rank higher, not
merely appear more often.

The practical reading: at `k_hop=3`, hop-wise assembles the complete supporting
set for **81.4%** of questions where one-shot manages **30.0%** on the same
budget. Since a MuSiQue question is unanswerable without every supporting
paragraph, that is the number that bounds any downstream reader.

### By hop count (full dev)

| k_hop | hops | one-shot R | hop-wise R | ΔR | one-shot AG | hop-wise AG | ΔAG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2 | 52.0% | 86.3% | +34.3 | 22.4% | 75.5% | +53.0 |
| 2 | 3 | 58.5% | 87.0% | +28.5 | 14.6% | 65.4% | +50.8 |
| 2 | 4 | 55.4% | 88.8% | +33.5 | 8.6% | 60.2% | +51.6 |
| 3 | 2 | 60.5% | 92.0% | +31.4 | 32.6% | 85.1% | +52.5 |
| 3 | 3 | 69.0% | 91.2% | +22.2 | 29.5% | 75.4% | +45.9 |
| 3 | 4 | 67.9% | 94.9% | +27.0 | 23.0% | 81.5% | +58.5 |

**On the depth hypothesis.** The project expected hop-wise retrieval to help
most as reasoning depth increases. The data does not support a clean version of
that claim: gains are large at every depth but show **no consistent monotonic
trend**, and the ordering flips between budgets (4-hop has the smallest all-gold
gain at `k_hop=1` and the largest at `k_hop=3`). What does hold is that deeper
questions remain harder in absolute terms, and that one-shot's all-gold rate
collapses with depth (22.4% → 14.6% → 8.6% at `k_hop=2`) while hop-wise degrades
far more gently (75.5% → 65.4% → 60.2%).

The seeded 300-question sample used by the Week 1 baseline is evaluated
alongside and agrees closely — e.g. 87.3% vs 86.9% hop-wise recall at `k_hop=2`.

## How to run

```bash
python scripts/run_hopwise_retrieval.py --config configs/hopwise.yaml
```

No API key and no LLM calls. About 90 seconds for both samples and all three
`k_hop` values. `--sample seeded` scores only the Week 1 comparability sample
(about 8 seconds).

**Inputs:** MuSiQue-Ans dev split (see `data/musique_ans/README.md`) and
`configs/hopwise.yaml` (split, retriever, `k_hop` sweep, seed, sample size).

**Outputs:** `reports/week2/hopwise_results.json` — per sample, per `k_hop`,
per hop count, both conditions plus deltas, and the substitution counts above.

## Evidence trace

Every run produces a trace matching the shared Week 2 interface in
`src/week2/evidence.py`, which Task 3 consumes:

```python
[
    EvidenceStep(hop=1, query="Green >> performer", retrieved_paragraphs=[...]),
    EvidenceStep(hop=2, query="Steve Hillage >> spouse", retrieved_paragraphs=[...]),
]
```

`accumulated_indices(trace, through_hop=n)` returns the deduplicated evidence
available after hop *n* — the prefix a stopping rule inspects when deciding
whether to continue. It lives outside `hopwise/` so the stopping work does not
depend on this module.

## Evaluation

Scoring reuses `score_one` from `evaluation/retrieval_eval.py` rather than
restating the definitions, so Week 1 and Week 2 numbers come from one
implementation (spec §15):

- **Recall** — fraction of gold supporting paragraphs retrieved
- **All gold** — fraction of questions where *every* gold paragraph is retrieved
- **MRR** — reciprocal rank of the first gold paragraph

Reported overall and split by 2/3/4-hop.

## Limitations

- **These are oracle numbers.** Gold sub-questions *and* gold previous-hop
  answers. They upper-bound what generated decomposition can deliver; they are
  not an end-to-end system result. Task 1's generated decompositions will score
  lower, and the gap between them is the interesting future measurement.
- **Retrieval only.** No answer generation, so this says nothing directly about
  EM/F1. It raises the ceiling those metrics are bounded by; whether the reader
  converts that into accuracy is a Week 3 question.
- **Closed corpus.** 20 candidate paragraphs per question, no open-domain
  retrieval.
- **One retriever.** BM25 only; a dense retriever might narrow the gap.
- Deduplication means hop-wise uses fewer unique paragraphs than its slot
  budget. Reported explicitly rather than corrected for, since it makes the
  comparison conservative.

## Research framing

This is an **iterative retrieval strategy inspired by the reasoning-and-retrieval
interaction studied by IRCoT**, with reasoning represented explicitly as ordered
sub-questions. It is *not* an implementation of IRCoT, which interleaves
chain-of-thought sentence generation with retrieval and queries on the latest
generated sentence (spec §2.1).
