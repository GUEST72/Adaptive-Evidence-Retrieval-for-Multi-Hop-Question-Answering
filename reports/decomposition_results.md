# Decomposition live-run results

## Verified run

The following live run was completed successfully on 2026-09-08:

```powershell
python scripts/run_decomposition.py `
  --live `
  --model qwen/qwen3.8-27b `
  --max-examples 5 `
  --seed 13 `
  --report reports/live-verified-5.json
```

The run loaded the local MuSiQue train/dev data and local Groq credentials
without printing or storing the credentials.

| Field | Result |
|---|---:|
| Split | `dev` |
| Examples | 5 |
| Training examples per prompt | 3 |
| Model | `qwen/qwen3.8-27b` |
| Live generation | Yes |
| Successful generations | 5 |
| Failed generations | 0 |
| Hop-count accuracy | 1.000 |
| Overall ROUGE-1 | 0.773 |
| Overall ROUGE-L | 0.721 |
| Overall token F1 | 0.773 |

All five responses were parsed into valid 2-4 step decompositions. The run
included four 2-hop questions and one 3-hop question.

## Example generated decompositions

### Two-hop question

```text
Question: Where was the person who acted in the film Sous les pieds des femmes born?

1. Who acted in the film Sous les pieds des femmes?
2. Where was the #1 born?
```

### Three-hop question

```text
Question: Who is the president of the newly declared independent country that
is part of the Commission of Truth and Friendship with the country where
Sumardi was born?

1. Sumardi >> country of birth
2. Which newly declared independent country is part of the Commission of Truth
   and Friendship with #1?
3. Who is the president of #2?
```

## Validation

The complete automated test suite passed:

```text
119 passed
```

Run it from the repository root with:

```powershell
python -m pytest -q
```

For setup, key loading, model selection, troubleshooting, and rerunning the
experiment, see [`src/llm/README.md`](../src/llm/README.md) and
[`src/decomposition/README.md`](../src/decomposition/README.md).
