# LLM integration guide

The `src.llm` package is the provider boundary used by decomposition and
future experiments:

```text
DecompositionGenerator -> LLMClient -> GroqClient -> KeyManager
```

## Configuration

The Groq client reads credentials only from these environment variables:

```text
GROQ_API_KEY
GROQ_API_KEY_2
```

The second key is optional. When both are configured, requests use round-robin
selection. A key is temporarily cooled down only after a classified rate-limit
or quota response. Authentication, malformed-request, server, and network
errors are surfaced instead of silently hiding the problem.

The repository never loads `.env` automatically. In PowerShell, load local
keys into the current process without printing them:

```powershell
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*(GROQ_API_KEY(?:_2)?)\s*=\s*(.*)\s*$') {
    [Environment]::SetEnvironmentVariable(
      $matches[1],
      $matches[2].Trim().Trim('"').Trim("'"),
      "Process"
    )
  }
}
```

Never commit `.env`, API keys, response fixtures containing secrets, or keys in
reports. `.env` and generated JSON reports are ignored by Git.

## Supported model

The verified model for the current implementation is:

```text
qwen/qwen3.8-27b
```

Model availability is account-dependent. To inspect models available to your
Groq account without exposing the key:

```powershell
$headers = @{ Authorization = "Bearer $env:GROQ_API_KEY" }
(Invoke-RestMethod `
  -Uri "https://api.groq.com/openai/v1/models" `
  -Headers $headers).data | Select-Object -ExpandProperty id
```

Pass an available model explicitly with `--model`.

## Running live decomposition

From the repository root, after the dataset is installed:

```powershell
python scripts/run_decomposition.py `
  --live `
  --model qwen/qwen3.8-27b `
  --split dev `
  --max-examples 5 `
  --seed 13 `
  --report reports/decomposition.json
```

The report contains prompts, parsed steps, metrics, and failure details. A
successful run has `live_generation: true` and `failure_count: 0`.

## Troubleshooting

- **No API key configured**: export `GROQ_API_KEY` in the current PowerShell
  process; setting a value in `.env` alone is not enough.
- **Model does not exist or is unavailable**: list account models and pass a
  currently available ID with `--model`.
- **All keys are rate limited**: wait for quota recovery or provide another
  configured key. The client intentionally does not retry forever.
- **JSON parsing failure**: inspect `generation_error_message`; the parser
  accepts plain JSON and common Markdown code-fence/preamble wrappers, but the
  model must still return 2-4 structured steps.

## Testing

Unit tests inject fake key mappings and HTTP clients. They never contact Groq:

```powershell
python -m pytest tests/test_key_manager.py tests/test_decomposition.py -v
```

The full project suite is:

```powershell
python -m pytest -q
```
