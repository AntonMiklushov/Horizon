# Codex CLI Provider

Horizon can use a local Codex CLI login as an AI provider. This path runs `codex exec` as a subprocess for each logical AI stage and does not use OpenAI API keys.

## Install

```powershell
npm i -g @openai/codex
```

## Login

```powershell
codex login
```

## Verify

```powershell
codex login status
codex exec "Ответь одним словом: OK"
```

## Horizon Config

Set the AI provider to `codex_cli`:

```json
{
  "ai": {
    "provider": "codex_cli",
    "model": "codex-cli",
    "languages": ["ru"],
    "codex_command": "codex",
    "codex_timeout_sec": 180,
    "codex_extra_args": [
      "--skip-git-repo-check",
      "-c",
      "model_reasoning_effort=\"medium\""
    ],
    "codex_use_output_last_message": true,
    "codex_use_json": false
  }
}
```

No `OPENAI_API_KEY` is required for this provider. Codex CLI uses the local authentication session created by `codex login`.

Horizon automatically requests medium reasoning for Codex CLI calls when the installed CLI supports `-c/--config`. The explicit `codex_extra_args` above are kept in the example for transparency and for generated configs; Horizon de-duplicates managed flags so `--skip-git-repo-check` and `model_reasoning_effort` are not passed twice.

## First Run

```powershell
.\.venv\Scripts\horizon.exe --hours 6
```

## Caveats

- Slower than direct API providers.
- Content scoring is batched for Codex CLI to avoid spawning one process per item. Other logical stages may still make separate Codex calls when they need different prompts or per-item context.
- Requires a valid local Codex login session.
- `--output-last-message` is preferred when supported by the local CLI.
- Not intended for high-frequency automation.
- JSONL mode is available through `codex_use_json`, but the default v1 path is artifact/stdout completion capture.
