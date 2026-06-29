# OpenClaw Integration

Horizon Brief should run as its own MCP server when OpenClaw needs a briefing. OpenClaw calls Horizon tools, while Horizon performs fetching, scoring, enrichment, and summary generation in its own clean pipeline context.

Use the OpenClaw shared provider when Horizon should use the same LLM backend and credentials as OpenClaw:

```json
{
  "ai": {
    "provider": "openclaw",
    "model": "openclaw",
    "languages": ["ru"],
    "openclaw_config_path": "~/.openclaw/openclaw.json",
    "openclaw_env_path": "~/.openclaw/openclaw.env",
    "openclaw_agent_id": "spermwhale"
  }
}
```

`model: "openclaw"` tells Horizon to read the selected OpenClaw agent model from `openclaw_config_path`. Horizon reads `agents.defaults.model.primary`, per-agent `model.primary`, and `models.providers.<id>` metadata for OpenAI-compatible, Anthropic, and Gemini-style providers. You can also set `model` directly to an OpenClaw-style value such as `openai/gpt-5.5`, `anthropic/claude-sonnet-4.5`, or `ollama/gemma4-e2b-8k:latest`.

Example OpenClaw MCP server entry:

```json5
mcp: {
  servers: {
    "horizon-brief": {
      enabled: true,
      command: "/var/lib/openclaw/horizon/.venv/bin/horizon-mcp",
      cwd: "/var/lib/openclaw/horizon",
      connectionTimeoutMs: 30000,
      requestTimeoutMs: 900000,
      env: {
        HORIZON_PATH: "/var/lib/openclaw/horizon",
        HORIZON_OPENCLAW_CONFIG_PATH: "/etc/openclaw/openclaw.json5",
        HORIZON_OPENCLAW_ENV_PATH: "/etc/openclaw/openclaw.env",
        HORIZON_OPENCLAW_AGENT_ID: "spermwhale"
      }
    }
  }
}
```

Call `hz_run_pipeline` for a full briefing run. Use `hz_get_run_summary` to read a generated summary by `run_id` and language.

Do not route the briefing text through an OpenClaw agent prompt as a substitute for Horizon's model calls. That would mix OpenClaw's conversation context into the briefing and bypass Horizon's evidence-aware pipeline.