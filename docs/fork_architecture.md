# Fork Architecture

This fork keeps the public Horizon name and command surface for now, while
isolating fork-only product additions under `src.horizon_ext`.

## Goals

- Keep upstream rebases easier by minimizing broad edits to upstream-style core files.
- Preserve user-facing commands such as `horizon`, `horizon-mcp`, `horizon-webhook`, and `horizon-web`.
- Make a future project rename easier by keeping new product surfaces behind one internal namespace.

## Namespace Rules

- Put new fork-only product features in `src.horizon_ext`.
- Keep `src.main`, `src.orchestrator`, `src.models`, scrapers, services, and MCP public modules as integration points or compatibility shims where practical.
- Keep old import paths working when existing users or tests already depend on them.
- Do not move local runtime state into tracked files. `data/config.json`, run artifacts, summaries, backups, and secret-like files stay ignored.

## Current Extension Areas

- `src.horizon_ext.web`: local-only browser dashboard.
- `src.horizon_ext.rendering`: digest rendering templates and channel renderers.
- `src.horizon_ext.personal`: evidence-aware personal briefing mode.
- `src.horizon_ext.pipeline`: shared fork pipeline contracts and source-quality helpers.
- `src.horizon_ext.mcp`: local-only MCP/run safety helpers.

## Rename Note

A future rename is intentional but out of scope for the current cleanup. Until
then, keep external package names, CLI commands, docs, and config examples using
`Horizon` unless a rename is explicitly requested.

