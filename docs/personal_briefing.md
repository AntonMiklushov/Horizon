# Personal Briefing Mode (Evidence-aware)

This mode adds a conservative editorial and evidence layer on top of Horizon's existing fetch, deduplicate, score, filter, enrich, and summarize pipeline.

## What it does

- Produces a compact Russian markdown briefing by default.
- Classifies source metadata before AI scoring, then excludes blocked, unclassified, and disallowed social sources from the LLM scoring path.
- Classifies claim type (`confirmed_fact`, `official_statement`, `party_claim`, `primary_statement`, etc.).
- Assigns `evidence_strength` and `confidence`.
- Allows an empty-day Russian briefing when nothing passes the significance and evidence threshold.

## Source tiers and restrictions

- `fact_layer`: wire-style factual sources such as Reuters and AP.
- `context_layer`: analysis/context sources such as FT and The Economist.
- `official_primary_source`: official pages such as `mos.ru`; for sensitive claims they are treated as official statements, not independent confirmation.
- `russian_institutional_frame`: Russian institutional/media sources such as Interfax, Kommersant, and RBC. For sensitive claims they provide frame/statement context, not independent confirmation.
- `science_primary_source`: journals and official science institutions.
- `science_preprint`: preprint servers such as arXiv, bioRxiv, medRxiv, and ChemRxiv. Preprints are labeled `not peer-reviewed` and cannot stay as high-confidence confirmed facts without additional evidence.
- `science_source_finder`: discovery platforms such as EurekAlert. These can surface early leads but require verification against a paper, journal, or institution before factual confirmation.
- `tech_primary_source`: primary technical release sources such as GitHub releases.
- `blocked_as_fact_source` and `unclassified`: excluded before LLM scoring in personal mode.
- Social sources are not factual confirmation. Allowed social actors may only enter as `primary_statement`; social posts can never remain `confirmed_fact`.

## Configuration

Set `personal_briefing` in local `data/config.json`:

- `enabled`
- `language` (`ru` default)
- `timezone` (`Europe/Paris` default)
- thresholds and critic-pass settings
- `source_policy_file` (sample: `data/config.personal-news.example.json`)
- `generate_standard_summaries` (`false` by default): when personal mode is enabled, generate only the Russian personal briefing unless you explicitly set this to `true`. When set to `true`, the personal language is included and the configured `ai.languages` standard summaries may also be generated.

Keep local secrets and source choices in `data/config.json`. Do not commit that file; commit only `data/config.example.json` and `data/config.personal-news.example.json`.

## Run

- `uv run horizon`
- `uv run horizon --hours 24`

### Windows PowerShell

When the checkout is inside a synced folder, keep uv's environment and cache outside
the repository:

```powershell
$horizonState = Join-Path $env:LOCALAPPDATA "Horizon"
$env:UV_PROJECT_ENVIRONMENT = Join-Path $horizonState "uv-env"
$env:UV_CACHE_DIR = Join-Path $horizonState "uv-cache"
New-Item -ItemType Directory -Force $horizonState | Out-Null
uv sync
```

- `& "$env:UV_PROJECT_ENVIRONMENT\Scripts\horizon.exe"`
- `& "$env:UV_PROJECT_ENVIRONMENT\Scripts\horizon.exe" --hours 24`

If your terminal is not UTF-8, set it before running Russian output:

```powershell
$env:PYTHONIOENCODING = "utf-8"
chcp 65001
```

For low-churn local QA on locked-down Windows machines:

```powershell
$horizonState = Join-Path $env:LOCALAPPDATA "Horizon"
$horizonTemp = Join-Path $horizonState "tmp"
$env:UV_PROJECT_ENVIRONMENT = Join-Path $horizonState "uv-env"
$env:UV_CACHE_DIR = Join-Path $horizonState "uv-cache"
$env:TMP = $horizonTemp
$env:TEMP = $horizonTemp
$env:PYTHONDONTWRITEBYTECODE = "1"
New-Item -ItemType Directory -Force $horizonTemp | Out-Null
& "$env:UV_PROJECT_ENVIRONMENT\Scripts\pytest.exe" -q -p no:cacheprovider
```
