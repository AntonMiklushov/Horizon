# Personal Media Briefing

Personal media mode adds a conservative evidence layer for feeds that mix primary reporting with discovery platforms.

## Source Policy

Use `data/source-policy.personal-media.example.json` with:

- canonical evidence domains in `source_domain_rules`
- Telegram, Reddit, and Hacker News listed as discovery source types
- optional allowlisted Telegram channels for primary statements only

`ContentItem.source_type` is treated as the discovery source. `ContentItem.url` is treated as the canonical evidence URL when a scraper found an external link. A Telegram, Reddit, or Hacker News item linking to Reuters is therefore classified by `reuters.com`, while the platform remains in `discovery_source_type` and does not count as independent confirmation.

## Evidence Metadata

Personal runs add audit fields to item metadata:

- `discovery_source_type`, `discovery_source_name`, `discovery_url`, `discovery_role`
- `content_source_domain`
- `can_confirm_fact`, `can_confirm_sensitive`, `counts_as_independent_confirmation`
- `policy_decision`, `source_policy_notes`
- `corroboration`
- `sensitive_topic_auto`
- `enrichment_skipped`

MCP and web runs also write `source_policy_decisions.json` into the run directory.

## Corroboration

Sensitive high-confidence claims require independent confirmation according to `personal_briefing.corroboration`.

Discovery-only sources do not confirm facts. Russian institutional sensitive items without fact-layer corroboration are downgraded to `party_claim`. Official sensitive items without external confirmation are downgraded to `official_statement`. High confidence is lowered to medium when corroboration fails.

## Enrichment

By default, sensitive personal items skip DuckDuckGo enrichment and stay in the summary with `enrichment_skipped: "sensitive topic"`. Non-sensitive enrichment search results are filtered through the source policy when `filter_search_results_by_source_policy` is enabled.

## Web Settings

The dashboard exposes compact controls at `/settings/policy`. The full JSON editor remains available at `/settings/config` for source allowlists and advanced policy changes.
