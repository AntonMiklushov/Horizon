"""Shared pipeline contracts and source-quality helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..models import ContentItem


TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "mkt_tok",
    "spm",
    "yclid",
}


@dataclass(frozen=True)
class SourceIdentity:
    """Normalized source identity used by selectors and renderers."""

    source_type: str
    name: str
    role: str = "unclassified"
    tier: str = "unknown"
    url: str = ""


@dataclass(frozen=True)
class SelectionTrace:
    """Reason why an item was selected or excluded."""

    item_id: str
    title: str
    selected: bool
    reason: str
    score: float | None = None
    source: SourceIdentity | None = None


@dataclass
class DigestItem:
    """Renderer-facing digest item."""

    item: ContentItem
    index: int
    title: str
    url: str
    score: float | None
    summary: str
    source: SourceIdentity
    evidence: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    references: list[dict[str, str]] = field(default_factory=list)


@dataclass
class DigestDocument:
    """Renderer-facing digest document assembled from selected items."""

    date: str
    language: str
    title: str
    items: list[DigestItem]
    total_fetched: int
    markdown: str
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    selection_traces: list[SelectionTrace] = field(default_factory=list)
    tracked_excluded: list[dict[str, str]] = field(default_factory=list)


@dataclass
class PipelineArtifacts:
    """Artifacts produced by a pipeline run."""

    raw_items: list[ContentItem] = field(default_factory=list)
    scored_items: list[ContentItem] = field(default_factory=list)
    selected_items: list[ContentItem] = field(default_factory=list)
    enriched_items: list[ContentItem] = field(default_factory=list)
    selection_traces: list[SelectionTrace] = field(default_factory=list)


def canonicalize_url(url: str) -> str:
    """Canonicalize a URL for cross-source dedupe without losing meaning."""

    parts = urlsplit(str(url).strip())
    scheme = (parts.scheme or "https").lower()
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m.") and host not in {"m.youtube.com"}:
        host = host[2:]

    netloc = host
    if parts.port:
        default_port = (scheme == "http" and parts.port == 80) or (scheme == "https" and parts.port == 443)
        if not default_port:
            netloc = f"{host}:{parts.port}"

    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    if path.endswith("/amp"):
        path = path[:-4] or "/"
    if path.startswith("/amp/"):
        path = path[4:] or "/"

    query_pairs = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        key_l = key.lower()
        if key_l in TRACKING_QUERY_KEYS or any(key_l.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES):
            continue
        query_pairs.append((key, value))
    query = urlencode(sorted(query_pairs), doseq=True)

    return urlunsplit((scheme, netloc, path, query, ""))


def source_identity(item: ContentItem) -> SourceIdentity:
    """Build a normalized source identity from item metadata."""

    meta = item.metadata
    name = (
        meta.get("source_name")
        or meta.get("feed_name")
        or meta.get("subreddit")
        or meta.get("channel")
        or meta.get("repo")
        or item.author
        or item.source_type.value
    )
    return SourceIdentity(
        source_type=item.source_type.value,
        name=str(name),
        role=str(meta.get("source_role", "unclassified")),
        tier=str(meta.get("source_reliability_tier", "unknown")),
        url=str(meta.get("source_url") or item.url),
    )


def build_selection_traces(
    selected: Iterable[ContentItem],
    excluded: Iterable[dict[str, str]] = (),
) -> list[SelectionTrace]:
    """Build transparent selected/excluded traces for diagnostics."""

    traces = [
        SelectionTrace(
            item_id=item.id,
            title=item.title,
            selected=True,
            reason="selected",
            score=item.ai_score,
            source=source_identity(item),
        )
        for item in selected
    ]
    traces.extend(
        SelectionTrace(
            item_id=str(entry.get("id", "")),
            title=str(entry.get("item", "")),
            selected=False,
            reason=str(entry.get("reason", "excluded")),
        )
        for entry in excluded
    )
    return traces


def apply_source_diversity(
    items: list[ContentItem],
    *,
    max_items_per_source: int,
) -> tuple[list[ContentItem], list[dict[str, str]]]:
    """Cap selected items per source identity while preserving score order."""

    kept: list[ContentItem] = []
    excluded: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for item in items:
        source = source_identity(item)
        key = f"{source.source_type}:{source.name}"
        count = counts.get(key, 0)
        if count >= max_items_per_source:
            excluded.append({"id": item.id, "item": item.title, "reason": "source diversity cap"})
            continue
        counts[key] = count + 1
        kept.append(item)
    return kept, excluded


def build_digest_document(
    *,
    date: str,
    language: str,
    title: str,
    items: list[ContentItem],
    total_fetched: int,
    markdown: str,
    selection_traces: list[SelectionTrace] | None = None,
    tracked_excluded: list[dict[str, str]] | None = None,
) -> DigestDocument:
    """Convert selected ContentItems into renderer-facing digest data."""

    digest_items: list[DigestItem] = []
    for index, item in enumerate(items, start=1):
        meta = item.metadata
        item_title = str(meta.get(f"title_{language}") or item.title)
        summary = str(
            meta.get(f"detailed_summary_{language}")
            or meta.get("detailed_summary")
            or meta.get("summary")
            or item.ai_summary
            or ""
        )
        refs = meta.get("sources") or []
        if not isinstance(refs, list):
            refs = []
        digest_items.append(
            DigestItem(
                item=item,
                index=index,
                title=item_title,
                url=str(item.url),
                score=item.ai_score,
                summary=summary,
                source=source_identity(item),
                evidence={
                    "confidence": meta.get("confidence"),
                    "evidence_strength": meta.get("evidence_strength"),
                    "claim_type": meta.get("claim_type"),
                    "why_it_matters": meta.get("why_it_matters") or item.ai_reason,
                    "confirmed_details": meta.get("confirmed_details") or [],
                    "who_claims": meta.get("who_claims") or [],
                },
                tags=list(item.ai_tags or []),
                references=[r for r in refs if isinstance(r, dict)],
            )
        )

    return DigestDocument(
        date=date,
        language=language,
        title=title,
        items=digest_items,
        total_fetched=total_fetched,
        markdown=markdown,
        selection_traces=selection_traces or [],
        tracked_excluded=tracked_excluded or [],
    )
