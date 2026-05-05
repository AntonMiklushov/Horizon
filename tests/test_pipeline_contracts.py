from __future__ import annotations

from datetime import datetime, timezone

from src.models import ContentItem, SourceType
from src.horizon_ext.pipeline import apply_source_diversity, canonicalize_url


def _item(item_id: str, url: str, source_name: str = "Source") -> ContentItem:
    item = ContentItem(
        id=item_id,
        source_type=SourceType.RSS,
        title=item_id,
        url=url,
        published_at=datetime(2026, 5, 3, tzinfo=timezone.utc),
        metadata={"feed_name": source_name, "source_name": source_name},
    )
    item.ai_score = 9
    return item


def test_canonicalize_url_strips_tracking_but_keeps_meaningful_query():
    assert canonicalize_url("https://www.example.com/a/?utm_source=x&id=42#frag") == "https://example.com/a?id=42"
    assert canonicalize_url("https://example.com/search?q=ai") == "https://example.com/search?q=ai"


def test_source_diversity_records_exclusions():
    kept, excluded = apply_source_diversity(
        [_item("a", "https://example.com/a"), _item("b", "https://example.com/b")],
        max_items_per_source=1,
    )

    assert [item.id for item in kept] == ["a"]
    assert excluded == [{"id": "b", "item": "b", "reason": "source diversity cap"}]
