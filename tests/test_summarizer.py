"""Unit tests for daily summary rendering."""

import asyncio
from datetime import datetime, timezone

from src.ai.summarizer import DailySummarizer
from src.models import ContentItem, SourceType


def _make_item(idx: int) -> ContentItem:
    item = ContentItem(
        id=f"rss:item-{idx}",
        source_type=SourceType.RSS,
        title=f"Important Item {idx}",
        url=f"https://example.com/items/{idx}",
        content="content",
        author="tester",
        published_at=datetime(2026, 4, 25, 8, 0, tzinfo=timezone.utc),
    )
    item.ai_score = 8.0
    item.ai_summary = f"Summary for item {idx}."
    item.ai_tags = ["AI", "News"]
    return item


def test_generate_webhook_overview_lists_items_without_full_details():
    summarizer = DailySummarizer()
    items = [_make_item(1), _make_item(2)]

    result = summarizer.generate_webhook_overview(
        items,
        date="2026-04-25",
        total_fetched=10,
        language="en",
    )

    assert "Selected 2 important items from 10 fetched items" in result
    assert "1. [Important Item 1](https://example.com/items/1)" in result
    assert "2. [Important Item 2](https://example.com/items/2)" in result
    assert "Summary for item 1." not in result


def test_generate_webhook_item_renders_single_item_detail():
    summarizer = DailySummarizer()

    result = summarizer.generate_webhook_item(
        _make_item(1),
        language="en",
        index=1,
        total=2,
    )

    assert result.startswith("Item 1/2")
    assert "## [Important Item 1](https://example.com/items/1)" in result
    assert "Summary for item 1." in result
    assert "**Tags**: `#AI`, `#News`" in result


def test_generate_summary_supports_russian_labels():
    summarizer = DailySummarizer()

    result = asyncio.run(
        summarizer.generate_summary(
            [_make_item(1)],
            date="2026-04-25",
            total_fetched=10,
            language="ru",
        )
    )

    assert result.startswith("# Ежедневная сводка Horizon Brief")
    assert "Из 10 материалов выбрано важных: 1" in result
    assert "**Источник**" not in result
    assert "**Теги**: `#AI`, `#News`" in result


def test_generate_empty_summary_supports_russian_labels():
    summarizer = DailySummarizer()

    result = asyncio.run(
        summarizer.generate_summary([], date="2026-04-25", total_fetched=10, language="ru")
    )

    assert "Проанализировано материалов: 10" in result
    assert "Сегодня нет значимых событий" in result
