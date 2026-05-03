import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import src.ai.analyzer as analyzer_module
from src.ai.analyzer import ContentAnalyzer
from src.models import AIProvider, ContentItem, SourceType


def _make_item(item_id: str) -> ContentItem:
    return ContentItem(
        id=item_id,
        source_type=SourceType.RSS,
        title=f"Item {item_id}",
        url="https://example.com/item",
        published_at=datetime(2026, 4, 26, tzinfo=timezone.utc),
    )


def test_analyze_batch_does_not_sleep_by_default(monkeypatch):
    analyzer = ContentAnalyzer(SimpleNamespace())
    items = [_make_item("rss:test:1"), _make_item("rss:test:2")]
    sleep_calls = []

    async def fake_analyze_item(item):
        item.ai_score = 8.0

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(analyzer, "_analyze_item", fake_analyze_item)
    monkeypatch.setattr(analyzer_module.asyncio, "sleep", fake_sleep)

    result = asyncio.run(analyzer.analyze_batch(items))

    assert len(result) == 2
    assert sleep_calls == []


def test_analyze_batch_sleeps_between_items_when_throttle_configured(monkeypatch):
    client = SimpleNamespace(config=SimpleNamespace(throttle_sec=1.5))
    analyzer = ContentAnalyzer(client)
    items = [_make_item("rss:test:1"), _make_item("rss:test:2"), _make_item("rss:test:3")]
    sleep_calls = []

    async def fake_analyze_item(item):
        item.ai_score = 8.0

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr(analyzer, "_analyze_item", fake_analyze_item)
    monkeypatch.setattr(analyzer_module.asyncio, "sleep", fake_sleep)

    asyncio.run(analyzer.analyze_batch(items))

    assert sleep_calls == [1.5, 1.5]


def test_codex_cli_analyze_batch_uses_single_batch_completion():
    calls = []

    class FakeCodexClient:
        config = SimpleNamespace(provider=AIProvider.CODEX_CLI, throttle_sec=0.0)

        async def complete(self, system, user):
            calls.append((system, user))
            payload = json.loads(user.split("Items:\n", 1)[1])
            return json.dumps({
                "items": [
                    {
                        "id": item["id"],
                        "score": 8,
                        "reason": "important",
                        "summary": f"summary {item['id']}",
                        "tags": ["test"],
                    }
                    for item in payload
                ]
            })

    analyzer = ContentAnalyzer(FakeCodexClient())
    items = [_make_item("rss:test:1"), _make_item("rss:test:2")]

    result = asyncio.run(analyzer.analyze_batch(items))

    assert len(calls) == 1
    assert result[0].ai_score == 8
    assert result[1].ai_summary == "summary rss:test:2"


def test_codex_cli_batch_failure_falls_back_to_item_analysis(monkeypatch):
    class FakeCodexClient:
        config = SimpleNamespace(provider=AIProvider.CODEX_CLI, throttle_sec=0.0)

        async def complete(self, system, user):
            raise AssertionError("batch call is monkeypatched below")

    analyzer = ContentAnalyzer(FakeCodexClient())
    items = [_make_item("rss:test:1"), _make_item("rss:test:2")]
    fallback_calls = []

    async def fake_analyze_item_chunk(chunk):
        raise ValueError("batch failed")

    async def fake_analyze_item(item):
        fallback_calls.append(item.id)
        item.ai_score = 7.0

    monkeypatch.setattr(analyzer, "_analyze_item_chunk", fake_analyze_item_chunk)
    monkeypatch.setattr(analyzer, "_analyze_item", fake_analyze_item)

    result = asyncio.run(analyzer.analyze_batch(items))

    assert fallback_calls == ["rss:test:1", "rss:test:2"]
    assert [item.ai_score for item in result] == [7.0, 7.0]
