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


class FakeProgress:
    instances = []

    def __init__(self, *columns, **kwargs):
        self.columns = columns
        self.kwargs = kwargs
        self.tasks = []
        self.updates = []
        FakeProgress.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_task(self, description, total=None):
        self.tasks.append({"description": description, "total": total})
        return len(self.tasks) - 1

    def update(self, task, **kwargs):
        self.updates.append({"task": task, **kwargs})


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


def test_per_item_analysis_progress_reports_current_item(monkeypatch):
    FakeProgress.instances = []
    monkeypatch.setattr(analyzer_module, "Progress", FakeProgress)

    analyzer = ContentAnalyzer(SimpleNamespace())
    items = [_make_item("rss:test:1"), _make_item("rss:test:2")]

    async def fake_analyze_item(item):
        item.ai_score = 8.0

    monkeypatch.setattr(analyzer, "_analyze_item", fake_analyze_item)

    asyncio.run(analyzer.analyze_batch(items))

    progress = FakeProgress.instances[0]
    descriptions = [u["description"] for u in progress.updates if "description" in u]
    assert progress.tasks[0]["total"] == 2
    assert "Analyzing item 1/2" in descriptions
    assert "Analyzing item 2/2" in descriptions


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


def test_codex_cli_batch_progress_tracks_batches_and_item_counts(monkeypatch):
    FakeProgress.instances = []
    monkeypatch.setattr(analyzer_module, "Progress", FakeProgress)

    class FakeCodexClient:
        config = SimpleNamespace(provider=AIProvider.CODEX_CLI, throttle_sec=0.0)

        async def complete(self, system, user):
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

    asyncio.run(analyzer.analyze_batch(items))

    progress = FakeProgress.instances[0]
    descriptions = [u["description"] for u in progress.updates if "description" in u]
    assert progress.tasks[0]["total"] == 1
    assert "Analyzing Codex batch 1/1 (0/2 items done, 2 in call)" in descriptions
    assert "Analyzed 2/2 items via Codex batches" in descriptions


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


def test_invalid_analysis_schema_becomes_controlled_failure():
    class FakeClient:
        async def complete(self, system, user):
            return json.dumps({"score": 42, "summary": "bad score"})

    analyzer = ContentAnalyzer(FakeClient())
    item = _make_item("rss:test:bad-schema")

    asyncio.run(analyzer._analyze_item(item))

    assert item.ai_score == 0.0
    assert item.ai_reason == "Analysis response schema invalid"
