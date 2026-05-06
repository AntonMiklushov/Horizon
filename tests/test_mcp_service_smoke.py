from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from src.models import ContentItem, SourceType
from src.mcp.server import hz_get_metrics
from src.mcp.service import HorizonPipelineService


def make_item(item_id: str, score: float | None = None) -> ContentItem:
    item = ContentItem(
        id=item_id,
        source_type=SourceType.RSS,
        title=f"Item {item_id}",
        url=f"https://example.com/{item_id}",
        content="content",
        author="tester",
        published_at=datetime.now(timezone.utc),
    )
    item.ai_score = score
    return item


def item_payloads(*items: ContentItem) -> list[dict]:
    return [item.model_dump(mode="json") for item in items]


def test_validate_config_smoke(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "config.json"
    config_path.write_text(
        (repo_root / "data" / "config.example.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    service = HorizonPipelineService(runs_root=tmp_path / "mcp-runs")
    result = asyncio.run(
        service.validate_config(
            horizon_path=str(repo_root),
            config_path=str(config_path),
            check_env=False,
        )
    )

    assert result["config_path"] == str(config_path.resolve())
    assert result["enabled_sources"]
    assert result["missing_env"] == []


def test_get_effective_config_can_filter_sources(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "config.json"
    config_path.write_text(
        (repo_root / "data" / "config.example.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    service = HorizonPipelineService(runs_root=tmp_path / "mcp-runs")
    result = service.get_effective_config(
        horizon_path=str(repo_root),
        config_path=str(config_path),
        sources=["rss"],
    )

    assert result["selected_sources"] == ["rss"]
    assert result["config"]["sources"]["github"] == []
    assert result["config"]["sources"]["rss"]


def test_metrics_tool_smoke() -> None:
    result = hz_get_metrics()

    assert result["ok"] is True
    assert result["tool"] == "hz_get_metrics"


def test_fetch_items_uses_public_orchestrator_api(tmp_path: Path, monkeypatch) -> None:
    service = HorizonPipelineService(runs_root=tmp_path / "mcp-runs")
    config_path = tmp_path / "config.json"

    monkeypatch.setattr(
        service,
        "_build_context",
        lambda **kwargs: (
            SimpleNamespace(
                horizon_path=tmp_path,
                config_path=config_path,
                runtime=SimpleNamespace(),
                config=SimpleNamespace(),
            ),
            ["rss"],
            [],
        ),
    )
    monkeypatch.setattr("src.mcp.service.make_storage", lambda runtime, config_path: object())

    class FakeOrchestrator:
        async def fetch_all_sources(self, since):  # type: ignore[no-untyped-def]
            return [make_item("item-1"), make_item("item-2")]

        def merge_cross_source_duplicates(self, items):  # type: ignore[no-untyped-def]
            return items[:1]

    monkeypatch.setattr(
        "src.mcp.service.make_orchestrator",
        lambda runtime, config, storage: FakeOrchestrator(),
    )

    result = asyncio.run(service.fetch_items(hours=6))

    assert result["fetched"] == 1
    assert result["raw_before_merge"] == 2
    assert service.run_store.load_items(result["run_id"], "raw")[0]["id"] == "item-1"


def test_filter_items_uses_public_topic_dedup_api(tmp_path: Path, monkeypatch) -> None:
    service = HorizonPipelineService(runs_root=tmp_path / "mcp-runs")
    service.run_store.create_run("run-topic-dedup")

    monkeypatch.setattr(
        service,
        "_load_stage_items",
        lambda **kwargs: (
            [make_item("item-1", score=9.0), make_item("item-2", score=8.0)],
            SimpleNamespace(
                runtime=SimpleNamespace(),
                config_path=tmp_path / "config.json",
                config=SimpleNamespace(filtering=SimpleNamespace(ai_score_threshold=7.0, max_items_per_source=5)),
            ),
        ),
    )
    monkeypatch.setattr("src.mcp.service.make_storage", lambda runtime, config_path: object())

    class FakeOrchestrator:
        async def merge_topic_duplicates(self, items):  # type: ignore[no-untyped-def]
            return items[:1]

    monkeypatch.setattr(
        "src.mcp.service.make_orchestrator",
        lambda runtime, config, storage: FakeOrchestrator(),
    )

    result = asyncio.run(service.filter_items(run_id="run-topic-dedup", topic_dedup=True))

    assert result["kept"] == 1
    assert result["removed_by_topic_dedup"] == 1
    assert service.run_store.load_items("run-topic-dedup", "filtered")[0]["id"] == "item-1"


def test_filter_items_applies_personal_evidence_gate(tmp_path: Path, monkeypatch) -> None:
    service = HorizonPipelineService(runs_root=tmp_path / "mcp-runs")
    service.run_store.create_run("run-personal-gate")
    good = make_item("good", score=9.0)
    good.metadata.update({"source_role": "fact_layer", "topic": "world", "include": True, "confidence": "medium"})
    missing_date = make_item("missing-date", score=9.0)
    missing_date.published_at = None
    missing_date.metadata.update({"source_role": "fact_layer", "topic": "world", "include": True, "confidence": "medium"})

    monkeypatch.setattr(
        service,
        "_load_stage_items",
        lambda **kwargs: (
            [good, missing_date],
            SimpleNamespace(
                runtime=SimpleNamespace(),
                config_path=tmp_path / "config.json",
                config=SimpleNamespace(
                        filtering=SimpleNamespace(ai_score_threshold=7.0, time_window_hours=24, max_items_per_source=5),
                    personal_briefing=SimpleNamespace(
                        enabled=True,
                        require_dates=True,
                        min_importance=7.0,
                        min_importance_priority_topics=6.5,
                    ),
                ),
            ),
        ),
    )
    monkeypatch.setattr("src.mcp.service.make_storage", lambda runtime, config_path: object())

    class FakeOrchestrator:
        def _classify_personal_source_metadata(self, items):  # type: ignore[no-untyped-def]
            return None

        async def merge_topic_duplicates(self, items):  # type: ignore[no-untyped-def]
            return items

    monkeypatch.setattr(
        "src.mcp.service.make_orchestrator",
        lambda runtime, config, storage: FakeOrchestrator(),
    )

    result = asyncio.run(service.filter_items(run_id="run-personal-gate", topic_dedup=True))

    assert result["kept"] == 1
    assert service.run_store.load_items("run-personal-gate", "filtered")[0]["id"] == "good"
    assert result["meta"]["personal_filter_excluded"][0]["reason"] == "missing publication date"


def test_effective_config_redacts_sensitive_fields() -> None:
    redacted = HorizonPipelineService._redact_config(
        {
            "ai": {"api_key_env": "OPENAI_API_KEY"},
            "email": {"email_address": "me@example.com", "password_env": "EMAIL_PASSWORD"},
            "webhook": {"headers": "Authorization: Bearer secret", "request_body": {"token": "abc"}},
        }
    )

    assert redacted["ai"]["api_key_env"] == "<redacted>"
    assert redacted["email"]["email_address"] == "<redacted>"
    assert redacted["webhook"]["headers"] == "<redacted>"
    assert redacted["webhook"]["request_body"] == "<redacted>"


def test_score_items_limit_is_optional_and_recorded(tmp_path: Path, monkeypatch) -> None:
    service = HorizonPipelineService(runs_root=tmp_path / "mcp-runs")
    service.run_store.create_run("run-score-limit")
    items = [make_item("item-1"), make_item("item-2"), make_item("item-3")]
    calls = []

    class FakeAnalyzer:
        def __init__(self, ai_client, personal_briefing_mode=False):  # type: ignore[no-untyped-def]
            self.ai_client = ai_client
            self.personal_briefing_mode = personal_briefing_mode

        async def analyze_batch(self, batch):  # type: ignore[no-untyped-def]
            calls.append((self.personal_briefing_mode, [item.id for item in batch]))
            for item in batch:
                item.ai_score = 8.0
            return batch

    monkeypatch.setattr(
        service,
        "_load_stage_items",
        lambda **kwargs: (
            items,
            SimpleNamespace(
                runtime=SimpleNamespace(
                    create_ai_client=lambda ai: object(),
                    ContentAnalyzer=FakeAnalyzer,
                ),
                config=SimpleNamespace(
                    ai=SimpleNamespace(),
                    filtering=SimpleNamespace(ai_score_threshold=7.0),
                    personal_briefing=SimpleNamespace(enabled=True),
                ),
            ),
        ),
    )

    result = asyncio.run(service.score_items("run-score-limit", max_items=2))

    assert calls == [(True, ["item-1", "item-2"])]
    assert result["source_items"] == 3
    assert result["items_used"] == 2
    assert result["skipped_by_limit"] == 1
    assert result["meta"]["scoring_limit"] == 2
    assert len(service.run_store.load_items("run-score-limit", "scored")) == 2


def test_score_items_without_limit_scores_all_items(tmp_path: Path, monkeypatch) -> None:
    service = HorizonPipelineService(runs_root=tmp_path / "mcp-runs")
    service.run_store.create_run("run-score-all")
    items = [make_item("item-1"), make_item("item-2")]
    calls = []

    class FakeAnalyzer:
        def __init__(self, ai_client, personal_briefing_mode=False):  # type: ignore[no-untyped-def]
            self.ai_client = ai_client

        async def analyze_batch(self, batch):  # type: ignore[no-untyped-def]
            calls.append([item.id for item in batch])
            return batch

    monkeypatch.setattr(
        service,
        "_load_stage_items",
        lambda **kwargs: (
            items,
            SimpleNamespace(
                runtime=SimpleNamespace(
                    create_ai_client=lambda ai: object(),
                    ContentAnalyzer=FakeAnalyzer,
                ),
                config=SimpleNamespace(
                    ai=SimpleNamespace(),
                    filtering=SimpleNamespace(ai_score_threshold=7.0),
                ),
            ),
        ),
    )

    result = asyncio.run(service.score_items("run-score-all"))

    assert calls == [["item-1", "item-2"]]
    assert result["items_used"] == 2
    assert result["skipped_by_limit"] == 0
    assert result["meta"]["scoring_limit"] is None


def test_enrich_items_limit_controls_llm_stage(tmp_path: Path, monkeypatch) -> None:
    service = HorizonPipelineService(runs_root=tmp_path / "mcp-runs")
    service.run_store.create_run("run-enrich-limit")
    items = [make_item("item-1", score=9.0), make_item("item-2", score=8.0), make_item("item-3", score=7.0)]
    calls = []

    class FakeEnricher:
        def __init__(self, ai_client):  # type: ignore[no-untyped-def]
            self.ai_client = ai_client

        async def enrich_batch(self, batch):  # type: ignore[no-untyped-def]
            calls.append([item.id for item in batch])
            for item in batch:
                item.metadata["sources"] = [{"url": str(item.url), "title": item.title}]

    monkeypatch.setattr(
        service,
        "_load_stage_items",
        lambda **kwargs: (
            items,
            SimpleNamespace(
                runtime=SimpleNamespace(
                    create_ai_client=lambda ai: object(),
                    ContentEnricher=FakeEnricher,
                ),
                config=SimpleNamespace(ai=SimpleNamespace()),
            ),
        ),
    )

    result = asyncio.run(service.enrich_items("run-enrich-limit", max_items=2))

    assert calls == [["item-1", "item-2"]]
    assert result["source_items"] == 3
    assert result["items_used"] == 2
    assert result["citation_count"] == 2
    assert result["meta"]["enrichment_limit"] == 2
    assert len(service.run_store.load_items("run-enrich-limit", "enriched")) == 2


def test_generate_summary_limit_controls_items_used(tmp_path: Path, monkeypatch) -> None:
    service = HorizonPipelineService(runs_root=tmp_path / "mcp-runs")
    service.run_store.create_run("run-summary-limit")
    items = [make_item("item-1", score=9.0), make_item("item-2", score=8.0), make_item("item-3", score=7.0)]

    class FakeSummarizer:
        async def generate_summary(self, batch, date, total_fetched, language="zh"):  # type: ignore[no-untyped-def]
            ids = ",".join(item.id for item in batch)
            return f"# {language} summary for {ids}; total={total_fetched}"

    monkeypatch.setattr(
        service,
        "_load_stage_items",
        lambda **kwargs: (
            items,
            SimpleNamespace(
                runtime=SimpleNamespace(DailySummarizer=FakeSummarizer),
                config=SimpleNamespace(),
            ),
        ),
    )

    result = asyncio.run(
        service.generate_summary(
            "run-summary-limit",
            language="ru",
            source_stage="filtered",
            max_items=2,
        )
    )

    assert result["source_items"] == 3
    assert result["items_used"] == 2
    assert result["skipped_by_limit"] == 1
    assert result["meta"]["summary_limit"] == 2
    assert "item-1,item-2" in service.run_store.load_summary("run-summary-limit", "ru")


def test_generate_summary_uses_personal_renderer_for_personal_language(tmp_path: Path, monkeypatch) -> None:
    service = HorizonPipelineService(runs_root=tmp_path / "mcp-runs")
    service.run_store.create_run("run-personal-summary")
    item = make_item("item-1", score=9.0)
    calls = []

    class FakeRenderer:
        def render(self, date, items, tracked=None):  # type: ignore[no-untyped-def]
            calls.append((date, [x.id for x in items], tracked))
            return "# Персональная сводка"

    monkeypatch.setattr(
        service,
        "_load_stage_items",
        lambda **kwargs: (
            [item],
            SimpleNamespace(
                runtime=SimpleNamespace(
                    PersonalBriefingRenderer=FakeRenderer,
                    run_briefing_critic=lambda summary, items: SimpleNamespace(
                        passed=True,
                        critical_issues=[],
                    ),
                ),
                config=SimpleNamespace(
                    personal_briefing=SimpleNamespace(
                        enabled=True,
                        language="ru",
                        critic_pass=SimpleNamespace(enabled=False, auto_revise_once=True),
                    ),
                ),
            ),
        ),
    )

    result = asyncio.run(
        service.generate_summary("run-personal-summary", language="ru", source_stage="filtered")
    )

    assert result["items_used"] == 1
    assert calls and calls[0][1] == ["item-1"]
    assert service.run_store.load_summary("run-personal-summary", "ru") == "# Персональная сводка"


def test_run_pipeline_passes_limits_to_llm_stages(tmp_path: Path, monkeypatch) -> None:
    service = HorizonPipelineService(runs_root=tmp_path / "mcp-runs")
    calls = []

    async def fake_fetch_items(**kwargs):  # type: ignore[no-untyped-def]
        run_id = service.run_store.create_run("run-pipeline-limit")
        calls.append(("fetch", kwargs))
        return {"run_id": run_id}

    async def fake_score_items(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(("score", kwargs))
        return {"scored": 2}

    async def fake_filter_items(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(("filter", kwargs))
        return {"kept": 1}

    async def fake_enrich_items(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(("enrich", kwargs))
        return {"enriched": 1}

    async def fake_generate_summary(**kwargs):  # type: ignore[no-untyped-def]
        calls.append(("summary", kwargs))
        return {"language": kwargs["language"]}

    monkeypatch.setattr(service, "fetch_items", fake_fetch_items)
    monkeypatch.setattr(service, "score_items", fake_score_items)
    monkeypatch.setattr(service, "filter_items", fake_filter_items)
    monkeypatch.setattr(service, "enrich_items", fake_enrich_items)
    monkeypatch.setattr(service, "generate_summary", fake_generate_summary)
    monkeypatch.setattr(
        service,
        "_build_context",
        lambda **kwargs: (
            SimpleNamespace(config=SimpleNamespace(ai=SimpleNamespace(languages=["ru"]))),
            ["rss"],
            [],
        ),
    )

    result = asyncio.run(service.run_pipeline(max_raw_items=2, max_filtered_items=1))

    assert result["limits"] == {"max_raw_items": 2, "max_filtered_items": 1}
    assert calls[1][0] == "score"
    assert calls[1][1]["max_items"] == 2
    assert calls[3][0] == "enrich"
    assert calls[3][1]["max_items"] == 1
    assert calls[4][0] == "summary"
    assert calls[4][1]["max_items"] is None
