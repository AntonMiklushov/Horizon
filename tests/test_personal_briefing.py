import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.horizon_ext.personal import (
    EvidenceChecker,
    PersonalBriefingRenderer,
    SourcePolicyClassifier,
    conservative_default_policy,
    load_source_policy,
    run_briefing_critic,
)
from src.ai.prompts import PERSONAL_BRIEFING_ANALYSIS_USER
from src.models import AIConfig, AIProvider, Config, ContentItem, FilteringConfig, SourceType, SourcesConfig
from src.orchestrator import HorizonOrchestrator
from src.storage.manager import StorageManager


PYTHON = "python"


def mk_item(url, source=SourceType.RSS, author="", title="t", published_at=None, content="content"):
    return ContentItem(
        id=title,
        source_type=source,
        title=title,
        url=url,
        author=author,
        content=content,
        published_at=published_at if published_at is not None else datetime.now(timezone.utc) - timedelta(hours=1),
    )


def mk_config(tmp_path, personal=None, languages=None):
    return Config(
        version="1",
        ai=AIConfig(
            provider=AIProvider.OPENAI,
            model="m",
            api_key_env="X",
            languages=languages or ["en"],
        ),
        sources=SourcesConfig(),
        filtering=FilteringConfig(),
        personal_briefing=personal or {"enabled": True, "source_policy_file": str(tmp_path / "p.json")},
    )


def mk_orchestrator(tmp_path, personal=None, languages=None):
    policy = tmp_path / "p.json"
    if not policy.exists():
        policy.write_text("{}", encoding="utf-8")
    cfg = mk_config(tmp_path, personal=personal, languages=languages)
    return HorizonOrchestrator(cfg, StorageManager(data_dir=str(tmp_path / "data")))


def test_personal_summary_title_is_neutral(tmp_path):
    o = mk_orchestrator(tmp_path, personal={"enabled": True, "language": "ru"})

    assert o._summary_title("2026-05-03", "ru") == "Сводка - 2026-05-03"


def test_prompt_format_does_not_crash():
    rendered = PERSONAL_BRIEFING_ANALYSIS_USER.format(
        title="Test title",
        source="rss",
        author="Unknown",
        url="https://reuters.com/world/test",
        published_at="2026-05-03T20:00:00+02:00",
        content="Test content",
        metadata='{"source_role":"fact_layer"}',
    )
    assert "Верни JSON" in rendered


def test_policy_load_and_override(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(
        '{"source_domain_rules":{"reuters.com":{"role":"context_layer","tier":"tier1","name":"R"}}}',
        encoding="utf-8",
    )
    policy, _ = load_source_policy(str(p))
    assert policy.source_domain_rules["reuters.com"]["role"] == "context_layer"


@pytest.mark.parametrize(
    ("item", "role", "source_name", "notes"),
    [
        (mk_item("https://reuters.com/world/test"), "fact_layer", "Reuters", ""),
        (mk_item("https://apnews.com/article/test"), "fact_layer", "Associated Press", ""),
        (mk_item("https://ft.com/content/test"), "context_layer", "Financial Times", ""),
        (mk_item("https://interfax.ru/world/test"), "russian_institutional_frame", "Interfax", ""),
        (mk_item("https://kommersant.ru/doc/test"), "russian_institutional_frame", "Kommersant", ""),
        (mk_item("https://rbc.ru/politics/test"), "russian_institutional_frame", "RBC", ""),
        (mk_item("https://mos.ru/news/test"), "official_primary_source", "mos.ru", ""),
        (mk_item("https://nature.com/articles/test"), "science_primary_source", "Nature", ""),
        (mk_item("https://arxiv.org/abs/2605.00001"), "science_preprint", "arXiv", "not peer-reviewed"),
        (mk_item("https://github.com/org/repo/releases/tag/v1"), "tech_primary_source", "GitHub", ""),
        (mk_item("https://unknown.example/news"), "unclassified", None, ""),
        (mk_item("https://reuters.com/world/test", source=SourceType.REDDIT), "blocked_as_fact_source", "Reddit", ""),
        (mk_item("https://apnews.com/article/test", source=SourceType.HACKERNEWS), "blocked_as_fact_source", "Hacker News", ""),
        (mk_item("https://github.com/org/repo/releases/tag/v1", source=SourceType.HACKERNEWS), "blocked_as_fact_source", "Hacker News", ""),
        (mk_item("https://t.me/channel/1", source=SourceType.TELEGRAM), "blocked_as_fact_source", "Telegram", ""),
        (mk_item("https://x.com/openai/status/1", source=SourceType.TWITTER), "social_primary_statement_only", "Twitter/X", ""),
        (mk_item("https://github.com/org/repo/releases/tag/v1", source=SourceType.TWITTER), "social_primary_statement_only", "Twitter/X", ""),
    ],
)
def test_source_policy_classifier_required_roles(item, role, source_name, notes):
    out = SourcePolicyClassifier(conservative_default_policy()).classify(item)
    assert out["source_role"] == role
    if source_name is not None:
        assert out["source_name"] == source_name
    assert out["source_policy_notes"] == notes


def test_prefilter_social_and_blocked_rules(tmp_path):
    personal = {
        "enabled": True,
        "source_policy_file": str(tmp_path / "p.json"),
    }
    (tmp_path / "p.json").write_text('{"allowed_social_primary_actors":["openai"]}', encoding="utf-8")
    o = mk_orchestrator(tmp_path, personal=personal)
    blocked = mk_item("https://reuters.com/a", source=SourceType.REDDIT, title="blocked")
    unknown = mk_item("https://unknown.example/a", title="unknown")
    allowed_social = mk_item("https://x.com/openai/status/1", source=SourceType.TWITTER, author="@openai", title="allowed")
    disallowed_social = mk_item("https://x.com/random/status/1", source=SourceType.TWITTER, author="@random", title="disallowed")
    allowed = mk_item("https://reuters.com/a", title="allowed-rss")
    for item in [blocked, unknown, allowed_social, disallowed_social, allowed]:
        item.metadata.update(o.personal_classifier.classify(item))
    candidates, excluded = o._prefilter_personal_candidates([blocked, unknown, allowed_social, disallowed_social, allowed])

    assert [i.title for i in candidates] == ["allowed", "allowed-rss"]
    assert allowed_social.metadata["claim_type"] == "primary_statement"
    assert {x["item"] for x in excluded} == {"blocked", "unknown", "disallowed"}
    merged = o.merge_cross_source_duplicates(candidates)
    assert [i.title for i in merged] == ["allowed", "allowed-rss"]


def test_evidence_checker_required_downgrades():
    checker = EvidenceChecker(time_window_hours=24)

    blocked = mk_item("https://reddit.com/r/news/1", source=SourceType.REDDIT)
    blocked.metadata.update({"source_role": "blocked_as_fact_source", "claim_type": "confirmed_fact", "confidence": "high"})
    checker.audit_item(blocked)
    assert blocked.metadata["claim_type"] == "unverified_report"
    assert blocked.metadata["confidence"] == "low"

    unclassified = mk_item("https://unknown.example/a")
    unclassified.metadata.update({"source_role": "unclassified", "claim_type": "confirmed_fact", "confidence": "high"})
    checker.audit_item(unclassified)
    assert unclassified.metadata["claim_type"] == "unverified_report"
    assert unclassified.metadata["confidence"] == "low"

    social_confirmed = mk_item("https://x.com/openai/status/1", source=SourceType.TWITTER)
    social_confirmed.metadata.update({"source_role": "social_primary_statement_only", "claim_type": "confirmed_fact", "confidence": "high"})
    checker.audit_item(social_confirmed)
    assert social_confirmed.metadata["claim_type"] == "primary_statement"
    assert social_confirmed.metadata["confidence"] == "low"

    for claim_type in ["official_statement", "party_claim", "market_reaction", "correction_or_update"]:
        social = mk_item(f"https://x.com/openai/status/{claim_type}", source=SourceType.TWITTER)
        social.metadata.update({"source_role": "social_primary_statement_only", "claim_type": claim_type, "confidence": "high"})
        checker.audit_item(social)
        assert social.metadata["claim_type"] == "primary_statement"
        assert social.metadata["confidence"] == "medium"
        assert social.metadata.get("include", True) is True

    ru_sensitive = mk_item("https://interfax.ru/a")
    ru_sensitive.metadata.update({
        "source_role": "russian_institutional_frame",
        "sensitive_topic": True,
        "claim_type": "confirmed_fact",
        "confidence": "high",
    })
    checker.audit_item(ru_sensitive)
    assert ru_sensitive.metadata["claim_type"] == "party_claim"
    assert ru_sensitive.metadata["confidence"] == "medium"

    official_sensitive = mk_item("https://mos.ru/a")
    official_sensitive.metadata.update({
        "source_role": "official_primary_source",
        "sensitive_topic": True,
        "claim_type": "confirmed_fact",
        "confidence": "high",
    })
    checker.audit_item(official_sensitive)
    assert official_sensitive.metadata["claim_type"] == "official_statement"
    assert official_sensitive.metadata["confidence"] == "medium"

    missing_date = mk_item("https://reuters.com/a", published_at=None)
    missing_date.published_at = None
    missing_date.metadata.update({"source_role": "fact_layer", "claim_type": "analysis", "confidence": "high"})
    checker.audit_item(missing_date)
    assert missing_date.metadata["confidence"] == "low"

    old = mk_item("https://reuters.com/old", published_at=datetime.now(timezone.utc) - timedelta(hours=25))
    old.metadata.update({"source_role": "fact_layer", "claim_type": "analysis", "confidence": "medium"})
    checker.audit_item(old)
    assert "outside time window" in old.metadata["source_conflicts"]


def test_renderer_required_sections_and_labels():
    items = []
    for topic in ["russia", "moscow", "world_economy", "tech_ai", "open_source", "big_tech", "science", "world", "other"]:
        item = mk_item(f"https://reuters.com/{topic}", title=f"{topic} title")
        item.metadata.update({
            "topic": topic,
            "confidence": "medium",
            "summary": f"{topic} summary",
            "confirmed_details": ["confirmed detail"],
            "who_claims": ["source claim"],
            "evidence_strength": "medium",
            "why_it_matters": "important",
            "source_role": "fact_layer",
            "source_name": "Reuters",
            "supporting_sources": [{
                "source_name": "mos.ru",
                "source_url": "https://mos.ru/news/1",
                "source_role": "official_primary_source",
                "source_reliability_tier": "tier2",
                "publication_date": "2026-05-03T20:00:00+03:00",
                "title": "support",
            }],
        })
        items.append(item)
    ru = mk_item("https://interfax.ru/a", title="institutional")
    ru.metadata.update({"topic": "russia", "confidence": "medium", "source_role": "russian_institutional_frame", "source_name": "Interfax"})
    official = mk_item("https://mos.ru/a", title="official")
    official.metadata.update({"topic": "moscow", "confidence": "medium", "source_role": "official_primary_source", "source_name": "mos.ru"})
    disputed = mk_item("https://reuters.com/disputed", title="disputed")
    disputed.metadata.update({"topic": "world", "confidence": "low", "source_role": "fact_layer", "source_name": "Reuters"})

    out = PersonalBriefingRenderer().render("2026-05-03", items + [ru, official, disputed])

    assert out.startswith("# Сводка — 2026-05-03")
    assert "## Главное" not in out
    assert "## Россия" in out
    assert "## Москва" in out
    assert "## Мировая экономика" in out
    assert "## Технологии: AI, Open Source, Big Tech" in out
    assert "## Наука" in out
    assert "## Мир" in out
    assert "## Другое" in out
    assert "## Спорные / слабоподтверждённые сообщения" in out
    assert "### disputed" in out
    assert "российская институциональная рамка" in out
    assert "официальный источник" in out
    assert "Поддерживающие источники:" in out
    for label in [
        "Что произошло",
        "Что подтверждено",
        "Кто что утверждает",
        "Оценка доказательств",
        "Почему важно",
        "Уверенность",
        "Источники",
    ]:
        assert label in out
    for topic in ["russia", "moscow", "world_economy", "tech_ai", "open_source", "big_tech", "science", "world", "other"]:
        assert out.count(f"### {topic} title") == 1


def test_renderer_does_not_render_main_for_disputed_only():
    item = mk_item("https://interfax.ru/a", title="disputed-only")
    item.metadata.update({"source_role": "russian_institutional_frame", "confidence": "low", "topic": "world", "summary": "x"})
    out = PersonalBriefingRenderer().render("2026-05-02", [item])
    assert "## Спорные / слабоподтверждённые сообщения" in out
    assert "## Главное" not in out


def test_renderer_puts_unknown_topics_in_other_without_critic_warning():
    item = mk_item("https://interfax.ru/security", title="security item")
    item.metadata.update({
        "topic": "russia_security",
        "confidence": "medium",
        "summary": "security summary",
        "source_role": "russian_institutional_frame",
        "source_name": "Interfax",
        "source_url": "https://interfax.ru/security",
        "claim_type": "party_claim",
    })

    out = PersonalBriefingRenderer().render("2026-05-03", [item])
    critic = run_briefing_critic(out, [item])

    assert "## Другое" in out
    assert "### security item" in out
    assert out.count("### security item") == 1
    assert critic.passed is True


def test_cross_source_duplicate_merge_preserves_supporting_sources(tmp_path):
    o = mk_orchestrator(tmp_path)
    primary = mk_item("https://reuters.com/a", title="primary", content="long content")
    duplicate = mk_item("https://www.reuters.com/a", title="duplicate", content="x")
    duplicate.metadata.update({
        "source_name": "Associated Press",
        "source_role": "fact_layer",
        "source_reliability_tier": "tier1",
        "publication_date": "2026-05-03T20:00:00+00:00",
    })
    merged = o.merge_cross_source_duplicates([primary, duplicate])
    assert len(merged) == 1
    supporting = merged[0].metadata["supporting_sources"]
    assert supporting == [{
        "source_name": "Associated Press",
        "source_url": "https://www.reuters.com/a",
        "source_role": "fact_layer",
        "source_reliability_tier": "tier1",
        "publication_date": "2026-05-03T20:00:00+00:00",
        "title": "duplicate",
    }]


def test_topic_duplicate_merge_preserves_supporting_sources(tmp_path, monkeypatch):
    o = mk_orchestrator(tmp_path)
    primary = mk_item("https://reuters.com/a", title="primary")
    duplicate = mk_item("https://apnews.com/a", title="duplicate")
    primary.ai_score = 9
    duplicate.ai_score = 8
    duplicate.metadata.update({
        "source_name": "Associated Press",
        "source_role": "fact_layer",
        "source_reliability_tier": "tier1",
        "publication_date": "2026-05-03T20:00:00+00:00",
    })

    class FakeClient:
        async def complete(self, system, user):
            return '{"duplicates": [[0, 1]]}'

    monkeypatch.setattr("src.orchestrator.create_ai_client", lambda config: FakeClient())
    merged = asyncio.run(o.merge_topic_duplicates([primary, duplicate]))
    assert merged == [primary]
    assert primary.metadata["supporting_sources"] == [{
        "source_name": "Associated Press",
        "source_url": "https://apnews.com/a",
        "source_role": "fact_layer",
        "source_reliability_tier": "tier1",
        "publication_date": "2026-05-03T20:00:00+00:00",
        "title": "duplicate",
    }]


def test_no_items_default_mode_returns_early(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = mk_config(tmp_path, personal={"enabled": False})
    o = HorizonOrchestrator(cfg, StorageManager(data_dir=str(tmp_path / "data")))

    async def _empty(_since):
        return []

    o.fetch_all_sources = _empty  # type: ignore[assignment]
    asyncio.run(o.run())
    assert not list((tmp_path / "data" / "summaries").glob("*.md"))
    assert not (tmp_path / "docs" / "_posts").exists()


def test_fetch_all_sources_raises_when_every_scheduled_source_fails(tmp_path):
    cfg = mk_config(tmp_path, personal={"enabled": False})
    cfg.sources.reddit.enabled = False
    cfg.sources.telegram.enabled = False
    cfg.sources.hackernews.enabled = True
    o = HorizonOrchestrator(cfg, StorageManager(data_dir=str(tmp_path / "data")))

    async def _fail(name, _scraper, _since):
        raise RuntimeError(f"{name} down")

    o._fetch_with_progress = _fail  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="All scheduled sources failed: Hacker News"):
        asyncio.run(o.fetch_all_sources(datetime.now(timezone.utc) - timedelta(hours=1)))


def test_twitter_reply_reanalysis_reapplies_threshold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = mk_config(tmp_path, personal={"enabled": False}, languages=["en"])
    cfg.rendering.output_formats = ["markdown"]
    cfg.publishing.enabled = False
    o = HorizonOrchestrator(cfg, StorageManager(data_dir=str(tmp_path / "data")))
    item = mk_item("https://twitter.com/x/status/1", source=SourceType.TWITTER, title="tweet")
    enriched_batches = []

    async def _fetch(_since):
        return [item]

    async def _analyze(items):
        for x in items:
            x.ai_score = 9
            x.ai_summary = "initial summary"
        return items

    async def _dedup(items):
        return items

    async def _expand(items):
        for x in items:
            x.ai_score = 6

    async def _enrich(items):
        enriched_batches.append(list(items))

    o.fetch_all_sources = _fetch  # type: ignore[assignment]
    o._analyze_content = _analyze  # type: ignore[assignment]
    o.merge_topic_duplicates = _dedup  # type: ignore[assignment]
    o._expand_twitter_discussion = _expand  # type: ignore[assignment]
    o._enrich_important_items = _enrich  # type: ignore[assignment]

    asyncio.run(o.run())

    assert enriched_batches == [[]]


def test_personal_no_items_saves_empty_without_analyze(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    personal = {
        "enabled": True,
        "language": "ru",
        "source_policy_file": str(tmp_path / "p.json"),
        "daily_empty_allowed": True,
        "generate_standard_summaries": False,
    }
    o = mk_orchestrator(tmp_path, personal=personal, languages=["en", "zh"])

    async def _empty(_since):
        return []

    async def _analyze(_items):
        raise AssertionError("_analyze_content must not be called for empty personal briefing")

    o.fetch_all_sources = _empty  # type: ignore[assignment]
    o._analyze_content = _analyze  # type: ignore[assignment]
    asyncio.run(o.run())
    files = list((tmp_path / "data" / "summaries").glob("*-ru.md"))
    assert len(files) == 1
    text = files[0].read_text(encoding="utf-8")
    assert text.startswith("# Сводка — ")
    assert "Сегодня нет событий" in text
    assert not list((tmp_path / "data" / "summaries").glob("*-en.md"))
    assert not list((tmp_path / "data" / "summaries").glob("*-zh.md"))


def test_personal_standard_summaries_false_skips_daily_summarizer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    personal = {
        "enabled": True,
        "language": "ru",
        "source_policy_file": str(tmp_path / "p.json"),
        "generate_standard_summaries": False,
    }
    o = mk_orchestrator(tmp_path, personal=personal, languages=["en", "zh"])
    item = mk_item("https://reuters.com/world/test", title="Reuters item")

    async def _fetch(_since):
        return [item]

    async def _analyze(items):
        assert [x.title for x in items] == ["Reuters item"]
        for x in items:
            x.ai_score = 9
            x.metadata.update({"topic": "world", "confidence": "medium", "include": True, "claim_type": "confirmed_fact"})
        return items

    async def _enrich(_items):
        return None

    async def _dedup(items):
        return items

    class ForbiddenSummarizer:
        def __init__(self):
            raise AssertionError("DailySummarizer should not be created for personal-only output")

    o.fetch_all_sources = _fetch  # type: ignore[assignment]
    o._analyze_content = _analyze  # type: ignore[assignment]
    o._enrich_important_items = _enrich  # type: ignore[assignment]
    o.merge_topic_duplicates = _dedup  # type: ignore[assignment]
    monkeypatch.setattr("src.orchestrator.DailySummarizer", ForbiddenSummarizer)

    asyncio.run(o.run())
    assert len(list((tmp_path / "data" / "summaries").glob("*-ru.md"))) == 1
    assert not list((tmp_path / "data" / "summaries").glob("*-en.md"))
    assert not list((tmp_path / "data" / "summaries").glob("*-zh.md"))


def test_personal_standard_summaries_true_includes_configured_languages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    personal = {
        "enabled": True,
        "language": "ru",
        "source_policy_file": str(tmp_path / "p.json"),
        "generate_standard_summaries": True,
    }
    o = mk_orchestrator(tmp_path, personal=personal, languages=["en"])
    item = mk_item("https://reuters.com/world/test", title="Reuters item")
    calls = []

    async def _fetch(_since):
        return [item]

    async def _analyze(items):
        for x in items:
            x.ai_score = 9
            x.metadata.update({"topic": "world", "confidence": "medium", "include": True, "claim_type": "confirmed_fact"})
        return items

    async def _enrich(_items):
        return None

    async def _dedup(items):
        return items

    class FakeSummarizer:
        async def generate_summary(self, items, date, total_fetched, language="en"):
            calls.append(language)
            return f"# standard {language}"

    o.fetch_all_sources = _fetch  # type: ignore[assignment]
    o._analyze_content = _analyze  # type: ignore[assignment]
    o._enrich_important_items = _enrich  # type: ignore[assignment]
    o.merge_topic_duplicates = _dedup  # type: ignore[assignment]
    monkeypatch.setattr("src.orchestrator.DailySummarizer", FakeSummarizer)

    asyncio.run(o.run())
    assert calls == ["en"]
    assert len(list((tmp_path / "data" / "summaries").glob("*-ru.md"))) == 1
    assert len(list((tmp_path / "data" / "summaries").glob("*-en.md"))) == 1


def test_local_personal_smoke_without_real_api(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    personal = {
        "enabled": True,
        "language": "ru",
        "source_policy_file": str(tmp_path / "p.json"),
        "generate_standard_summaries": False,
    }
    o = mk_orchestrator(tmp_path, personal=personal, languages=["en", "zh"])
    reuters = mk_item("https://reuters.com/world/test", title="Reuters world")
    moscow = mk_item("https://mos.ru/news/test", title="Moscow official")
    reddit = mk_item("https://reuters.com/world/test", source=SourceType.REDDIT, title="Reddit link")

    async def _fetch(_since):
        return [reuters, moscow, reddit]

    async def _analyze(items):
        assert [i.title for i in items] == ["Reuters world", "Moscow official"]
        for item in items:
            item.ai_score = 9
            item.metadata.update({
                "topic": "moscow" if "Moscow" in item.title else "world",
                "confidence": "medium",
                "include": True,
                "claim_type": "confirmed_fact",
                "summary": f"summary {item.title}",
                "confirmed_details": ["detail"],
                "who_claims": ["source"],
                "why_it_matters": "matters",
            })
        return items

    async def _enrich(_items):
        return None

    async def _dedup(items):
        return items

    o.fetch_all_sources = _fetch  # type: ignore[assignment]
    o._analyze_content = _analyze  # type: ignore[assignment]
    o._enrich_important_items = _enrich  # type: ignore[assignment]
    o.merge_topic_duplicates = _dedup  # type: ignore[assignment]

    asyncio.run(o.run())
    [summary_path] = list((tmp_path / "data" / "summaries").glob("*-ru.md"))
    text = summary_path.read_text(encoding="utf-8")
    assert "Reuters world" in text
    assert "Moscow official" in text
    assert "Reddit link" not in text
    assert "## Мир" in text
    assert "## Москва" in text
    assert "fact_layer" in text
    assert "официальный источник" in text
    assert "https://reuters.com/world/test" in text
    assert "https://mos.ru/news/test" in text
    assert not list((tmp_path / "data" / "summaries").glob("*-en.md"))
    assert not list((tmp_path / "data" / "summaries").glob("*-zh.md"))


def test_example_config_has_generate_standard_summaries():
    cfg = json.loads(open("data/config.example.json", encoding="utf-8").read())
    assert "generate_standard_summaries" in cfg.get("personal_briefing", {})
