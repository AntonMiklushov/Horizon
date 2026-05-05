from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from src.models import Config, ContentItem, EmailConfig, SourceType, WebhookConfig
from src.mcp.service import HorizonPipelineService
from src.horizon_ext.web.app import WebRunState, _run_pipeline, _stage_payloads, create_app
from src.horizon_ext.web.config_forms import apply_basic_settings, apply_source_settings, web_default_hours


class Form(dict):
    def getlist(self, key: str):
        value = self.get(key, [])
        if isinstance(value, list):
            return value
        return [value]


def load_example_config() -> Config:
    root = Path(__file__).resolve().parents[1]
    return Config.model_validate_json((root / "data" / "config.example.json").read_text(encoding="utf-8"))


def write_config(tmp_path: Path, config: Config | None = None) -> Path:
    config = config or load_example_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.model_dump(mode="json")), encoding="utf-8")
    return config_path


def test_basic_settings_form_validates_and_sets_backup_fields() -> None:
    config = load_example_config()
    updated = apply_basic_settings(
        config,
        Form(
            {
                "web_default_hours": "12",
                "ai_score_threshold": "6.5",
                "max_items_per_source": "4",
                "languages": "ru,en",
                "output_formats": ["markdown", "html"],
            }
        ),
    )

    assert web_default_hours(updated) == 12
    assert updated.filtering.ai_score_threshold == 6.5
    assert updated.filtering.max_items_per_source == 4
    assert updated.ai.languages == ["ru", "en"]
    assert updated.rendering.output_formats == ["markdown", "html"]
    assert updated.publishing.enabled is False


def test_source_settings_form_adds_rss_without_exposing_secrets() -> None:
    config = load_example_config()
    updated = apply_source_settings(
        config,
        Form(
            {
                "github_count": "0",
                "hackernews_enabled": "on",
                "hackernews_fetch_top_stories": "20",
                "hackernews_min_score": "10",
                "hackernews_story_lists": ["top"],
                "rss_count": "0",
                "rss_new_enabled": "on",
                "rss_new_name": "Example",
                "rss_new_url": "https://example.com/feed.xml",
                "reddit_fetch_comments": "0",
                "reddit_subreddit_count": "0",
                "reddit_user_count": "0",
                "telegram_count": "0",
                "twitter_users": "",
            }
        ),
    )

    assert updated.sources.rss[0].name == "Example"
    assert str(updated.sources.rss[0].url) == "https://example.com/feed.xml"
    assert updated.webhook is None or updated.webhook.enabled is False


def test_dashboard_route_renders_config_and_run_form(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "Run Report" in response.text
    assert "Selection Instructions" in response.text
    assert "Edit Sources" in response.text


def test_settings_save_writes_backup(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))
    response = TestClient(app).post(
        "/settings/basic",
        data={
            "web_default_hours": "8",
            "ai_score_threshold": "7",
            "max_items_per_source": "3",
            "languages": "ru",
            "output_formats": "markdown",
        },
    )

    assert response.status_code == 200
    assert "Settings saved" in response.text
    assert config_path.with_suffix(".json.bak").exists()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["notes"]["web_default_hours"] == "8"


def test_source_settings_save_writes_backup(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))
    response = TestClient(app).post(
        "/settings/sources",
        data={
            "github_count": "0",
            "hackernews_fetch_top_stories": "10",
            "hackernews_min_score": "0",
            "hackernews_story_lists": "top",
            "rss_count": "0",
            "rss_new_enabled": "on",
            "rss_new_name": "Example",
            "rss_new_url": "https://example.com/feed.xml",
            "reddit_fetch_comments": "0",
            "reddit_subreddit_count": "0",
            "reddit_user_count": "0",
            "telegram_count": "0",
            "twitter_users": "",
        },
    )

    assert response.status_code == 200
    assert "Sources saved" in response.text
    assert config_path.with_suffix(".json.bak").exists()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["sources"]["rss"][0]["name"] == "Example"


def test_run_detail_renders_report_tabs_and_raw_stage(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    service = HorizonPipelineService(runs_root=tmp_path / "runs")
    service.run_store.create_run("web-demo")
    service.run_store.save_items(
        "web-demo",
        "raw",
        [
            {
                "id": "item-1",
                "source_type": "rss",
                "title": "Demo item",
                "url": "https://example.com/item",
                "metadata": {},
            }
        ],
    )
    service.run_store.update_meta(
        "web-demo",
        {"hours": 12, "raw_count": 1, "local_only": True, "run_instructions": "prioritize AI"},
    )
    service.run_store.save_summary("web-demo", "en", "# Horizon Demo\n\n- Demo item")
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path), service=service)

    response = TestClient(app).get("/runs/web-demo")

    assert response.status_code == 200
    assert "Horizon Demo" in response.text
    assert "Markdown" in response.text
    assert "Items" in response.text
    assert "Raw Data" in response.text
    assert "Demo item" in response.text


def test_web_run_pipeline_forces_local_only_and_stores_instructions() -> None:
    calls = []

    class FakeService:
        async def run_pipeline(self, **kwargs):
            calls.append(kwargs)
            return {"run_id": kwargs["run_id"], "summaries": []}

    app = SimpleNamespace(state=SimpleNamespace(service=FakeService()))
    state = WebRunState(run_id="web-run")
    params = {
        "run_id": "web-run",
        "hours": 12,
        "languages": ["ru"],
        "threshold": 6.5,
        "sources": ["rss"],
        "enrich": False,
        "topic_dedup": True,
        "run_instructions": "prioritize AI infra",
        "config_path": "data/config.json",
        "save_to_horizon_data": False,
        "local_only": True,
    }

    asyncio.run(_run_pipeline(app, state, params))

    assert state.status == "completed"
    assert calls[0]["local_only"] is True
    assert calls[0]["save_to_horizon_data"] is False
    assert calls[0]["run_instructions"] == "prioritize AI infra"


def test_service_passes_run_instructions_to_analyzer(tmp_path: Path, monkeypatch) -> None:
    service = HorizonPipelineService(runs_root=tmp_path / "runs")
    service.run_store.create_run("run-instructions")
    service.run_store.save_items(
        "run-instructions",
        "raw",
        [
            {
                "id": "item-1",
                "source_type": "rss",
                "title": "Item",
                "url": "https://example.com/item",
                "content": "body",
                "metadata": {},
            }
        ],
    )
    captured = {}

    class FakeAnalyzer:
        def __init__(self, ai_client, personal_briefing_mode=False, run_instructions=None):
            captured["instructions"] = run_instructions

        async def analyze_batch(self, items):
            for item in items:
                item.ai_score = 8
            return items

    item = ContentItem(id="item-1", source_type=SourceType.RSS, title="Item", url="https://example.com/item")
    monkeypatch.setattr(
        service,
        "_load_stage_items",
        lambda **kwargs: (
            [item],
            SimpleNamespace(
                runtime=SimpleNamespace(create_ai_client=lambda ai: object(), ContentAnalyzer=FakeAnalyzer),
                config=SimpleNamespace(
                    ai=SimpleNamespace(),
                    filtering=SimpleNamespace(ai_score_threshold=7.0),
                    personal_briefing=SimpleNamespace(enabled=False),
                ),
            ),
        ),
    )

    result = asyncio.run(
        service.score_items("run-instructions", run_instructions="prioritize AI infra")
    )

    assert result["scored"] == 1
    assert captured["instructions"] == "prioritize AI infra"
    assert result["meta"]["run_instructions"] == "prioritize AI infra"


def test_local_only_config_disables_external_outputs() -> None:
    config = load_example_config()
    clone = config.model_copy(deep=True)
    clone.email = EmailConfig(
        enabled=True,
        imap_server="imap.example.com",
        smtp_server="smtp.example.com",
        email_address="me@example.com",
    )
    clone.webhook = WebhookConfig(enabled=True, url_env="HORIZON_WEBHOOK_URL")
    clone.publishing.enabled = True
    local = HorizonPipelineService._local_only_config(clone)

    assert local.email.enabled is False
    assert local.webhook.enabled is False
    assert local.publishing.enabled is False


def test_run_stage_payload_redacts_secret_fields(tmp_path: Path) -> None:
    service = HorizonPipelineService(runs_root=tmp_path / "runs")
    service.run_store.create_run("web-secret")
    service.run_store.save_items(
        "web-secret",
        "raw",
        [
            {
                "id": "item-1",
                "source_type": "rss",
                "title": "Secret item",
                "url": "https://example.com/item",
                "metadata": {"api_token": "secret-token", "headers": {"Authorization": "Bearer secret"}},
            }
        ],
    )

    payloads = _stage_payloads(service, "web-secret", {"raw": True}, limit=20)

    assert "<redacted>" in payloads[0]["json"]
    assert "secret-token" not in payloads[0]["json"]
    assert "Bearer secret" not in payloads[0]["json"]
