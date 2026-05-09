from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx

from src.models import AIProvider, Config, ContentItem, EmailConfig, RSSSourceConfig, SourceType, WebhookConfig
from src.mcp.service import HorizonPipelineService
import src.horizon_ext.web.app as web_app_module
from src.horizon_ext.web import main as web_main
from src.horizon_ext.web.app import WebRunState, _run_params_from_form, _stage_payloads, _run_pipeline, create_app
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


def test_policy_settings_page_saves_controls_and_has_no_telegram_handling(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))
    client = TestClient(app)

    response = client.get("/settings/policy")
    assert response.status_code == 200
    assert 'name="telegram_handling"' not in response.text
    assert "source policy JSON" in response.text

    save = client.post(
        "/settings/policy",
        data={
            "personal_enabled": "on",
            "policy_preset": "personal-media",
            "source_policy_file": "data/source-policy.personal-media.example.json",
            "corroboration_enabled": "on",
            "sensitive_requires_confirmation": "on",
            "min_independent_confirmations": "1",
            "disable_sensitive_enrichment": "on",
            "filter_enrichment_results": "on",
            "selection_caps_enabled": "on",
            "max_sensitive_statement_items": "3",
        },
    )

    assert save.status_code == 200
    assert "Policy settings saved." in save.text
    saved = Config.model_validate_json(config_path.read_text(encoding="utf-8"))
    assert saved.personal_briefing.enabled is True
    assert saved.personal_briefing.source_policy_file == "data/source-policy.personal-media.example.json"
    assert saved.personal_briefing.corroboration.enabled is True
    assert saved.personal_briefing.corroboration.sensitive_requires_independent_confirmation is True
    assert saved.personal_briefing.corroboration.min_independent_confirmations == 1
    assert saved.personal_briefing.enrichment.disable_for_sensitive_topics is True
    assert saved.personal_briefing.enrichment.filter_search_results_by_source_policy is True
    assert saved.personal_briefing.selection_caps.enabled is True
    assert saved.personal_briefing.selection_caps.max_sensitive_statement_items == 3


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


def test_basic_settings_form_can_switch_to_lm_studio() -> None:
    config = load_example_config()
    config.ai.codex_extra_args = ["-c", 'model_reasoning_effort="high"']
    updated = apply_basic_settings(
        config,
        Form(
            {
                "llm_provider_mode": "lm_studio",
                "ai_model": "qwen2.5-7b-instruct",
                "ai_base_url": "",
                "ai_api_key_env": "SHOULD_NOT_SAVE",
                "codex_reasoning_effort": "low",
                "web_default_hours": "12",
                "ai_score_threshold": "6.5",
                "max_items_per_source": "4",
                "languages": "ru,en",
                "output_formats": ["markdown", "html"],
            }
        ),
    )

    assert updated.ai.provider == AIProvider.OPENAI
    assert updated.ai.model == "qwen2.5-7b-instruct"
    assert updated.ai.base_url == "http://127.0.0.1:1234/v1"
    assert updated.ai.api_key_env is None
    assert updated.ai.codex_extra_args == []


def test_basic_settings_form_can_switch_to_codex_cli_low_thinking() -> None:
    config = load_example_config()
    updated = apply_basic_settings(
        config,
        Form(
            {
                "llm_provider_mode": "codex_cli",
                "ai_model": "codex-cli",
                "codex_reasoning_effort": "low",
                "web_default_hours": "12",
                "ai_score_threshold": "6.5",
                "max_items_per_source": "4",
                "languages": "ru,en",
                "output_formats": ["markdown", "html"],
            }
        ),
    )

    assert updated.ai.provider == AIProvider.CODEX_CLI
    assert updated.ai.api_key_env is None
    assert updated.ai.base_url is None
    assert updated.ai.codex_extra_args == ["-c", 'model_reasoning_effort="low"']


def test_basic_settings_form_switches_cloud_provider_defaults_and_clears_codex_args() -> None:
    config = load_example_config()
    config.ai.codex_extra_args = ["-c", 'model_reasoning_effort="high"']
    updated = apply_basic_settings(
        config,
        Form(
            {
                "llm_provider_mode": "doubao",
                "ai_model": "doubao-seed-1.6",
                "ai_base_url": "",
                "ai_api_key_env": "",
                "codex_reasoning_effort": "low",
                "web_default_hours": "12",
                "ai_score_threshold": "6.5",
                "max_items_per_source": "4",
                "languages": "ru,en",
                "output_formats": ["markdown", "html"],
            }
        ),
    )

    assert updated.ai.provider == AIProvider.DOUBAO
    assert updated.ai.model == "doubao-seed-1.6"
    assert updated.ai.base_url == "https://ark.cn-beijing.volces.com/api/v3"
    assert updated.ai.api_key_env == "DOUBAO_API_KEY"
    assert updated.ai.codex_extra_args == []


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


def test_source_settings_form_can_remove_existing_rss() -> None:
    config = load_example_config()
    config.sources.rss = [
        RSSSourceConfig(name="Keep", url="https://example.com/keep.xml", enabled=True),
        RSSSourceConfig(name="Remove", url="https://example.com/remove.xml", enabled=True),
    ]
    updated = apply_source_settings(
        config,
        Form(
            {
                "github_count": "0",
                "hackernews_enabled": "on",
                "hackernews_fetch_top_stories": "20",
                "hackernews_min_score": "10",
                "hackernews_story_lists": ["top"],
                "rss_count": "2",
                "rss_0_enabled": "on",
                "rss_0_name": "Keep",
                "rss_0_url": "https://example.com/keep.xml",
                "rss_0_undated_policy": "drop",
                "rss_1_remove": "on",
                "reddit_fetch_comments": "0",
                "reddit_subreddit_count": "0",
                "reddit_user_count": "0",
                "telegram_count": "0",
                "twitter_users": "",
            }
        ),
    )

    assert [source.name for source in updated.sources.rss] == ["Keep"]


def test_dashboard_route_renders_config_and_run_form(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "Run Report" in response.text
    assert "dashboard-grid" in response.text
    assert "run-default-grid" in response.text
    assert "Selection Instructions" in response.text
    assert "Edit Sources" in response.text
    assert 'value="github" checked' in response.text
    assert 'value="rss" checked' in response.text
    assert 'value="twitter" checked' not in response.text


def test_dashboard_warns_when_personal_prefilter_is_active(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config = load_example_config()
    config.personal_briefing.enabled = True
    config_path = write_config(tmp_path, config)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "Personal briefing prefilter is active" in response.text
    assert "personal-prefilter-dashboard-notice" in response.text


def test_dashboard_local_lm_studio_defaults_to_bounded_ai_items(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config = load_example_config()
    config.ai.provider = AIProvider.OPENAI
    config.ai.base_url = "http://127.0.0.1:1234/v1"
    config.ai.api_key_env = None
    config_path = write_config(tmp_path, config)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert 'name="max_ai_items"' in response.text
    assert 'id="max_ai_items"' in response.text
    assert 'value="8"' in response.text
    assert "Provider: LM Studio /" in response.text


def test_web_app_uses_localappdata_run_store_fallback_when_data_runs_is_unwritable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fallback_base = tmp_path / "localappdata"
    monkeypatch.setenv("LOCALAPPDATA", str(fallback_base))
    monkeypatch.setattr(
        web_app_module,
        "_can_write_directory",
        lambda path: Path(path) == fallback_base / "HorizonBrief" / "mcp-runs",
    )

    app = create_app(config_path=str(write_config(tmp_path)), data_dir=str(tmp_path))

    assert Path(app.state.runs_root) == fallback_base / "HorizonBrief" / "mcp-runs"
    assert app.state.service.run_store.root == Path(app.state.runs_root)


def test_web_app_uses_writable_config_copy_when_config_file_is_unwritable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = write_config(tmp_path)
    fallback_base = tmp_path / "localappdata"
    monkeypatch.setenv("LOCALAPPDATA", str(fallback_base))
    monkeypatch.setattr(web_app_module, "_can_write_file", lambda path: Path(path) != config_path)

    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))

    active_config_path = Path(app.state.config_path)
    assert app.state.config_fallback_active is True
    assert active_config_path != config_path
    assert active_config_path.exists()
    payload = json.loads(active_config_path.read_text(encoding="utf-8"))
    assert Path(payload["personal_briefing"]["source_policy_file"]).is_absolute()


def test_runtime_config_banner_renders_when_using_writable_copy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    fallback_base = tmp_path / "localappdata"
    monkeypatch.setenv("LOCALAPPDATA", str(fallback_base))
    monkeypatch.setattr(web_app_module, "_can_write_file", lambda path: Path(path) != config_path)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))

    response = TestClient(app).get("/settings/basic")

    assert response.status_code == 200
    assert 'data-testid="config-runtime-banner"' in response.text
    assert str(config_path) in response.text
    assert str(app.state.config_path) in response.text


def test_full_config_editor_saves_valid_json_to_source_config(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config = load_example_config()
    config_path = write_config(tmp_path, config)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))
    client = TestClient(app)

    response = client.get("/settings/config")

    assert response.status_code == 200
    assert 'name="config_json"' in response.text
    assert "Save to Source Config" in response.text
    assert "Config path:" in response.text
    assert "Runtime copy:" not in response.text
    assert "Active config:" not in response.text
    assert response.text.count(str(config_path)) == 1

    payload = config.model_dump(mode="json")
    payload["notes"]["web_default_hours"] = "11"
    save_response = client.post(
        "/settings/config",
        data={"config_json": json.dumps(payload), "save_target": "primary"},
    )

    assert save_response.status_code == 200
    assert "Config saved to" in save_response.text
    saved_payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_payload["notes"]["web_default_hours"] == "11"


def test_full_config_editor_can_promote_runtime_copy_to_source_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from fastapi.testclient import TestClient

    config = load_example_config()
    config_path = write_config(tmp_path, config)
    monkeypatch.setattr(web_app_module, "_can_write_file", lambda path: Path(path) != config_path)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))
    assert app.state.config_fallback_active is True
    config_response = TestClient(app).get("/settings/config")
    assert config_response.status_code == 200
    assert "Config path:" in config_response.text
    assert "Runtime copy:" in config_response.text
    assert "Active config:" not in config_response.text

    payload = config.model_dump(mode="json")
    payload["notes"]["web_default_hours"] = "13"
    response = TestClient(app).post(
        "/settings/config",
        data={"config_json": json.dumps(payload), "save_target": "primary"},
    )

    assert response.status_code == 200
    assert app.state.config_fallback_active is False
    assert Path(app.state.config_path) == config_path
    saved_payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert saved_payload["notes"]["web_default_hours"] == "13"


def test_full_config_editor_returns_validation_error_for_bad_json(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))

    response = TestClient(app).post(
        "/settings/config",
        data={"config_json": '{"version": 1', "save_target": "primary"},
    )

    assert response.status_code == 400
    assert "Expecting" in response.text
    assert "version" in response.text


def test_basic_settings_route_renders_llm_provider_controls(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))

    response = TestClient(app).get("/settings/basic")

    assert response.status_code == 200
    assert 'name="llm_provider_mode"' in response.text
    assert 'value="lm_studio"' in response.text
    assert 'data-provider-field="api-key"' in response.text
    assert 'data-provider-field="base-url"' in response.text
    assert 'data-provider-field="codex-thinking"' in response.text
    assert 'name="ai_base_url"' in response.text
    assert 'name="codex_reasoning_effort"' in response.text


def test_web_main_disables_access_log_by_default(monkeypatch) -> None:
    calls = []

    monkeypatch.setattr(sys, "argv", ["horizon-web"])
    monkeypatch.setattr(web_main, "create_app", lambda config_path, data_dir: "app")
    monkeypatch.setattr(web_main.uvicorn, "run", lambda app, **kwargs: calls.append((app, kwargs)))

    web_main.main()

    assert calls[0][0] == "app"
    assert calls[0][1]["host"] == "127.0.0.1"
    assert calls[0][1]["port"] == 8787
    assert calls[0][1]["access_log"] is False
    assert calls[0][1]["log_level"] == "info"


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


def test_settings_save_returns_controlled_error_on_filesystem_failure(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))

    def fail_save(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise PermissionError("backup is locked")

    app.state.storage.save_config = fail_save
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

    assert response.status_code == 400
    assert "backup is locked" in response.text
    assert "Internal Server Error" not in response.text


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


def test_source_settings_save_returns_controlled_error_on_filesystem_failure(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))

    def fail_save(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise PermissionError("config backup is locked")

    app.state.storage.save_config = fail_save
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

    assert response.status_code == 400
    assert "config backup is locked" in response.text
    assert "Internal Server Error" not in response.text


def test_source_settings_form_has_specific_labels(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))

    response = TestClient(app).get("/settings/sources")

    assert response.status_code == 200
    for expected in [
        'for="github_0_enabled"',
        'for="github_0_username"',
        'for="hackernews_fetch_top_stories"',
        'for="rss_0_url"',
        'for="reddit_enabled"',
        'for="telegram_enabled"',
        'for="twitter_users"',
    ]:
        assert expected in response.text


def test_source_settings_route_uses_compact_source_manager(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))

    response = TestClient(app).get("/settings/sources")

    assert response.status_code == 200
    assert "data-source-manager" in response.text
    assert 'data-source-tab="rss"' in response.text
    assert 'data-source-panel="rss"' in response.text
    assert 'class="source-table' in response.text
    assert 'class="source-card' not in response.text


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
        {
            "hours": 12,
            "raw_count": 1,
            "local_only": True,
            "run_instructions": "prioritize AI",
            "status": "completed",
        },
    )
    service.run_store.save_summary("web-demo", "en", "# Horizon Demo\n\n- Demo item")
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path), service=service)

    response = TestClient(app).get("/runs/web-demo")

    assert response.status_code == 200
    assert "Horizon Demo" in response.text
    assert "Markdown" in response.text
    assert "Items" in response.text
    assert "Raw Data" in response.text
    assert "Model Activity" in response.text
    assert "Activity" in response.text
    assert "data-trace-list" in response.text
    assert "textContent" in response.text
    assert "Demo item" in response.text
    assert '<span class="pill">completed</span>' in response.text


def test_run_detail_renders_source_policy_summary_and_item_evidence(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    service = HorizonPipelineService(runs_root=tmp_path / "runs")
    service.run_store.create_run("web-evidence")
    service.run_store.save_items(
        "web-evidence",
        "filtered",
        [
            {
                "id": "item-1",
                "source_type": "rss",
                "title": "Evidence item",
                "url": "https://example.com/evidence",
                "ai_score": 8,
                "metadata": {
                    "source_role": "fact_layer",
                    "policy_decision": "allowed",
                    "claim_type": "confirmed_fact",
                    "confidence": "medium",
                    "counts_as_independent_confirmation": True,
                },
            }
        ],
    )
    service.run_store.write_json(
        "web-evidence",
        "source_policy_decisions.json",
        {
            "items": [
                {
                    "id": "item-1",
                    "source_role": "fact_layer",
                    "policy_decision": "allowed",
                    "claim_type": "confirmed_fact",
                    "confidence": "medium",
                },
                {
                    "id": "item-2",
                    "source_role": "unclassified",
                    "policy_decision": "allowed",
                    "source_policy_notes": "downgraded because source cannot independently confirm facts",
                },
            ],
            "excluded": [
                {
                    "id": "item-3",
                    "source_role": "blocked_as_fact_source",
                    "policy_decision": "excluded",
                }
            ],
        },
    )
    service.run_store.update_meta(
        "web-evidence",
        {"hours": 24, "raw_count": 3, "local_only": True, "status": "completed"},
    )
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path), service=service)

    response = TestClient(app).get("/runs/web-evidence")

    assert response.status_code == 200
    assert "Source policy decisions:" in response.text
    assert "3 items" in response.text
    assert "1 exclusions" in response.text
    assert "1 unclassified" in response.text
    assert "1 downgraded" in response.text
    for expected in [
        "Source role",
        "Policy decision",
        "Claim",
        "Confidence",
        "Independent",
        "fact_layer",
        "allowed",
        "confirmed_fact",
        "medium",
        "yes",
    ]:
        assert expected in response.text


def test_run_detail_renders_useful_empty_completed_report(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    service = HorizonPipelineService(runs_root=tmp_path / "runs")
    service.run_store.create_run("web-empty")
    service.run_store.update_meta(
        "web-empty",
        {"hours": 168, "raw_count": 5, "local_only": True, "status": "completed"},
    )
    service.run_store.save_summary(
        "web-empty",
        "ru",
        (
            "# Сводка — 2026-05-08\n\n"
            "Сегодня нет событий, прошедших порог отбора.\n\n"
            "## Контекст отбора\n\n"
            "- Получено из источников: 5\n"
            "- Прошло в итоговую сводку: 0\n"
            "- Порог отбора: 7\n"
        ),
    )
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path), service=service)

    response = TestClient(app).get("/runs/web-empty")

    assert response.status_code == 200
    assert "Сегодня нет событий" in response.text
    assert "Прошло в итоговую сводку: 0" in response.text
    assert "No rendered report is available yet." not in response.text


def test_run_detail_surfaces_prior_failed_activity(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    service = HorizonPipelineService(runs_root=tmp_path / "runs")
    service.run_store.create_run("web-failed")
    service.run_store.update_meta(
        "web-failed",
        {
            "hours": 12,
            "raw_count": 1,
            "local_only": True,
            "status": "failed",
            "current_stage": "filtered",
        },
    )
    service.run_store.append_trace_event(
        "web-failed",
        {
            "name": "llm.analysis.item",
            "stage": "analyze",
            "status": "failed",
            "message": "Analyzing item 1/1 (failed)",
            "fields": {"error": "APIConnectionError: Connection error."},
        },
    )
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path), service=service)
    app.state.runs["web-failed"] = WebRunState(
        run_id="web-failed",
        status="failed",
        message="Run failed",
        error="HZ_EMPTY_INPUT: No items available for enrichment.",
    )

    response = TestClient(app).get("/runs/web-failed")

    assert response.status_code == 200
    assert "Earlier failed activity" in response.text
    assert "APIConnectionError: Connection error." in response.text


def test_run_detail_surfaces_rss_source_diagnostics(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    service = HorizonPipelineService(runs_root=tmp_path / "runs")
    service.run_store.create_run("web-rss-failed")
    service.run_store.update_meta(
        "web-rss-failed",
        {
            "hours": 3,
            "raw_count": 0,
            "local_only": True,
            "status": "failed",
            "current_stage": "fetch",
        },
    )
    service.run_store.append_trace_event(
        "web-rss-failed",
        {
            "name": "source.RSS Feeds",
            "stage": "fetch",
            "status": "failed",
            "message": "Fetch failed for RSS Feeds",
            "fields": {
                "error": "RSSFetchError",
                "detail": "All 18 enabled RSS feeds failed.",
                "diagnostics": [
                    {
                        "source": "Example",
                        "error": "ConnectError",
                        "message": "Connection failed",
                    }
                ],
            },
        },
    )
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path), service=service)
    app.state.runs["web-rss-failed"] = WebRunState(
        run_id="web-rss-failed",
        status="failed",
        message="Run failed",
        error="HZ_EMPTY_INPUT: No items available for scoring.",
    )

    response = TestClient(app).get("/runs/web-rss-failed")

    assert response.status_code == 200
    assert "Fetch failed for RSS Feeds: RSSFetchError: All 18 enabled RSS feeds failed." in response.text


def test_run_detail_surfaces_rss_source_warnings_when_empty_run_fails(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    service = HorizonPipelineService(runs_root=tmp_path / "runs")
    service.run_store.create_run("web-rss-warning")
    service.run_store.update_meta(
        "web-rss-warning",
        {
            "hours": 3,
            "raw_count": 0,
            "local_only": True,
            "status": "failed",
            "current_stage": "analyze",
        },
    )
    service.run_store.append_trace_event(
        "web-rss-warning",
        {
            "name": "source.RSS Feeds",
            "stage": "fetch",
            "status": "warning",
            "message": "Fetched RSS Feeds with source warnings",
            "fields": {
                "diagnostics": [
                    {
                        "source": "Example",
                        "error": "HTTPStatusError",
                        "message": "Server error",
                    }
                ],
            },
        },
    )
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path), service=service)
    app.state.runs["web-rss-warning"] = WebRunState(
        run_id="web-rss-warning",
        status="failed",
        message="Run failed",
        error="HZ_EMPTY_INPUT: No items available for scoring.",
    )

    response = TestClient(app).get("/runs/web-rss-warning")

    assert response.status_code == 200
    assert "Fetched RSS Feeds with source warnings" in response.text
    assert "first: Example: HTTPStatusError: Server error" in response.text


def test_run_detail_explains_personal_prefilter_exclusions(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    service = HorizonPipelineService(runs_root=tmp_path / "runs")
    service.run_store.create_run("web-prefilter")
    service.run_store.update_meta(
        "web-prefilter",
        {
            "hours": 12,
            "raw_count_before_merge": 2,
            "raw_count_after_personal_prefilter": 0,
            "raw_count": 0,
            "personal_prefilter_excluded": [{"id": "item-1"}, {"id": "item-2"}],
            "local_only": True,
            "status": "failed",
            "current_stage": "raw",
        },
    )
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path), service=service)

    response = TestClient(app).get("/runs/web-prefilter")

    assert response.status_code == 200
    assert "Personal briefing prefilter excluded all 2 fetched items before scoring." in response.text


def test_run_detail_renders_cancel_for_active_run(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    service = HorizonPipelineService(runs_root=tmp_path / "runs")
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path), service=service)
    app.state.runs["web-running"] = WebRunState(run_id="web-running", status="running", message="Fetching sources")

    response = TestClient(app).get("/runs/web-running")

    assert response.status_code == 200
    assert 'action="/runs/web-running/cancel"' in response.text
    assert "Cancel run" in response.text


def test_dashboard_active_runs_filters_terminal_states(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    service = HorizonPipelineService(runs_root=tmp_path / "runs")
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path), service=service)
    app.state.runs["web-running"] = WebRunState(run_id="web-running", status="running")
    app.state.runs["web-completed"] = WebRunState(run_id="web-completed", status="completed")

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "web-running" in response.text
    assert "web-completed" not in response.text


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


def test_run_params_applies_local_lm_studio_default_ai_item_cap(tmp_path: Path) -> None:
    config = load_example_config()
    config.ai.provider = AIProvider.OPENAI
    config.ai.base_url = "http://127.0.0.1:1234/v1"
    config.ai.api_key_env = None
    config_path = write_config(tmp_path, config)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))
    request = SimpleNamespace(app=app)

    params = _run_params_from_form(
        Form({"hours": "1", "languages": "ru", "sources": ["rss"]}),
        "web-run",
        request,
    )

    assert params["max_raw_items"] == 8
    assert params["max_filtered_items"] == 8


def test_run_params_respects_explicit_ai_item_cap(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path))
    request = SimpleNamespace(app=app)

    params = _run_params_from_form(
        Form({"hours": "1", "languages": "ru", "sources": ["rss"], "max_ai_items": "3"}),
        "web-run",
        request,
    )

    assert params["max_raw_items"] == 3
    assert params["max_filtered_items"] == 3


def test_web_run_pipeline_marks_state_failed_when_trace_setup_fails() -> None:
    class TraceFailingService:
        def create_trace_reporter(self, run_id: str):  # type: ignore[no-untyped-def]
            raise PermissionError("run store is locked")

        async def run_pipeline(self, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("pipeline should not start after trace setup failure")

    app = SimpleNamespace(state=SimpleNamespace(service=TraceFailingService()))
    state = WebRunState(run_id="web-trace-failure", status="running")

    asyncio.run(_run_pipeline(app, state, {"run_id": "web-trace-failure"}))

    assert state.status == "failed"
    assert state.message == "Run failed"
    assert state.error == "PermissionError: run store is locked"
    assert state.completed_at is not None


def test_web_run_pipeline_marks_state_cancelled_when_task_is_cancelled() -> None:
    class SlowService:
        async def run_pipeline(self, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.Event().wait()

    async def scenario() -> WebRunState:
        app = SimpleNamespace(state=SimpleNamespace(service=SlowService()))
        state = WebRunState(run_id="web-cancel", status="running")
        task = asyncio.create_task(_run_pipeline(app, state, {"run_id": "web-cancel"}))
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.wait_for(task, timeout=1)
        return state

    state = asyncio.run(scenario())

    assert state.status == "cancelled"
    assert state.message == "Run cancelled"
    assert state.completed_at is not None


def test_cancel_route_cancels_active_run_task(tmp_path: Path) -> None:
    class SlowService:
        async def run_pipeline(self, **kwargs):  # type: ignore[no-untyped-def]
            await asyncio.Event().wait()

    async def scenario() -> WebRunState:
        config_path = write_config(tmp_path)
        app = create_app(config_path=str(config_path), data_dir=str(tmp_path), service=SlowService())  # type: ignore[arg-type]
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/runs",
                data={"hours": "1", "languages": "ru", "sources": "rss"},
                follow_redirects=False,
            )
            assert response.status_code == 303
            run_id = response.headers["location"].rsplit("/", 1)[-1]

            cancel_response = await client.post(f"/runs/{run_id}/cancel", follow_redirects=False)
            assert cancel_response.status_code == 303

            for _ in range(50):
                state = app.state.runs[run_id]
                if state.status == "cancelled":
                    break
                await asyncio.sleep(0.01)
            return app.state.runs[run_id]

    state = asyncio.run(scenario())

    assert state.status == "cancelled"
    assert state.cancel_requested_at is not None


def test_run_api_includes_redacted_trace_events(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    service = HorizonPipelineService(runs_root=tmp_path / "runs")
    service.run_store.create_run("web-trace")
    service.run_store.append_trace_event(
        "web-trace",
        {
            "timestamp": "2026-05-06T00:00:00+00:00",
            "stage": "analyze",
            "status": "calling",
            "message": "Analyzing item 1/1",
            "fields": {"api_token": "secret-token", "headers": {"Authorization": "Bearer secret"}},
        },
    )
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path), service=service)

    response = TestClient(app).get("/api/runs/web-trace")

    assert response.status_code == 200
    payload = response.json()
    assert payload["trace"][0]["message"] == "Analyzing item 1/1"
    rendered = json.dumps(payload)
    assert "<redacted>" in rendered
    assert "secret-token" not in rendered
    assert "Bearer secret" not in rendered


def test_run_api_uses_activity_message_for_running_state(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    config_path = write_config(tmp_path)
    service = HorizonPipelineService(runs_root=tmp_path / "runs")
    service.run_store.create_run("web-running-trace")
    service.run_store.append_trace_event(
        "web-running-trace",
        {
            "timestamp": "2026-05-06T00:00:00+00:00",
            "name": "llm.enrichment.item",
            "stage": "enrich",
            "status": "calling",
            "message": "Enriching item 3/8",
            "progress": {"current": 3, "total": 8},
            "fields": {"index": 3, "total": 8},
        },
    )
    app = create_app(config_path=str(config_path), data_dir=str(tmp_path), service=service)
    app.state.runs["web-running-trace"] = WebRunState(
        run_id="web-running-trace",
        status="running",
        message="Fetching and analyzing sources",
    )

    response = TestClient(app).get("/api/runs/web-running-trace")

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"]["message"] == "Enriching item 3/8"
    assert payload["activity"]["stage"] == "enrich"


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
