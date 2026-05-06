from src.models import AIConfig, AIProvider
from src.setup.wizard import build_config, _source_is_safe_default


def _ai_cfg(langs=None):
    return AIConfig(provider=AIProvider.CODEX_CLI, model="codex-cli", languages=langs or ["ru"])


def test_build_config_personal_briefing_ru_and_no_hn_default():
    cfg = build_config(_ai_cfg(["ru"]), [], enable_personal_briefing=True)
    assert cfg.personal_briefing.enabled is True
    assert cfg.personal_briefing.language == "ru"
    assert cfg.personal_briefing.generate_standard_summaries is False
    assert cfg.sources.hackernews.enabled is False


def test_build_config_non_personal_does_not_force_standard_summaries_true():
    cfg = build_config(_ai_cfg(["en"]), [], enable_personal_briefing=False)
    assert cfg.personal_briefing.enabled is False
    assert cfg.personal_briefing.language == "en"
    assert cfg.personal_briefing.generate_standard_summaries is False


def test_build_config_personal_briefing_forces_ru_language():
    cfg = build_config(_ai_cfg(["en"]), [], enable_personal_briefing=True)
    assert cfg.personal_briefing.language == "ru"


def test_source_safety_filters_social_and_placeholder_lwn():
    assert _source_is_safe_default({"type": "github_repo", "config": {"owner": "a", "repo": "b"}})
    assert not _source_is_safe_default({"type": "hackernews", "config": {}})
    assert not _source_is_safe_default({"type": "reddit_subreddit", "config": {"subreddit": "news"}})
    assert not _source_is_safe_default({"type": "telegram", "config": {"channel": "x"}})
    assert not _source_is_safe_default({"type": "rss", "config": {"url": "https://lwn.net/headlines/full_text?key=${LWN_KEY}"}})
