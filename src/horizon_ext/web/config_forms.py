"""Safe web form parsing for Horizon Brief configuration."""

from __future__ import annotations

from typing import Any, Iterable

from pydantic import ValidationError

from ...models import (
    AIProvider,
    Config,
    GitHubSourceConfig,
    HackerNewsConfig,
    RSSSourceConfig,
    RedditConfig,
    RedditSubredditConfig,
    RedditUserConfig,
    TelegramConfig,
    TelegramChannelConfig,
    TwitterConfig,
)


OUTPUT_FORMATS = ("markdown", "html", "email_html")
STORY_LISTS = ("top", "new", "best", "ask", "show", "job")
LLM_PROVIDER_MODES = ("codex_cli", "lm_studio", "openai", "anthropic", "gemini", "ali", "doubao", "minimax")
LM_STUDIO_DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"
LM_STUDIO_DEFAULT_MODEL = "local-model"
CODEX_REASONING_EFFORTS = ("minimal", "low", "medium", "high")
API_KEY_ENV_DEFAULTS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "ali": "DASHSCOPE_API_KEY",
    "doubao": "DOUBAO_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}
BASE_URL_DEFAULTS = {
    "ali": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
    "minimax": "https://api.minimax.io/v1",
}


class ConfigFormError(ValueError):
    """Raised when a submitted settings form is invalid."""


def source_summary(config: Config) -> dict[str, int]:
    """Return enabled source counts by source type."""

    return {
        "github": len([source for source in config.sources.github if source.enabled]),
        "hackernews": 1 if config.sources.hackernews.enabled else 0,
        "rss": len([source for source in config.sources.rss if source.enabled]),
        "reddit": (
            len([source for source in config.sources.reddit.subreddits if source.enabled])
            + len([source for source in config.sources.reddit.users if source.enabled])
            if config.sources.reddit.enabled
            else 0
        ),
        "telegram": (
            len([source for source in config.sources.telegram.channels if source.enabled])
            if config.sources.telegram.enabled
            else 0
        ),
        "twitter": len(config.sources.twitter.users) if config.sources.twitter and config.sources.twitter.enabled else 0,
    }


def web_default_hours(config: Config) -> int:
    """Read the dashboard's default run window from config notes."""

    try:
        return max(1, min(720, int(config.notes.get("web_default_hours", "24"))))
    except ValueError:
        return 24


def apply_basic_settings(config: Config, form: Any) -> Config:
    """Apply safe basic settings from a web form."""

    clone = config.model_copy(deep=True)
    clone.notes["web_default_hours"] = str(_int(form, "web_default_hours", 24, minimum=1, maximum=720))
    clone.filtering.ai_score_threshold = _float(form, "ai_score_threshold", minimum=0.0, maximum=10.0)
    clone.filtering.max_items_per_source = _int(form, "max_items_per_source", 5, minimum=1, maximum=100)
    clone.ai.languages = _csv(form, "languages", required=True)
    _apply_ai_settings(clone, form)
    clone.rendering.output_formats = _choices(form, "output_formats", OUTPUT_FORMATS, required=True)
    clone.publishing.enabled = _checked(form, "publishing_enabled")
    return _validated(clone)


def apply_source_settings(config: Config, form: Any) -> Config:
    """Apply safe source settings from a web form."""

    clone = config.model_copy(deep=True)
    clone.sources.github = _github_sources(form)
    clone.sources.hackernews = HackerNewsConfig(
        enabled=_checked(form, "hackernews_enabled"),
        fetch_top_stories=_int(form, "hackernews_fetch_top_stories", 30, minimum=1, maximum=500),
        min_score=_int(form, "hackernews_min_score", 100, minimum=0),
        story_lists=_choices(form, "hackernews_story_lists", STORY_LISTS, required=True),
        include_jobs=_checked(form, "hackernews_include_jobs"),
    )
    clone.sources.rss = _rss_sources(form)
    clone.sources.reddit = _reddit_config(form)
    clone.sources.telegram.channels = _telegram_channels(form)
    clone.sources.telegram.enabled = _checked(form, "telegram_enabled")
    clone.sources.twitter = _twitter_config(form)
    return _validated(clone)


def apply_policy_settings(config: Config, form: Any) -> Config:
    """Apply personal briefing policy controls from the compact policy page."""

    clone = config.model_copy(deep=True)
    personal = clone.personal_briefing
    personal.enabled = _checked(form, "personal_enabled")
    personal.source_policy_file = _str(form, "source_policy_file") or personal.source_policy_file
    personal.corroboration.enabled = _checked(form, "corroboration_enabled")
    personal.corroboration.sensitive_requires_independent_confirmation = _checked(
        form,
        "sensitive_requires_confirmation",
    )
    personal.corroboration.min_independent_confirmations = _int(
        form,
        "min_independent_confirmations",
        1,
        minimum=0,
        maximum=5,
    )
    personal.enrichment.disable_for_sensitive_topics = _checked(form, "disable_sensitive_enrichment")
    personal.enrichment.filter_search_results_by_source_policy = _checked(form, "filter_enrichment_results")
    personal.selection_caps.enabled = _checked(form, "selection_caps_enabled")
    personal.selection_caps.max_sensitive_statement_items = _int(
        form,
        "max_sensitive_statement_items",
        2,
        minimum=0,
        maximum=50,
    )

    preset = _str(form, "policy_preset")
    if preset == "personal-media":
        personal.source_policy_file = "data/source-policy.personal-media.example.json"
    elif preset == "personal-news":
        personal.source_policy_file = "data/config.personal-news.example.json"

    return _validated(clone)


def _github_sources(form: Any) -> list[GitHubSourceConfig]:
    sources = []
    for index in range(_int(form, "github_count", 0, minimum=0, maximum=500)):
        prefix = f"github_{index}"
        if _checked(form, f"{prefix}_remove"):
            continue
        source = _github_source_from_prefix(form, prefix)
        if source:
            sources.append(source)
    new_source = _github_source_from_prefix(form, "github_new", skip_empty=True)
    if new_source:
        sources.append(new_source)
    return sources


def _github_source_from_prefix(
    form: Any,
    prefix: str,
    skip_empty: bool = False,
) -> GitHubSourceConfig | None:
    source_type = _str(form, f"{prefix}_type") or "user_events"
    enabled = _checked(form, f"{prefix}_enabled")
    username = _str(form, f"{prefix}_username")
    owner = _str(form, f"{prefix}_owner")
    repo = _str(form, f"{prefix}_repo")
    if skip_empty and not any((username, owner, repo)):
        return None
    if source_type == "user_events":
        if not username:
            raise ConfigFormError("GitHub user sources require a username.")
        return GitHubSourceConfig(type="user_events", username=username, enabled=enabled)
    if source_type == "repo_releases":
        if not owner or not repo:
            raise ConfigFormError("GitHub release sources require owner and repo.")
        return GitHubSourceConfig(type="repo_releases", owner=owner, repo=repo, enabled=enabled)
    raise ConfigFormError(f"Unsupported GitHub source type: {source_type}")


def _rss_sources(form: Any) -> list[RSSSourceConfig]:
    sources = []
    for index in range(_int(form, "rss_count", 0, minimum=0, maximum=1000)):
        prefix = f"rss_{index}"
        if _checked(form, f"{prefix}_remove"):
            continue
        source = _rss_source_from_prefix(form, prefix)
        if source:
            sources.append(source)
    new_source = _rss_source_from_prefix(form, "rss_new", skip_empty=True)
    if new_source:
        sources.append(new_source)
    return sources


def _rss_source_from_prefix(form: Any, prefix: str, skip_empty: bool = False) -> RSSSourceConfig | None:
    name = _str(form, f"{prefix}_name")
    url = _str(form, f"{prefix}_url")
    if skip_empty and not name and not url:
        return None
    if not name or not url:
        raise ConfigFormError("RSS sources require name and URL.")
    return RSSSourceConfig(
        name=name,
        url=url,
        enabled=_checked(form, f"{prefix}_enabled"),
        category=_str(form, f"{prefix}_category") or None,
        undated_policy=_str(form, f"{prefix}_undated_policy") or "drop",
    )


def _reddit_config(form: Any) -> RedditConfig:
    subreddits = []
    for index in range(_int(form, "reddit_subreddit_count", 0, minimum=0, maximum=500)):
        prefix = f"reddit_subreddit_{index}"
        if _checked(form, f"{prefix}_remove"):
            continue
        name = _str(form, f"{prefix}_name")
        if not name:
            raise ConfigFormError("Reddit subreddit rows require a subreddit name.")
        subreddits.append(
            RedditSubredditConfig(
                subreddit=name,
                enabled=_checked(form, f"{prefix}_enabled"),
                sort=_str(form, f"{prefix}_sort") or "hot",
                time_filter=_str(form, f"{prefix}_time_filter") or "day",
                fetch_limit=_int(form, f"{prefix}_fetch_limit", 25, minimum=1, maximum=100),
                min_score=_int(form, f"{prefix}_min_score", 10, minimum=0),
            )
        )
    new_subreddit = _str(form, "reddit_subreddit_new_name")
    if new_subreddit:
        subreddits.append(RedditSubredditConfig(subreddit=new_subreddit))

    users = []
    for index in range(_int(form, "reddit_user_count", 0, minimum=0, maximum=500)):
        prefix = f"reddit_user_{index}"
        if _checked(form, f"{prefix}_remove"):
            continue
        username = _str(form, f"{prefix}_username")
        if not username:
            raise ConfigFormError("Reddit user rows require a username.")
        users.append(
            RedditUserConfig(
                username=username,
                enabled=_checked(form, f"{prefix}_enabled"),
                sort=_str(form, f"{prefix}_sort") or "new",
                fetch_limit=_int(form, f"{prefix}_fetch_limit", 10, minimum=1, maximum=100),
            )
        )
    new_user = _str(form, "reddit_user_new_username")
    if new_user:
        users.append(RedditUserConfig(username=new_user))

    return RedditConfig(
        enabled=_checked(form, "reddit_enabled"),
        subreddits=subreddits,
        users=users,
        fetch_comments=_int(form, "reddit_fetch_comments", 5, minimum=0, maximum=20),
    )


def _telegram_channels(form: Any) -> list[TelegramChannelConfig]:
    channels = []
    for index in range(_int(form, "telegram_count", 0, minimum=0, maximum=500)):
        prefix = f"telegram_{index}"
        if _checked(form, f"{prefix}_remove"):
            continue
        channel = _str(form, f"{prefix}_channel")
        if not channel:
            raise ConfigFormError("Telegram rows require a channel.")
        channels.append(
            TelegramChannelConfig(
                channel=channel,
                enabled=_checked(form, f"{prefix}_enabled"),
                fetch_limit=_int(form, f"{prefix}_fetch_limit", 20, minimum=1, maximum=100),
            )
        )
    new_channel = _str(form, "telegram_new_channel")
    if new_channel:
        channels.append(TelegramChannelConfig(channel=new_channel))
    return channels


def _twitter_config(form: Any) -> TwitterConfig | None:
    users = _csv(form, "twitter_users", required=False)
    enabled = _checked(form, "twitter_enabled")
    if not users and not enabled:
        return None
    return TwitterConfig(
        enabled=enabled,
        apify_token_env=_str(form, "twitter_apify_token_env") or "APIFY_TOKEN",
        actor_id=_str(form, "twitter_actor_id") or "altimis~scweet",
        users=users,
        fetch_limit=_int(form, "twitter_fetch_limit", 10, minimum=1, maximum=500),
        fetch_reply_text=_checked(form, "twitter_fetch_reply_text"),
        max_replies_per_tweet=_int(form, "twitter_max_replies_per_tweet", 3, minimum=0, maximum=50),
        max_tweets_to_expand=_int(form, "twitter_max_tweets_to_expand", 10, minimum=0, maximum=100),
        reply_min_likes=_int(form, "twitter_reply_min_likes", 0, minimum=0),
    )


def _validated(config: Config) -> Config:
    try:
        return Config.model_validate(config.model_dump(mode="json"))
    except ValidationError as exc:
        raise ConfigFormError(str(exc)) from exc


def _apply_ai_settings(config: Config, form: Any) -> None:
    provider_mode = _str(form, "llm_provider_mode")
    if not provider_mode:
        return
    if provider_mode not in LLM_PROVIDER_MODES:
        raise ConfigFormError(f"Unsupported LLM provider: {provider_mode}")

    model = _str(form, "ai_model")
    base_url = _str(form, "ai_base_url")
    api_key_env = _str(form, "ai_api_key_env")

    if provider_mode == "lm_studio":
        config.ai.provider = AIProvider.OPENAI
        config.ai.model = model or LM_STUDIO_DEFAULT_MODEL
        config.ai.base_url = base_url or LM_STUDIO_DEFAULT_BASE_URL
        config.ai.api_key_env = None
        config.ai.codex_extra_args = []
        return

    if provider_mode == AIProvider.CODEX_CLI.value:
        config.ai.provider = AIProvider.CODEX_CLI
        config.ai.model = model or "codex-cli"
        config.ai.base_url = None
        config.ai.api_key_env = None
        effort = _str(form, "codex_reasoning_effort") or "medium"
        if effort not in CODEX_REASONING_EFFORTS:
            raise ConfigFormError(f"Unsupported Codex reasoning effort: {effort}")
        config.ai.codex_extra_args = ["-c", f'model_reasoning_effort="{effort}"']
        return

    provider = AIProvider(provider_mode)
    config.ai.provider = provider
    if model:
        config.ai.model = model
    config.ai.base_url = base_url or BASE_URL_DEFAULTS.get(provider_mode) or None
    config.ai.api_key_env = api_key_env or API_KEY_ENV_DEFAULTS.get(provider_mode)
    config.ai.codex_extra_args = []


def _checked(form: Any, key: str) -> bool:
    return key in form and str(form.get(key)).lower() not in {"", "0", "false", "off", "none"}


def _str(form: Any, key: str) -> str:
    value = form.get(key, "")
    return str(value).strip()


def _int(
    form: Any,
    key: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = _str(form, key)
    if not value:
        number = default
    else:
        try:
            number = int(value)
        except ValueError as exc:
            raise ConfigFormError(f"{key} must be an integer.") from exc
    if minimum is not None and number < minimum:
        raise ConfigFormError(f"{key} must be at least {minimum}.")
    if maximum is not None and number > maximum:
        raise ConfigFormError(f"{key} must be at most {maximum}.")
    return number


def _float(
    form: Any,
    key: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = _str(form, key)
    try:
        number = float(value)
    except ValueError as exc:
        raise ConfigFormError(f"{key} must be a number.") from exc
    if minimum is not None and number < minimum:
        raise ConfigFormError(f"{key} must be at least {minimum}.")
    if maximum is not None and number > maximum:
        raise ConfigFormError(f"{key} must be at most {maximum}.")
    return number


def _csv(form: Any, key: str, required: bool) -> list[str]:
    values = []
    for raw in _list(form, key):
        values.extend(part.strip() for part in str(raw).split(",") if part.strip())
    if required and not values:
        raise ConfigFormError(f"{key} must include at least one value.")
    return values


def _choices(form: Any, key: str, allowed: Iterable[str], required: bool) -> list[str]:
    allowed_set = set(allowed)
    values = [value for value in _list(form, key) if value in allowed_set]
    if required and not values:
        raise ConfigFormError(f"{key} must include at least one valid option.")
    return values


def _list(form: Any, key: str) -> list[str]:
    if hasattr(form, "getlist"):
        return [str(value) for value in form.getlist(key)]
    value = form.get(key, [])
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if value == "":
        return []
    return [str(value)]
