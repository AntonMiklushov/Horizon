"""Core data models for Horizon Brief."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class StrictModel(BaseModel):
    """Base model for runtime configuration with typo detection."""

    model_config = ConfigDict(extra="forbid")


class SourceType(str, Enum):
    """Supported information source types."""
    GITHUB = "github"
    HACKERNEWS = "hackernews"
    RSS = "rss"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    TWITTER = "twitter"


class ContentItem(BaseModel):
    """Unified content item model from any source."""

    id: str  # Format: {source}:{subtype}:{native_id}
    source_type: SourceType
    title: str
    url: HttpUrl
    content: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # AI analysis results
    ai_score: Optional[float] = None  # 0-10 importance score
    ai_reason: Optional[str] = None
    ai_summary: Optional[str] = None
    ai_tags: List[str] = Field(default_factory=list)


class AIProvider(str, Enum):
    """Supported AI providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    ALI = "ali"
    GEMINI = "gemini"
    DOUBAO = "doubao"
    MINIMAX = "minimax"
    CODEX_CLI = "codex_cli"


class AIConfig(StrictModel):
    """AI client configuration."""

    provider: AIProvider
    model: str
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=256, le=200000)
    throttle_sec: float = 0.0
    languages: List[str] = Field(default_factory=lambda: ["en"])
    codex_command: str = "codex"
    codex_timeout_sec: int = Field(default=180, ge=30, le=3600)
    codex_extra_args: List[str] = Field(default_factory=list)
    codex_use_output_last_message: bool = True
    codex_use_json: bool = False


class GitHubSourceConfig(StrictModel):
    """GitHub source configuration."""

    type: Literal["user_events", "repo_releases"]
    username: Optional[str] = None
    owner: Optional[str] = None
    repo: Optional[str] = None
    enabled: bool = True


class HackerNewsConfig(StrictModel):
    """Hacker News configuration."""

    enabled: bool = True
    fetch_top_stories: int = Field(default=30, ge=1, le=500)
    min_score: int = Field(default=100, ge=0)
    story_lists: List[Literal["top", "new", "best", "ask", "show", "job"]] = Field(default_factory=lambda: ["top"])
    include_jobs: bool = False


class RSSSourceConfig(StrictModel):
    """RSS feed source configuration."""

    name: str
    url: HttpUrl
    enabled: bool = True
    category: Optional[str] = None
    undated_policy: Literal["drop", "include_with_low_freshness", "fetched_at"] = "drop"


class RedditSubredditConfig(StrictModel):
    """Configuration for monitoring a specific subreddit."""
    subreddit: str
    enabled: bool = True
    sort: Literal["hot", "new", "top", "rising", "controversial"] = "hot"
    time_filter: Literal["hour", "day", "week", "month", "year", "all"] = "day"
    fetch_limit: int = Field(default=25, ge=1, le=100)
    min_score: int = Field(default=10, ge=0)


class RedditUserConfig(StrictModel):
    """Configuration for monitoring a specific Reddit user."""
    username: str               # without u/ prefix
    enabled: bool = True
    sort: Literal["new", "hot", "top", "controversial"] = "new"
    fetch_limit: int = Field(default=10, ge=1, le=100)


class RedditConfig(StrictModel):
    """Reddit source configuration."""
    enabled: bool = True
    subreddits: List[RedditSubredditConfig] = Field(default_factory=list)
    users: List[RedditUserConfig] = Field(default_factory=list)
    fetch_comments: int = Field(default=5, ge=0, le=20)
    user_agent: str = "HorizonBrief/0.1 (+https://github.com/AntonMiklushov/Horizon; configure REDDIT_USER_AGENT)"
    user_agent_env: str = "REDDIT_USER_AGENT"
    client_id_env: Optional[str] = "REDDIT_CLIENT_ID"
    client_secret_env: Optional[str] = "REDDIT_CLIENT_SECRET"


class TelegramChannelConfig(StrictModel):
    """Configuration for monitoring a specific Telegram channel."""
    channel: str            # channel username, e.g. "zaihuapd"
    enabled: bool = True
    fetch_limit: int = Field(default=20, ge=1, le=100)


class TelegramConfig(StrictModel):
    """Telegram source configuration."""
    enabled: bool = True
    channels: List[TelegramChannelConfig] = Field(default_factory=list)


class TwitterConfig(StrictModel):
    """Twitter source configuration via Apify."""
    enabled: bool = True
    apify_token_env: str = "APIFY_TOKEN"
    actor_id: str = "altimis~scweet"
    users: List[str] = Field(default_factory=list)
    fetch_limit: int = Field(default=10, ge=1, le=500)
    fetch_reply_text: bool = False
    max_replies_per_tweet: int = Field(default=3, ge=0, le=50)
    max_tweets_to_expand: int = Field(default=10, ge=0, le=100)
    reply_min_likes: int = Field(default=0, ge=0)


class SourcesConfig(StrictModel):
    """All sources configuration."""

    github: List[GitHubSourceConfig] = Field(default_factory=list)
    hackernews: HackerNewsConfig = Field(default_factory=HackerNewsConfig)
    rss: List[RSSSourceConfig] = Field(default_factory=list)
    reddit: RedditConfig = Field(default_factory=RedditConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    twitter: Optional[TwitterConfig] = None


class WebhookConfig(StrictModel):
    """Webhook notification configuration."""

    url_env: Optional[str] = None          # Environment variable name containing the webhook URL
    request_body: Optional[Union[str, dict, list]] = None  # POST body: real JSON object or string with #{key} placeholders; if empty, will use GET
    headers: Optional[str] = None          # Custom headers, "Key: Value" per line
    delivery: Literal["summary", "summary_and_items"] = "summary"
    overview_position: Literal["first", "last"] = "first"
    platform: Literal["generic", "feishu", "lark", "dingtalk", "slack", "discord"] = "generic"
    layout: Literal["markdown", "collapsible"] = "markdown"
    fallback_layout: Literal["markdown"] = "markdown"
    languages: Optional[List[str]] = None  # Optional language filter for webhook delivery; defaults to all AI languages
    enabled: bool = False


class EmailConfig(StrictModel):
    """Email configuration for updates/subscriptions."""
    imap_server: str
    imap_port: int = Field(default=993, ge=1, le=65535)
    smtp_server: str
    smtp_port: int = Field(default=465, ge=1, le=65535)
    email_address: str
    password_env: str = "EMAIL_PASSWORD"
    sender_name: str = "Horizon Brief Daily"
    subscribe_keyword: str = "SUBSCRIBE"
    unsubscribe_keyword: str = "UNSUBSCRIBE"
    enabled: bool = False


class FilteringConfig(StrictModel):
    """Content filtering configuration."""

    ai_score_threshold: float = Field(default=7.0, ge=0.0, le=10.0)
    time_window_hours: int = Field(default=24, ge=1, le=24 * 30)
    max_items_per_source: int = Field(default=5, ge=1, le=100)


class PersonalBriefingCriticConfig(StrictModel):
    enabled: bool = True
    auto_revise_once: bool = True


class PersonalBriefingConfig(StrictModel):
    enabled: bool = False
    generate_standard_summaries: bool = False
    language: str = "ru"
    timezone: str = "Europe/Paris"
    daily_empty_allowed: bool = True
    min_importance: float = Field(default=7.0, ge=0.0, le=10.0)
    min_importance_priority_topics: float = Field(default=6.5, ge=0.0, le=10.0)
    require_dates: bool = True
    source_policy_file: str = "data/config.personal-news.example.json"
    priority_topics: List[str] = Field(
        default_factory=lambda: ["russia", "moscow", "world_economy", "tech_ai", "open_source", "big_tech", "science"]
    )
    critic_pass: PersonalBriefingCriticConfig = Field(default_factory=PersonalBriefingCriticConfig)


class RenderingConfig(StrictModel):
    """Digest rendering configuration."""

    template_dir: Optional[str] = None
    output_formats: List[Literal["markdown", "html", "email_html"]] = Field(default_factory=lambda: ["markdown", "html"])
    inline_email_css: bool = True


class PublishingConfig(StrictModel):
    """Static publishing configuration."""

    enabled: bool = True
    docs_dir: str = "docs"
    site_base_url: Optional[str] = None


class Config(StrictModel):
    """Main configuration model."""

    version: str = "1.0"
    ai: AIConfig
    sources: SourcesConfig
    filtering: FilteringConfig
    email: Optional[EmailConfig] = None
    webhook: Optional[WebhookConfig] = None
    personal_briefing: PersonalBriefingConfig = Field(default_factory=PersonalBriefingConfig)
    rendering: RenderingConfig = Field(default_factory=RenderingConfig)
    publishing: PublishingConfig = Field(default_factory=PublishingConfig)
    codex_cli_provider_example: Optional[AIConfig] = None
    notes: Dict[str, str] = Field(default_factory=dict)
