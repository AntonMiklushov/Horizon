"""Main orchestrator coordinating the entire workflow."""

import asyncio
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, List, Dict
import httpx
from pydantic import ValidationError

from .console import make_console
from .models import Config, ContentItem
from .storage.manager import StorageManager
from .services.email import EmailManager
from .services.webhook import WebhookNotifier
from .scrapers.github import GitHubScraper
from .scrapers.hackernews import HackerNewsScraper
from .scrapers.rss import RSSScraper
from .scrapers.reddit import RedditScraper
from .scrapers.telegram import TelegramScraper
from .scrapers.twitter import TwitterScraper
from .ai.client import create_ai_client
from .ai.analyzer import ContentAnalyzer
from .ai.summarizer import DailySummarizer
from .ai.enricher import ContentEnricher
from .horizon_ext.personal import (
    drop_items_flagged_by_critic,
    EvidenceChecker,
    PersonalBriefingRenderer,
    SourcePolicyClassifier,
    load_source_policy,
    prefilter_personal_candidates,
    run_briefing_critic,
    select_personal_important_items,
)
from .ai.schemas import TopicDedupResult
from .ai.tokens import get_usage_snapshot
from .horizon_ext.pipeline import apply_source_diversity, build_digest_document, build_selection_traces, canonicalize_url
from .horizon_ext.rendering import DigestRenderer


_SENSITIVE_VERBOSE_KEYS = {
    "api_key",
    "api_key_env",
    "authorization",
    "email",
    "email_address",
    "headers",
    "password",
    "request_body",
    "secret",
    "token",
    "url_env",
    "webhook",
}


class _VerboseReporter:
    """Small opt-in console reporter for stage-level diagnostics."""

    def __init__(self, console, enabled: bool = False):
        self.console = console
        self.enabled = enabled

    def start(self, stage: str, **fields: Any) -> float:
        started_at = time.monotonic()
        if self.enabled:
            self.console.print(f"[dim]VERBOSE START {stage}{self._fields(fields)}[/dim]")
        return started_at

    def end(self, stage: str, started_at: float | None = None, **fields: Any) -> None:
        if not self.enabled:
            return
        payload = dict(fields)
        if started_at is not None:
            payload["duration_sec"] = f"{time.monotonic() - started_at:.2f}"
        self.console.print(f"[dim]VERBOSE END {stage}{self._fields(payload)}[/dim]")

    def event(self, name: str, **fields: Any) -> None:
        if self.enabled:
            self.console.print(f"[dim]VERBOSE {name}{self._fields(fields)}[/dim]")

    @classmethod
    def _fields(cls, fields: Dict[str, Any]) -> str:
        if not fields:
            return ""
        rendered = []
        for key, value in fields.items():
            rendered.append(f" {key}={cls._sanitize(key, value)}")
        return "".join(rendered)

    @classmethod
    def _sanitize(cls, key: str, value: Any) -> str:
        key_l = str(key).lower()
        if any(token in key_l for token in _SENSITIVE_VERBOSE_KEYS):
            return "<redacted>"
        if isinstance(value, dict):
            return "{" + ", ".join(f"{k}:{cls._sanitize(str(k), v)}" for k, v in value.items()) + "}"
        if isinstance(value, (list, tuple, set)):
            return "[" + ", ".join(cls._sanitize(key, v) for v in value) + "]"
        text = str(value).replace("\n", " ")
        if len(text) > 160:
            text = text[:157] + "..."
        return text


class HorizonOrchestrator:
    """Orchestrates the complete workflow for content aggregation and analysis."""

    def __init__(self, config: Config, storage: StorageManager, verbose: bool = False):
        """Initialize orchestrator.

        Args:
            config: Application configuration
            storage: Storage manager
        """
        self.config = config
        self.storage = storage
        self.console = make_console()
        self.verbose = verbose
        self.verbose_reporter = _VerboseReporter(self.console, enabled=verbose)
        self.email_manager = EmailManager(config.email, console=self.console) if config.email else None
        self.webhook_notifier = (
            WebhookNotifier(config.webhook, console=self.console)
            if config.webhook and config.webhook.enabled
            else None
        )
        self.personal_policy = None
        self.personal_classifier = None
        if self.config.personal_briefing.enabled:
            policy_path = self.storage.resolve_runtime_path(self.config.personal_briefing.source_policy_file)
            policy, warning = load_source_policy(str(policy_path))
            self.personal_policy = policy
            self.personal_classifier = SourcePolicyClassifier(policy)
            if warning:
                self.console.print(f"[yellow]⚠️ {warning}[/yellow]")

    async def run(self, force_hours: int = None) -> None:
        """Execute the complete workflow.

        Args:
            force_hours: Optional override for time window in hours
        """
        self.console.print("[bold cyan]🌅 Horizon Brief - Starting aggregation...[/bold cyan]\n")
        run_timer = self.verbose_reporter.start(
            "run",
            provider=self.config.ai.provider.value,
            model=self.config.ai.model,
            languages=",".join(self.config.ai.languages),
            personal=self.config.personal_briefing.enabled,
            force_hours=force_hours or "config",
        )

        # Check email subscriptions if configured
        if self.email_manager and self.config.email and self.config.email.enabled:
            email_timer = self.verbose_reporter.start("email.subscriptions")
            self.console.print("📧 Checking for new email subscriptions...")
            self.email_manager.check_subscriptions(self.storage)
            self.verbose_reporter.end("email.subscriptions", email_timer)

        try:
            # 1. Determine time window
            since = self._determine_time_window(force_hours)
            self.console.print(f"📅 Fetching content since: {since.strftime('%Y-%m-%d %H:%M:%S')}\n")

            # 2. Fetch content from all sources
            fetch_timer = self.verbose_reporter.start("fetch", since=since.isoformat())
            all_items = await self.fetch_all_sources(since)
            source_counts = Counter(item.source_type.value for item in all_items)
            self.verbose_reporter.end(
                "fetch",
                fetch_timer,
                total_items=len(all_items),
                source_counts=dict(sorted(source_counts.items())),
            )
            if self.config.personal_briefing.enabled:
                classify_timer = self.verbose_reporter.start("personal.classify", items=len(all_items))
                self._classify_personal_source_metadata(all_items)
                role_counts = Counter(item.metadata.get("source_role", "unknown") for item in all_items)
                self.verbose_reporter.end(
                    "personal.classify",
                    classify_timer,
                    role_counts=dict(sorted(role_counts.items())),
                )
            self.console.print(f"📥 Fetched {len(all_items)} items from all sources\n")
            if not all_items and not self.config.personal_briefing.enabled:
                self.console.print("[yellow]No new content found. Exiting.[/yellow]")
                self.verbose_reporter.end("run", run_timer, status="empty")
                return


            tracked_excluded = []
            candidates = all_items
            if self.config.personal_briefing.enabled:
                prefilter_timer = self.verbose_reporter.start("personal.prefilter", input_items=len(all_items))
                candidates, pretracked = self._prefilter_personal_candidates(all_items)
                tracked_excluded.extend(pretracked)
                self.verbose_reporter.end(
                    "personal.prefilter",
                    prefilter_timer,
                    kept=len(candidates),
                    excluded=len(pretracked),
                    excluded_reasons=self._reason_counts(pretracked),
                )

            # 3. Merge cross-source duplicates (same URL from different sources)
            merge_timer = self.verbose_reporter.start("dedup.url", input_items=len(candidates))
            merged_items = self.merge_cross_source_duplicates(candidates)
            self.verbose_reporter.end(
                "dedup.url",
                merge_timer,
                output_items=len(merged_items),
                merged=len(candidates) - len(merged_items),
            )
            if len(merged_items) < len(candidates):
                self.console.print(
                    f"🔗 Merged {len(candidates) - len(merged_items)} cross-source duplicates "
                    f"→ {len(merged_items)} unique items\n"
                )

            # 4. Analyze with AI
            analysis_timer = self.verbose_reporter.start("llm.analysis", input_items=len(merged_items))
            analyzed_items = await self._analyze_content(merged_items) if merged_items else []
            self.verbose_reporter.end("llm.analysis", analysis_timer, output_items=len(analyzed_items))
            self.console.print(f"🤖 Analyzed {len(analyzed_items)} items with AI\n")

            filter_timer = self.verbose_reporter.start("filter", input_items=len(analyzed_items))
            if self.config.personal_briefing.enabled:
                checker = EvidenceChecker(self.config.filtering.time_window_hours)
                important_items, posttracked = select_personal_important_items(
                    analyzed_items,
                    checker=checker,
                    min_importance=self.config.personal_briefing.min_importance,
                    min_importance_priority_topics=self.config.personal_briefing.min_importance_priority_topics,
                    require_dates=self.config.personal_briefing.require_dates,
                    priority_topics=self._priority_topics(),
                )
                tracked_excluded.extend(posttracked)
                self.verbose_reporter.end(
                    "filter",
                    filter_timer,
                    kept=len(important_items),
                    excluded=len(posttracked),
                    excluded_reasons=self._reason_counts(posttracked),
                )
            else:
                threshold = self.config.filtering.ai_score_threshold
                important_items = self._filter_standard_important_items(analyzed_items)
                self.verbose_reporter.end(
                    "filter",
                    filter_timer,
                    threshold=threshold,
                    kept=len(important_items),
                    excluded=len(analyzed_items) - len(important_items),
                )
                self.console.print(f"⭐️ {len(important_items)} items scored ≥ {threshold}\n")
            important_items.sort(key=lambda x: x.ai_score or 0, reverse=True)

            # 5.5 Semantic deduplication: drop items covering the same topic
            topic_timer = self.verbose_reporter.start("dedup.topic", input_items=len(important_items))
            deduped_items = await self.merge_topic_duplicates(important_items)
            self.verbose_reporter.end(
                "dedup.topic",
                topic_timer,
                output_items=len(deduped_items),
                removed=len(important_items) - len(deduped_items),
            )
            if len(deduped_items) < len(important_items):
                self.console.print(
                    f"🧹 Removed {len(important_items) - len(deduped_items)} topic duplicates "
                    f"→ {len(deduped_items)} unique items\n"
                )
            important_items = deduped_items
            if self.config.personal_briefing.enabled and important_items:
                reaudit_timer = self.verbose_reporter.start("personal.reaudit", input_items=len(important_items))
                checker = EvidenceChecker(self.config.filtering.time_window_hours)
                important_items, retracked = select_personal_important_items(
                    important_items,
                    checker=checker,
                    min_importance=self.config.personal_briefing.min_importance,
                    min_importance_priority_topics=self.config.personal_briefing.min_importance_priority_topics,
                    require_dates=self.config.personal_briefing.require_dates,
                    priority_topics=self._priority_topics(),
                )
                tracked_excluded.extend(retracked)
                self.verbose_reporter.end(
                    "personal.reaudit",
                    reaudit_timer,
                    kept=len(important_items),
                    excluded=len(retracked),
                    excluded_reasons=self._reason_counts(retracked),
                )

            # 5.6 Optional second-stage Twitter reply expansion + targeted re-analysis
            if not self.config.personal_briefing.enabled:
                twitter_timer = self.verbose_reporter.start("twitter.replies", candidate_items=len(important_items))
                await self._expand_twitter_discussion(important_items)
                before_refilter = len(important_items)
                important_items = self._filter_standard_important_items(important_items)
                important_items.sort(key=lambda x: x.ai_score or 0, reverse=True)
                removed_after_reanalysis = before_refilter - len(important_items)
                if removed_after_reanalysis:
                    self.console.print(
                        f"â­ï¸ Removed {removed_after_reanalysis} items after Twitter reply re-analysis\n"
                    )
                self.verbose_reporter.event(
                    "twitter.replies.refilter",
                    kept=len(important_items),
                    removed=removed_after_reanalysis,
                    threshold=self.config.filtering.ai_score_threshold,
                )
                self.verbose_reporter.end("twitter.replies", twitter_timer)
            # In personal mode we skip reply expansion for now because reply expansion
            # can mutate claim metadata after evidence checks.

            diversity_timer = self.verbose_reporter.start("selection.diversity", input_items=len(important_items))
            important_items, diversity_excluded = apply_source_diversity(
                important_items,
                max_items_per_source=self.config.filtering.max_items_per_source,
            )
            tracked_excluded.extend(diversity_excluded)
            self.verbose_reporter.end(
                "selection.diversity",
                diversity_timer,
                kept=len(important_items),
                excluded=len(diversity_excluded),
                max_per_source=self.config.filtering.max_items_per_source,
            )

            # Show per-sub-source selection breakdown
            selected_counts: Dict[str, int] = defaultdict(int)
            for item in important_items:
                key = f"{item.source_type.value}/{self._sub_source_label(item)}"
                selected_counts[key] += 1
            for source_key, count in sorted(selected_counts.items()):
                self.console.print(f"      • {source_key}: {count}")
            self.console.print("")

            # 6. Search related stories + enrich with background knowledge (2nd AI pass)
            enrichment_timer = self.verbose_reporter.start("llm.enrichment", input_items=len(important_items))
            await self._enrich_important_items(important_items)
            self.verbose_reporter.end("llm.enrichment", enrichment_timer, output_items=len(important_items))

            # 7. Generate and save daily summaries
            summary_timer = self.verbose_reporter.start("summaries")
            saved_summaries = 0
            tz = ZoneInfo(self.config.personal_briefing.timezone if self.config.personal_briefing.enabled else "UTC")
            today = datetime.now(tz).strftime("%Y-%m-%d")
            langs = list(self.config.ai.languages)
            if self.config.personal_briefing.enabled and not self.config.personal_briefing.generate_standard_summaries:
                langs = [self.config.personal_briefing.language]
            elif self.config.personal_briefing.enabled and self.config.personal_briefing.language not in langs:
                langs.append(self.config.personal_briefing.language)
            self.verbose_reporter.event("summaries.languages", languages=",".join(langs), date=today)
            for lang in langs:
                summarizer = None
                summary_items = list(important_items)
                self.verbose_reporter.event("summary.render", language=lang, items=len(important_items))
                if self.config.personal_briefing.enabled and lang == self.config.personal_briefing.language:
                    personal_renderer = PersonalBriefingRenderer()
                    summary = personal_renderer.render(today, summary_items, tracked=tracked_excluded[:10])
                    if self.config.personal_briefing.critic_pass.enabled:
                        critic_timer = self.verbose_reporter.start("personal.critic", language=lang, items=len(summary_items))
                        critic = run_briefing_critic(summary, summary_items)
                        if (not critic.passed) and self.config.personal_briefing.critic_pass.auto_revise_once:
                            revised_items = drop_items_flagged_by_critic(summary_items, critic)
                            if len(revised_items) < len(summary_items):
                                self.verbose_reporter.event(
                                    "personal.critic.revise",
                                    removed=len(summary_items) - len(revised_items),
                                )
                                summary_items = revised_items
                                summary = personal_renderer.render(today, summary_items, tracked=tracked_excluded[:10])
                                critic = run_briefing_critic(summary, summary_items)
                        self.verbose_reporter.end(
                            "personal.critic",
                            critic_timer,
                            passed=critic.passed,
                            critical=len(critic.critical_issues),
                            minor=len(critic.minor_issues),
                        )
                        if not critic.passed:
                            failed = summary + "\n\n## Предупреждения аудита\n" + "\n".join(
                                [f"- {x}" for x in critic.critical_issues]
                            )
                            failed_path = self.storage.save_daily_summary(
                                today,
                                failed,
                                language=f"{lang}-audit-failed",
                            )
                            self.console.print(
                                f"[red]Personal briefing failed audit; saved audit artifact to: {failed_path}[/red]\n"
                            )
                            self.verbose_reporter.event("summary.audit_failed", language=lang, artifact=failed_path)
                            continue
                else:
                    summarizer = DailySummarizer()
                    summary = await summarizer.generate_summary(important_items, today, len(all_items), language=lang)

                renderer = DigestRenderer.from_config(self.config, self.storage)
                digest = build_digest_document(
                    date=today,
                    language=lang,
                    title=self._summary_title(today, lang),
                    items=summary_items,
                    total_fetched=len(all_items),
                    markdown=summary,
                    selection_traces=build_selection_traces(summary_items, tracked_excluded),
                    tracked_excluded=tracked_excluded[:10],
                )

                # Save to data/summaries/
                summary_path = self.storage.save_daily_summary(today, summary, language=lang)
                saved_summaries += 1
                self.verbose_reporter.event("summary.saved", language=lang, path=summary_path)
                self.console.print(f"💾 Saved {lang.upper()} summary to: {summary_path}\n")

                if "html" in self.config.rendering.output_formats:
                    html_path = self.storage.save_summary_artifact(
                        today,
                        renderer.render_html(digest),
                        language=lang,
                        extension="html",
                    )
                    self.verbose_reporter.event("summary.html_saved", language=lang, path=html_path)

                # Copy to docs/ for GitHub Pages
                if self.config.publishing.enabled:
                    try:
                        dest_path = self._publish_jekyll_post(today, lang, renderer.render_jekyll_post(digest))
                        self.console.print(f"📄 Copied {lang.upper()} summary to GitHub Pages: {dest_path}\n")
                        self.verbose_reporter.event("summary.docs", language=lang, path=dest_path, status="saved")
                    except Exception as e:
                        self.console.print(f"[yellow]⚠️  Failed to copy {lang.upper()} summary to docs/: {e}[/yellow]\n")
                        self.verbose_reporter.event("summary.docs", language=lang, status="failed", error=type(e).__name__)
                else:
                    self.verbose_reporter.event("summary.docs", language=lang, status="disabled")

                # Send email if configured
                if self.email_manager and self.config.email and self.config.email.enabled:
                    self.console.print(f"📧 Sending {lang.upper()} email summary...")
                    subscribers = self.storage.load_subscribers()
                    self.verbose_reporter.event("summary.email", language=lang, subscribers=len(subscribers))
                    subject = f"Horizon Brief Summary ({lang.upper()}) - {today}"
                    self.email_manager.send_daily_summary(
                        summary,
                        subject,
                        subscribers,
                        html_body=renderer.render_email_html(digest),
                        text_body=renderer.render_plain_text(digest),
                    )
                else:
                    self.verbose_reporter.event("summary.email", language=lang, status="disabled")

                # Send webhook notification if configured
                if self.webhook_notifier:
                    if summarizer is None:
                        summarizer = DailySummarizer()
                    self.verbose_reporter.event("summary.webhook", language=lang, status="sending")
                    await self.webhook_notifier.send_daily_summary(
                        summary=summary,
                        important_items=important_items,
                        all_items_count=len(all_items),
                        date=today,
                        lang=lang,
                        summarizer=summarizer,
                    )
                else:
                    self.verbose_reporter.event("summary.webhook", language=lang, status="disabled")

            self.verbose_reporter.end("summaries", summary_timer, saved=saved_summaries)
            self.console.print("[bold green]✅ Horizon Brief completed successfully![/bold green]")
            usage = get_usage_snapshot()
            if usage.total_tokens > 0:
                self.console.print(
                    f"\n🧮 Token usage this run: "
                    f"{usage.total_tokens} tokens "
                    f"(input: {usage.total_input_tokens}, output: {usage.total_output_tokens})"
                )
                for provider, u in sorted(usage.per_provider.items()):
                    if u.total <= 0:
                        continue
                    self.console.print(
                        f"   • {provider}: {u.total} tokens "
                        f"(in: {u.input_tokens}, out: {u.output_tokens})"
                    )
            self.verbose_reporter.end("run", run_timer, status="ok")

        except Exception as e:
            self.verbose_reporter.end("run", run_timer, status="failed", error=type(e).__name__)
            self.console.print(f"[bold red]❌ Error: {e}[/bold red]")

            # Send webhook failure notification if configured
            if self.webhook_notifier:
                await self.webhook_notifier.send_failure(
                    date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    error_message=str(e),
                )

            raise

    def _determine_time_window(self, force_hours: int = None) -> datetime:
        if force_hours:
            since = datetime.now(timezone.utc) - timedelta(hours=force_hours)
        else:
            hours = self.config.filtering.time_window_hours
            since = datetime.now(timezone.utc) - timedelta(hours=hours)
        return since

    async def fetch_all_sources(self, since: datetime) -> List[ContentItem]:
        """Fetch content from all configured sources.

        This is a stable stage entry point for integrations such as MCP.

        Args:
            since: Fetch items published after this time

        Returns:
            List[ContentItem]: All fetched items
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = []
            scheduled_sources = []

            # GitHub sources
            if self.config.sources.github:
                github_scraper = GitHubScraper(self.config.sources.github, client)
                tasks.append(self._fetch_with_progress("GitHub", github_scraper, since))
                scheduled_sources.append("GitHub")

            # Hacker News
            if self.config.sources.hackernews.enabled:
                hn_scraper = HackerNewsScraper(self.config.sources.hackernews, client)
                tasks.append(self._fetch_with_progress("Hacker News", hn_scraper, since))
                scheduled_sources.append("Hacker News")

            # RSS feeds
            if self.config.sources.rss:
                rss_scraper = RSSScraper(self.config.sources.rss, client)
                tasks.append(self._fetch_with_progress("RSS Feeds", rss_scraper, since))
                scheduled_sources.append("RSS Feeds")

            # Reddit
            if self.config.sources.reddit.enabled:
                reddit_scraper = RedditScraper(self.config.sources.reddit, client)
                tasks.append(self._fetch_with_progress("Reddit", reddit_scraper, since))
                scheduled_sources.append("Reddit")

            # Telegram
            if self.config.sources.telegram.enabled:
                telegram_scraper = TelegramScraper(self.config.sources.telegram, client)
                tasks.append(self._fetch_with_progress("Telegram", telegram_scraper, since))
                scheduled_sources.append("Telegram")

            # Twitter
            if self.config.sources.twitter and self.config.sources.twitter.enabled:
                twitter_scraper = TwitterScraper(self.config.sources.twitter, client)
                tasks.append(self._fetch_with_progress("Twitter", twitter_scraper, since))
                scheduled_sources.append("Twitter")

            # Fetch all concurrently
            self.verbose_reporter.event(
                "sources.scheduled",
                count=len(scheduled_sources),
                sources=",".join(scheduled_sources) or "none",
            )
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Flatten results
            all_items = []
            error_count = 0
            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                    self.console.print(f"[red]Error fetching source: {result}[/red]")
                elif isinstance(result, list):
                    all_items.extend(result)

            self.verbose_reporter.event(
                "sources.aggregated",
                total_items=len(all_items),
                errors=error_count,
            )
            if tasks and error_count == len(tasks):
                failed_sources = ", ".join(scheduled_sources)
                raise RuntimeError(f"All scheduled sources failed: {failed_sources}")
            return all_items

    async def _fetch_with_progress(self, name: str, scraper, since: datetime) -> List[ContentItem]:
        """Fetch from a scraper with progress indication.

        Args:
            name: Source name for display
            scraper: Scraper instance
            since: Fetch items after this time

        Returns:
            List[ContentItem]: Fetched items
        """
        source_timer = self.verbose_reporter.start(f"source.{name}", since=since.isoformat())
        self.console.print(f"🔍 Fetching from {name}...")
        items = await scraper.fetch(since)
        self.console.print(f"   Found {len(items)} items from {name}")

        # Show per-sub-source breakdown when there are multiple sub-sources
        sub_counts: Dict[str, int] = defaultdict(int)
        for item in items:
            sub_counts[self._sub_source_label(item)] += 1
        if self.personal_classifier:
            self._classify_personal_source_metadata(items)
        if len(sub_counts) > 1:
            for sub, count in sorted(sub_counts.items()):
                self.console.print(f"      • {sub}: {count}")

        self.verbose_reporter.end(
            f"source.{name}",
            source_timer,
            items=len(items),
            sub_sources=dict(sorted(sub_counts.items())),
        )
        return items

    def _classify_personal_source_metadata(self, items: List[ContentItem]) -> None:
        if not self.personal_classifier:
            return
        for item in items:
            item.metadata.setdefault("source_name", item.metadata.get("feed_name") or item.source_type.value)
            item.metadata.setdefault("source_url", str(item.url))
            item.metadata.setdefault("source_role", "unclassified")
            item.metadata.setdefault("source_reliability_tier", "unknown")
            item.metadata.setdefault("source_policy_notes", "")
            item.metadata.setdefault("publication_date", item.published_at.isoformat() if item.published_at else None)
            item.metadata.setdefault("fetched_at", item.fetched_at.isoformat())
            item.metadata.setdefault("original_language", None)
            item.metadata.update(self.personal_classifier.classify(item))

    @staticmethod
    def _sub_source_label(item: ContentItem) -> str:
        """Return a human-readable sub-source label for an item."""
        meta = item.metadata
        if meta.get("subreddit"):
            return f"r/{meta['subreddit']}"
        if meta.get("feed_name"):
            return meta["feed_name"]
        if meta.get("channel"):
            return f"@{meta['channel']}"
        if meta.get("repo"):
            return meta["repo"]
        return item.author or "unknown"

    @staticmethod
    def _reason_counts(tracked: List[Dict[str, str]]) -> Dict[str, int]:
        return dict(sorted(Counter(t.get("reason", "unknown") for t in tracked).items()))

    def _filter_standard_important_items(self, items: List[ContentItem]) -> List[ContentItem]:
        threshold = self.config.filtering.ai_score_threshold
        return [item for item in items if item.ai_score is not None and item.ai_score >= threshold]

    def _prefilter_personal_candidates(self, items: List[ContentItem]) -> tuple[List[ContentItem], List[Dict[str, str]]]:
        return prefilter_personal_candidates(items, self.personal_policy)

    def _priority_topics(self) -> set[str]:
        return {str(topic) for topic in self.config.personal_briefing.priority_topics}

    def _summary_title(self, date: str, language: str) -> str:
        if self.config.personal_briefing.enabled and language == self.config.personal_briefing.language:
            return f"Вечерняя сводка - {date}"
        return f"Horizon Brief Summary: {date} ({language.upper()})"

    def _publish_jekyll_post(self, date: str, language: str, content: str):
        posts_dir = self.storage.resolve_runtime_path(self.config.publishing.docs_dir) / "_posts"
        posts_dir.mkdir(parents=True, exist_ok=True)
        dest_path = posts_dir / f"{date}-summary-{language}.md"
        dest_path.write_text(content, encoding="utf-8")
        return dest_path

    def merge_cross_source_duplicates(self, items: List[ContentItem]) -> List[ContentItem]:
        """Merge items that point to the same URL from different sources.

        This is a stable stage helper for integrations such as MCP.

        Keeps the item with the richest content and combines metadata.

        Args:
            items: Items to deduplicate

        Returns:
            List[ContentItem]: Deduplicated items
        """
        # Group by normalized URL
        url_groups: Dict[str, List[ContentItem]] = {}
        for item in items:
            key = canonicalize_url(str(item.url))
            url_groups.setdefault(key, []).append(item)

        merged = []
        for key, group in url_groups.items():
            if len(group) == 1:
                merged.append(group[0])
                continue

            # Pick the item with the richest content as primary
            primary = max(group, key=lambda x: len(x.content or ""))

            # Merge metadata and source info from other items
            all_sources = set()
            supporting = list(primary.metadata.get("supporting_sources", []))
            for item in group:
                all_sources.add(item.source_type.value)
                # Merge metadata (engagement, discussion, etc.)
                for mk, mv in item.metadata.items():
                    if mk not in primary.metadata or not primary.metadata[mk]:
                        primary.metadata[mk] = mv

                if item is not primary:
                    supporting.append({
                        "source_name": item.metadata.get("source_name", item.source_type.value),
                        "source_url": str(item.url),
                        "source_role": item.metadata.get("source_role", "unclassified"),
                        "source_reliability_tier": item.metadata.get("source_reliability_tier", "unknown"),
                        "publication_date": item.metadata.get("publication_date"),
                        "title": item.title,
                    })

                # Append content (e.g., comments from another source)
                if item is not primary and item.content:
                    if primary.content and item.content not in primary.content:
                        primary.content = (primary.content or "") + f"\n\n--- From {item.source_type.value} ---\n" + item.content

            primary.metadata["merged_sources"] = list(all_sources)
            if supporting:
                primary.metadata["supporting_sources"] = supporting
            merged.append(primary)

        return merged

    async def merge_topic_duplicates(self, items: List[ContentItem]) -> List[ContentItem]:
        """Merge items covering the same topic using AI semantic deduplication.

        This is a stable stage helper for integrations such as MCP.

        Sends all item titles, tags, and summaries to AI in a single call.
        Items must already be sorted by ai_score descending so that the first
        item in each duplicate group is always the highest-scored one.
        Content (comments) from duplicate items is merged into the primary.

        Falls back to returning items unchanged if the AI call fails.
        """
        if len(items) <= 1:
            self.verbose_reporter.event("llm.topic_dedup", status="skipped", items=len(items))
            return items

        from .ai.prompts import TOPIC_DEDUP_SYSTEM, TOPIC_DEDUP_USER
        from .ai.utils import parse_json_response

        # Build the item list for the prompt
        lines = []
        for i, item in enumerate(items):
            tags = ", ".join(item.ai_tags) if item.ai_tags else "—"
            summary = item.ai_summary or "—"
            lines.append(f"[{i}] {item.title}\n    Tags: {tags}\n    Summary: {summary}")
        items_text = "\n\n".join(lines)

        try:
            self.verbose_reporter.event(
                "llm.topic_dedup",
                status="calling",
                provider=self.config.ai.provider.value,
                model=self.config.ai.model,
                items=len(items),
            )
            ai_client = create_ai_client(self.config.ai)
            response = await ai_client.complete(
                system=TOPIC_DEDUP_SYSTEM,
                user=TOPIC_DEDUP_USER.format(items=items_text),
            )
            result = parse_json_response(response)
            if result is None:
                self.verbose_reporter.event("llm.topic_dedup", status="parse_failed", items=len(items))
                self.console.print("[yellow]  dedup: could not parse AI response, skipping[/yellow]")
                return items

            duplicate_groups = TopicDedupResult.model_validate(result).duplicates
            self.verbose_reporter.event(
                "llm.topic_dedup",
                status="validated",
                groups=len(duplicate_groups),
            )
        except ValidationError as e:
            self.verbose_reporter.event("llm.topic_dedup", status="invalid_schema")
            self.console.print(f"[yellow]  dedup: invalid AI response schema ({e}), skipping[/yellow]")
            return items
        except Exception as e:
            self.verbose_reporter.event("llm.topic_dedup", status="failed", error=type(e).__name__)
            self.console.print(f"[yellow]  dedup: AI call failed ({e}), skipping[/yellow]")
            return items

        if not duplicate_groups:
            self.verbose_reporter.event("llm.topic_dedup", status="no_duplicates")
            return items

        # Build a set of indices to drop (all non-primary duplicates)
        drop_indices: set[int] = set()
        for group in duplicate_groups:
            if not isinstance(group, list) or len(group) < 2:
                continue
            primary_idx = group[0]
            if primary_idx < 0 or primary_idx >= len(items):
                continue
            primary = items[primary_idx]
            for dup_idx in group[1:]:
                if not isinstance(dup_idx, int) or dup_idx < 0 or dup_idx >= len(items):
                    continue
                if dup_idx == primary_idx:
                    continue
                dup = items[dup_idx]
                primary.metadata.setdefault("supporting_sources", [])
                primary.metadata["supporting_sources"].append({
                    "source_name": dup.metadata.get("source_name", dup.source_type.value),
                    "source_url": str(dup.url),
                    "source_role": dup.metadata.get("source_role", "unclassified"),
                    "source_reliability_tier": dup.metadata.get("source_reliability_tier", "unknown"),
                    "publication_date": dup.metadata.get("publication_date"),
                    "title": dup.title,
                })
                # Merge comments/content from the duplicate into the primary
                if dup.content:
                    if not primary.content or dup.content not in primary.content:
                        label = dup.source_type.value
                        primary.content = (primary.content or "") + f"\n\n--- From {label} ---\n{dup.content}"
                self.console.print(
                    f"   [dim]dedup: keep [{primary_idx}] {primary.title}[/dim]\n"
                    f"   [dim]       drop [{dup_idx}] {dup.title}[/dim]"
                )
                drop_indices.add(dup_idx)

        self.verbose_reporter.event(
            "llm.topic_dedup",
            status="merged",
            dropped=len(drop_indices),
            kept=len(items) - len(drop_indices),
        )
        return [item for i, item in enumerate(items) if i not in drop_indices]

    async def _expand_twitter_discussion(self, items: List[ContentItem]) -> None:
        """Second-stage: fetch reply text for important Twitter items and re-analyze.

        Only runs when sources.twitter.fetch_reply_text is True.
        Bounded by max_tweets_to_expand to control cost.
        """
        tw_cfg = self.config.sources.twitter
        if not tw_cfg or not tw_cfg.enabled or not tw_cfg.fetch_reply_text:
            self.verbose_reporter.event("twitter.replies", status="skipped")
            return

        from .models import SourceType

        twitter_items = [
            item for item in items
            if item.source_type == SourceType.TWITTER
        ][:tw_cfg.max_tweets_to_expand]

        if not twitter_items:
            self.verbose_reporter.event("twitter.replies", status="no_twitter_items")
            return

        self.verbose_reporter.event(
            "twitter.replies",
            status="fetching",
            selected=len(twitter_items),
            max_tweets=tw_cfg.max_tweets_to_expand,
        )
        self.console.print(
            f"💬 Fetching reply text for {len(twitter_items)} Twitter items..."
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            scraper = TwitterScraper(tw_cfg, client)
            expanded = []
            for item in twitter_items:
                try:
                    reply_lines = await scraper.fetch_replies_for_item(item)
                    if TwitterScraper.append_discussion_content(item, reply_lines):
                        expanded.append(item)
                        self.console.print(
                            f"   💬 {len(reply_lines)} replies added to: {item.title[:60]}"
                        )
                except Exception as exc:
                    self.console.print(
                        f"   [yellow]⚠️  Reply fetch failed for {item.id}: {exc}[/yellow]"
                    )

        if not expanded:
            self.verbose_reporter.event("twitter.replies", status="no_expanded_items")
            return

        self.console.print(
            f"   Re-analyzing {len(expanded)} Twitter items with reply context...\n"
        )
        self.verbose_reporter.event("twitter.replies", status="reanalyzing", expanded=len(expanded))
        ai_client = create_ai_client(self.config.ai)
        analyzer = ContentAnalyzer(
            ai_client,
            personal_briefing_mode=self.config.personal_briefing.enabled,
            verbose_reporter=self.verbose_reporter,
        )
        await analyzer.analyze_batch(expanded)

    async def _enrich_important_items(self, items: List[ContentItem]) -> None:
        """Enrich items with background knowledge (2nd AI pass).

        For each item that passed the score threshold, call AI to generate
        background knowledge based on the item's actual content.

        Args:
            items: Important items to enrich (modified in-place)
        """
        if not items:
            return

        self.console.print("📚 Enriching with background knowledge...")
        ai_client = create_ai_client(self.config.ai)
        enricher = ContentEnricher(ai_client, verbose_reporter=self.verbose_reporter)
        await enricher.enrich_batch(items)
        self.console.print(f"   Enriched {len(items)} items\n")

    async def _analyze_content(self, items: List[ContentItem]) -> List[ContentItem]:
        """Analyze content items with AI.

        Args:
            items: Items to analyze

        Returns:
            List[ContentItem]: Analyzed items
        """
        self.console.print("🤖 Analyzing content with AI...")

        ai_client = create_ai_client(self.config.ai)
        analyzer = ContentAnalyzer(
            ai_client,
            personal_briefing_mode=self.config.personal_briefing.enabled,
            verbose_reporter=self.verbose_reporter,
        )

        return await analyzer.analyze_batch(items)

    async def _generate_summary(
        self,
        items: List[ContentItem],
        date: str,
        total_fetched: int,
        language: str = "en",
    ) -> str:
        """Generate daily summary.

        Args:
            items: Important items to include (already enriched with background/related)
            date: Date string
            total_fetched: Total items fetched
            language: Output language ("en" or "zh")

        Returns:
            str: Markdown summary
        """
        self.console.print("📝 Generating daily summary...")

        summarizer = DailySummarizer()

        return await summarizer.generate_summary(items, date, total_fetched, language=language)
