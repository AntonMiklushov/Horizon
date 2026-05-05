"""RSS feed scraper implementation."""

import calendar
import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from typing import List
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import httpx
import feedparser

from .base import BaseScraper
from ..models import ContentItem, SourceType, RSSSourceConfig

logger = logging.getLogger(__name__)


_SECRET_QUERY_TOKENS = ("key", "token", "secret", "password")


def _redact_feed_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<redacted-url>"

    query = urlencode(
        [
            (
                key,
                "<redacted>" if any(token in key.lower() for token in _SECRET_QUERY_TOKENS) else value,
            )
            for key, value in parse_qsl(parts.query, keep_blank_values=True)
        ],
        doseq=True,
    )
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _redact_error_text(error: Exception, expanded_url: str) -> str:
    safe_url = _redact_feed_url(expanded_url)
    return str(error).replace(expanded_url, safe_url)


class RSSScraper(BaseScraper):
    """Scraper for RSS/Atom feeds."""

    def __init__(self, sources: List[RSSSourceConfig], http_client: httpx.AsyncClient):
        """Initialize RSS scraper.

        Args:
            sources: List of RSS feed configurations
            http_client: Shared async HTTP client
        """
        super().__init__({"sources": sources}, http_client)

    async def fetch(self, since: datetime) -> List[ContentItem]:
        """Fetch RSS feed items.

        Args:
            since: Only fetch items published after this time

        Returns:
            List[ContentItem]: Fetched content items
        """
        items = []
        sources = self.config["sources"]

        for source in sources:
            if not source.enabled:
                continue

            feed_items = await self._fetch_feed(source, since)
            items.extend(feed_items)

        return items

    async def _fetch_feed(
        self,
        source: RSSSourceConfig,
        since: datetime
    ) -> List[ContentItem]:
        """Fetch items from a single RSS feed.

        Args:
            source: RSS feed configuration
            since: Only fetch items after this time

        Returns:
            List[ContentItem]: Feed content items
        """
        items = []
        feed_url = str(source.url)

        try:
            # Expand environment variables in URL (e.g. ${LWN_TOKEN})
            feed_url = re.sub(
                r'\$\{(\w+)\}',
                lambda m: os.environ.get(m.group(1), m.group(0)).strip(),
                str(source.url),
            )

            # Fetch feed content
            response = await self.client.get(feed_url, follow_redirects=True)
            response.raise_for_status()

            # Parse feed
            feed = feedparser.parse(response.text)

            for entry in feed.entries:
                # Parse published date
                published_at = self._parse_date(entry)
                freshness = "published"
                if not published_at:
                    if source.undated_policy == "drop":
                        continue
                    if source.undated_policy == "fetched_at":
                        published_at = datetime.now(timezone.utc)
                        freshness = "fetched_at"
                    else:
                        freshness = "undated"
                if published_at and published_at < since:
                    continue

                # Generate unique ID from feed URL and entry ID
                feed_id = str(source.url).split("//")[1].replace("/", "_")
                entry_id = entry.get("id", entry.get("link", ""))
                stable_entry_id = hashlib.sha256(f"{source.url}|{entry_id}".encode("utf-8")).hexdigest()[:16]

                # Extract content
                content = self._extract_content(entry)

                item = ContentItem(
                    id=self._generate_id("rss", feed_id, stable_entry_id),
                    source_type=SourceType.RSS,
                    title=entry.get("title", "Untitled"),
                    url=entry.get("link", str(source.url)),
                    content=content,
                    author=entry.get("author", source.name),
                    published_at=published_at,
                    metadata={
                        "feed_name": source.name,
                        "category": source.category,
                        "tags": [tag.term for tag in entry.get("tags", [])],
                        "freshness": freshness,
                    }
                )
                items.append(item)

        except httpx.HTTPError as e:
            logger.warning(
                "Error fetching RSS feed %s (%s): %s",
                source.name,
                _redact_feed_url(feed_url),
                _redact_error_text(e, feed_url),
            )
        except Exception as e:
            logger.warning(
                "Error parsing RSS feed %s (%s): %s",
                source.name,
                _redact_feed_url(feed_url),
                _redact_error_text(e, feed_url),
            )

        return items

    def _parse_date(self, entry: dict) -> datetime:
        """Parse publication date from feed entry.

        Args:
            entry: Feed entry data

        Returns:
            datetime: Parsed publication date or None
        """
        # Try different date fields
        for field in ["published", "updated", "created"]:
            if field in entry:
                try:
                    # Try parsing structured time first
                    if f"{field}_parsed" in entry and entry[f"{field}_parsed"]:
                        return datetime.fromtimestamp(
                            calendar.timegm(entry[f"{field}_parsed"]),
                            tz=timezone.utc
                        )
                    # Fallback to string parsing
                    date_str = entry[field]
                    return parsedate_to_datetime(date_str)
                except Exception:
                    continue

        return None

    def _extract_content(self, entry: dict) -> str:
        """Extract text content from feed entry.

        Args:
            entry: Feed entry data

        Returns:
            str: Extracted text content
        """
        # Try different content fields
        if "summary" in entry:
            return entry.summary
        elif "description" in entry:
            return entry.description
        elif "content" in entry and entry.content:
            # content is usually a list
            return entry.content[0].get("value", "")

        return ""
