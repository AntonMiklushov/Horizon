from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import httpx

from src.models import RSSSourceConfig
from src.scrapers.rss import RSSScraper


FEED = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>T</title>
<item><guid>stable-guid</guid><title>Item</title><link>https://example.com/a</link><pubDate>Sun, 03 May 2026 10:00:00 GMT</pubDate><description>Body</description></item>
</channel></rss>"""

UNDATED_FEED = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>T</title>
<item><guid>undated</guid><title>Undated</title><link>https://example.com/u</link><description>Body</description></item>
</channel></rss>"""


def _client(feed: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text=feed)))


def test_rss_ids_are_stable_across_scraper_instances():
    source = RSSSourceConfig(name="T", url="https://example.com/feed.xml")
    since = datetime(2026, 5, 2, tzinfo=timezone.utc)

    first_client = _client(FEED)
    second_client = _client(FEED)
    first = asyncio.run(RSSScraper([source], first_client).fetch(since))
    second = asyncio.run(RSSScraper([source], second_client).fetch(since))
    asyncio.run(first_client.aclose())
    asyncio.run(second_client.aclose())

    assert first[0].id == second[0].id
    assert "stable-guid" not in first[0].id


def test_rss_undated_policy_can_include_with_low_freshness():
    source = RSSSourceConfig(
        name="T",
        url="https://example.com/feed.xml",
        undated_policy="include_with_low_freshness",
    )
    client = _client(UNDATED_FEED)
    items = asyncio.run(RSSScraper([source], client).fetch(datetime.now(timezone.utc) - timedelta(hours=1)))
    asyncio.run(client.aclose())

    assert len(items) == 1
    assert items[0].published_at is None
    assert items[0].metadata["freshness"] == "undated"


def test_rss_fetch_error_redacts_expanded_url_secret(caplog, monkeypatch):
    monkeypatch.setenv("LWN_KEY", "super-secret-token")
    source = RSSSourceConfig(
        name="LWN",
        url="https://lwn.net/headlines/full_text?key=${LWN_KEY}",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with caplog.at_level(logging.WARNING):
        items = asyncio.run(RSSScraper([source], client).fetch(datetime.now(timezone.utc) - timedelta(hours=1)))
    asyncio.run(client.aclose())

    assert items == []
    assert "super-secret-token" not in caplog.text
    assert "key=%3Credacted%3E" in caplog.text
