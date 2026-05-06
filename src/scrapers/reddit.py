"""Reddit scraper implementation."""

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, List, Optional

import httpx

from .base import BaseScraper
from ..models import ContentItem, RedditConfig, RedditSubredditConfig, RedditUserConfig, SourceType

logger = logging.getLogger(__name__)

REDDIT_BASE = "https://www.reddit.com"
REDDIT_OAUTH_BASE = "https://oauth.reddit.com"
DEFAULT_USER_AGENT = "HorizonBrief/0.1 (+https://github.com/AntonMiklushov/Horizon; configure REDDIT_USER_AGENT)"
REDDIT_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{REDDIT_BASE}/",
}
MAX_COMMENT_CONCURRENCY = 2


class RedditScraper(BaseScraper):
    """Scraper for Reddit posts and comments."""

    def __init__(self, config: RedditConfig, http_client: httpx.AsyncClient):
        super().__init__(config.model_dump(), http_client)
        self.reddit_config = config
        self._comment_semaphore = asyncio.Semaphore(MAX_COMMENT_CONCURRENCY)
        self._access_token: str | None = None

    async def fetch(self, since: datetime) -> List[ContentItem]:
        if not self.config.get("enabled", True):
            return []

        tasks = []
        for sub_cfg in self.reddit_config.subreddits:
            if sub_cfg.enabled:
                tasks.append(self._fetch_subreddit(sub_cfg, since))
        for user_cfg in self.reddit_config.users:
            if user_cfg.enabled:
                tasks.append(self._fetch_user(user_cfg, since))

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        items = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Error fetching Reddit source: %s", result)
            elif isinstance(result, list):
                items.extend(result)
        return items

    async def _fetch_subreddit(self, cfg: RedditSubredditConfig, since: datetime) -> List[ContentItem]:
        params = {"limit": min(cfg.fetch_limit, 100), "raw_json": 1}
        if cfg.sort in ("top", "controversial"):
            params["t"] = cfg.time_filter

        url = await self._api_url(f"/r/{cfg.subreddit}/{cfg.sort}.json")
        data = await self._reddit_get(url, params)
        if not data:
            return []

        posts = [child["data"] for child in data.get("data", {}).get("children", [])
                 if child.get("kind") == "t3"]
        return await self._process_posts(
            posts, since, "subreddit", cfg.subreddit, cfg.min_score
        )

    async def _fetch_user(self, cfg: RedditUserConfig, since: datetime) -> List[ContentItem]:
        params = {"limit": min(cfg.fetch_limit, 100), "sort": cfg.sort, "raw_json": 1}
        url = await self._api_url(f"/user/{cfg.username}/submitted.json")
        data = await self._reddit_get(url, params)
        if not data:
            return []

        posts = [child["data"] for child in data.get("data", {}).get("children", [])
                 if child.get("kind") == "t3"]
        return await self._process_posts(
            posts, since, "user", cfg.username, min_score=0
        )

    async def _process_posts(
        self,
        posts: list,
        since: datetime,
        subtype: str,
        source_name: str,
        min_score: int,
    ) -> List[ContentItem]:
        valid_posts = []
        comment_tasks = []
        fetch_comments = self.reddit_config.fetch_comments

        for post in posts:
            created = datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc)
            if created < since:
                continue
            if post.get("score", 0) < min_score:
                continue
            valid_posts.append(post)
            if fetch_comments > 0:
                comment_tasks.append(
                    self._fetch_comments(post.get("subreddit", ""), post["id"])
                )
            else:
                comment_tasks.append(self._empty_comments())

        if not valid_posts:
            return []

        all_comments = await asyncio.gather(*comment_tasks, return_exceptions=True)

        items = []
        for post, comments in zip(valid_posts, all_comments):
            if isinstance(comments, Exception):
                comments = []
            item = self._parse_post(post, comments, subtype)
            if item:
                items.append(item)
        return items

    @staticmethod
    async def _empty_comments() -> List[dict]:
        return []

    async def _fetch_comments(self, subreddit: str, post_id: str) -> List[dict]:
        fetch_limit = self.reddit_config.fetch_comments
        url = await self._api_url(f"/r/{subreddit}/comments/{post_id}.json")
        params = {"limit": fetch_limit, "depth": 1, "sort": "top", "raw_json": 1}

        async with self._comment_semaphore:
            data = await self._reddit_get(url, params)
        if not data or not isinstance(data, list) or len(data) < 2:
            return []

        comments = []
        for child in data[1].get("data", {}).get("children", []):
            if child.get("kind") != "t1":
                continue
            c = child["data"]
            if c.get("body") and not c.get("distinguished") == "moderator":
                comments.append(c)

        comments.sort(key=lambda c: c.get("score", 0), reverse=True)
        return comments[:fetch_limit]

    def _parse_post(self, post: dict, comments: List[dict], subtype: str) -> Optional[ContentItem]:
        post_id = post["id"]
        title = post.get("title", "")
        is_self = post.get("is_self", False)
        subreddit = post.get("subreddit", "")
        discussion_url = f"https://www.reddit.com{post.get('permalink', '')}"

        # For link posts, use the external URL; for self posts, use the discussion URL
        url = discussion_url if is_self else post.get("url", discussion_url)

        author = post.get("author", "unknown")
        created = datetime.fromtimestamp(post.get("created_utc", 0), tz=timezone.utc)

        # Build content
        parts = []
        if post.get("selftext"):
            text = post["selftext"]
            if len(text) > 1500:
                text = text[:1497] + "..."
            parts.append(text)

        if comments:
            parts.append("\n--- Top Comments ---")
            for c in comments:
                commenter = c.get("author", "anon")
                body = c.get("body", "")
                body = body.strip()
                if len(body) > 500:
                    body = body[:497] + "..."
                score = c.get("score", 0)
                parts.append(f"[{commenter} ({score} pts)]: {body}")

        content = "\n\n".join(parts)

        return ContentItem(
            id=self._generate_id("reddit", subtype, post_id),
            source_type=SourceType.REDDIT,
            title=title,
            url=url,
            content=content,
            author=author,
            published_at=created,
            metadata={
                "score": post.get("score", 0),
                "upvote_ratio": post.get("upvote_ratio"),
                "num_comments": post.get("num_comments", 0),
                "subreddit": subreddit,
                "is_self": is_self,
                "flair": post.get("link_flair_text"),
                "discussion_url": discussion_url,
            },
        )

    async def _reddit_get(self, url: str, params: dict) -> Optional[Any]:
        try:
            response = await self.client.get(
                url,
                params=params,
                headers=await self._request_headers(),
                follow_redirects=True,
            )
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 5))
                logger.warning("Reddit rate limited, retrying after %ds", retry_after)
                await asyncio.sleep(retry_after)
                response = await self.client.get(
                    url,
                    params=params,
                    headers=await self._request_headers(),
                    follow_redirects=True,
                )
            if response.status_code == 403 and "/comments/" in url:
                logger.info("Reddit blocked comments request for %s; continuing without comments", url)
                return None
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.warning("Reddit request failed for %s: %s", url, e)
            return None

    async def _api_url(self, path: str) -> str:
        token = await self._get_access_token()
        base = REDDIT_OAUTH_BASE if token else REDDIT_BASE
        return f"{base}{path}"

    async def _request_headers(self) -> dict[str, str]:
        user_agent = os.getenv(self.reddit_config.user_agent_env) or self.reddit_config.user_agent
        headers = {**REDDIT_HEADERS, "User-Agent": user_agent}
        token = await self._get_access_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
            headers.pop("Referer", None)
        return headers

    async def _get_access_token(self) -> str | None:
        if self._access_token:
            return self._access_token
        client_id = os.getenv(self.reddit_config.client_id_env or "")
        client_secret = os.getenv(self.reddit_config.client_secret_env or "")
        if not client_id or not client_secret:
            return None
        try:
            response = await self.client.post(
                f"{REDDIT_BASE}/api/v1/access_token",
                data={"grant_type": "client_credentials"},
                auth=(client_id, client_secret),
                headers={"User-Agent": os.getenv(self.reddit_config.user_agent_env) or self.reddit_config.user_agent},
                timeout=30.0,
            )
            response.raise_for_status()
            token = response.json().get("access_token")
            if isinstance(token, str) and token:
                self._access_token = token
        except httpx.HTTPError as exc:
            logger.warning("Reddit OAuth unavailable, falling back to public JSON: %s", exc)
        return self._access_token
