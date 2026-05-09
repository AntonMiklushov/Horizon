"""Content enrichment using AI (second-pass analysis).

For items that pass the score threshold, this module:
1. Searches the web for relevant context (via DuckDuckGo)
2. Feeds search results + item content to AI to generate grounded background knowledge
"""

import json
import re
import sys
import os
from typing import Callable, List, Optional
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn
from ddgs import DDGS

from .client import AIClient
from .prompts import (
    CONCEPT_EXTRACTION_SYSTEM, CONCEPT_EXTRACTION_USER,
    CONTENT_ENRICHMENT_SYSTEM, CONTENT_ENRICHMENT_USER,
)
from .schemas import EnrichmentResult
from .utils import parse_json_response
from ..models import ContentItem


class ContentEnricher:
    """Enriches high-scoring content items with background knowledge."""

    def __init__(
        self,
        ai_client: AIClient,
        verbose_reporter=None,
        search_result_filter: Callable[[dict], bool] | None = None,
    ):
        self.client = ai_client
        self.verbose_reporter = verbose_reporter
        self.search_result_filter = search_result_filter

    async def enrich_batch(self, items: List[ContentItem]) -> None:
        """Enrich items in-place with background knowledge.

        Args:
            items: Content items to enrich (modified in-place)
        """
        self._verbose_event(
            "llm.enrichment.mode",
            items=len(items),
            expected_llm_calls=len(items) * 2,
            **self._client_meta(),
        )
        totals = {"queries": 0, "search_results": 0, "enriched": 0, "failed": 0}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self._progress_console(),
            transient=True,
        ) as progress:
            task = progress.add_task("Enriching", total=len(items))

            for index, item in enumerate(items):
                progress.update(task, description=f"Enriching item {index + 1}/{len(items)}", refresh=True)
                try:
                    self._verbose_event("llm.enrichment.item", index=index + 1, total=len(items), status="calling")
                    stats = await self._enrich_item(item)
                    self._verbose_event(
                        "llm.enrichment.item",
                        index=index + 1,
                        total=len(items),
                        status="completed",
                        queries=stats.get("queries", 0),
                        search_results=stats.get("search_results", 0),
                        enriched=bool(stats.get("enriched")),
                    )
                    totals["queries"] += int(stats.get("queries", 0))
                    totals["search_results"] += int(stats.get("search_results", 0))
                    totals["enriched"] += 1 if stats.get("enriched") else 0
                except Exception as e:
                    totals["failed"] += 1
                    self._verbose_event(
                        "llm.enrichment.item",
                        index=index + 1,
                        total=len(items),
                        status="failed",
                        error=type(e).__name__,
                    )
                    print(f"Error enriching item {item.id}: {e}")
                progress.update(task, advance=1, refresh=True)
        self._verbose_event("llm.enrichment.result", **totals)

    async def _web_search(self, query: str, max_results: int = 3) -> list:
        """Search the web for context via DuckDuckGo.

        Returns:
            List of dicts with keys: title, url, body
        """
        try:
            # Suppress primp "Impersonate ... does not exist" stderr warning
            stderr = sys.stderr
            sys.stderr = open(os.devnull, "w")
            try:
                ddgs = DDGS()
                results = ddgs.text(query, max_results=max_results)
            finally:
                sys.stderr.close()
                sys.stderr = stderr
        except Exception:
            return []

        normalized = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "body": r.get("body", "")}
            for r in (results or [])
        ]
        if self.search_result_filter is not None:
            normalized = [result for result in normalized if self.search_result_filter(result)]
        return normalized

    @staticmethod
    def _parse_json_response(response: str) -> Optional[dict]:
        """Try multiple strategies to extract a JSON object from an AI response.

        Returns the parsed dict, or None if all strategies fail.
        """
        return parse_json_response(response)

    async def _extract_concepts(self, item: ContentItem, content_text: str) -> List[str]:
        """Ask AI to identify concepts that need explanation.

        Args:
            item: Content item
            content_text: Extracted content text

        Returns:
            List of search queries for concepts that need explanation
        """
        user_prompt = CONCEPT_EXTRACTION_USER.format(
            title=item.title,
            summary=item.ai_summary or item.title,
            tags=", ".join(item.ai_tags) if item.ai_tags else "",
            content=content_text[:1000],
        )

        try:
            response = await self.client.complete(
                system=CONCEPT_EXTRACTION_SYSTEM,
                user=user_prompt,
            )
            result = self._parse_json_response(response)
            if result is None:
                return []
            queries = result.get("queries", [])
            return queries[:3]
        except Exception:
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10)
    )
    async def _enrich_item(self, item: ContentItem) -> dict:
        """Enrich a single item with background knowledge.

        Steps:
        1. Ask AI which concepts in the news need explanation
        2. Search the web for those concepts
        3. Ask AI to generate background based on search results

        Args:
            item: Content item to enrich (modified in-place via metadata)
        """
        # Extract content text and comments separately
        content_text = ""
        comments_text = ""
        if item.content:
            if "--- Top Comments ---" in item.content:
                main, comments_part = item.content.split("--- Top Comments ---", 1)
                content_text = main.strip()[:4000]
                comments_text = comments_part.strip()[:2000]
            else:
                content_text = item.content[:4000]

        # Step 1: AI identifies concepts to explain
        queries = await self._extract_concepts(item, content_text)

        # Step 2: Search web for each concept
        all_results = []
        web_sections = []
        for query in queries:
            results = await self._web_search(query)
            all_results.extend(results)
            if results:
                lines = [f"- [{r['title']}]({r['url']}): {r['body']}" for r in results]
                web_sections.append(f"**{query}:**\n" + "\n".join(lines))
        web_context = "\n\n".join(web_sections) if web_sections else ""

        # Index of available URLs for citation validation
        available_urls = {r["url"]: r["title"] for r in all_results if r.get("url")}

        # Step 3: AI generates background grounded in search results
        user_prompt = CONTENT_ENRICHMENT_USER.format(
            title=item.title,
            url=str(item.url),
            summary=item.ai_summary or item.title,
            score=item.ai_score or 0,
            reason=item.ai_reason or "",
            tags=", ".join(item.ai_tags) if item.ai_tags else "",
            content=content_text,
            comments_section=f"\n**Community Comments:**\n{comments_text}" if comments_text else "",
            web_context=web_context or "No web search results available.",
        )

        response = await self.client.complete(
            system=CONTENT_ENRICHMENT_SYSTEM,
            user=user_prompt,
        )

        # Parse JSON response with robust fallback
        result = self._parse_json_response(response)
        if result is None:
            # Gracefully degrade: skip enrichment instead of raising
            # (raising would trigger retries that won't help with a parse error)
            print(f"Warning: could not parse enrichment response for {item.id}, skipping enrichment")
            return {"queries": len(queries), "search_results": len(all_results), "enriched": False}

        try:
            result = EnrichmentResult.model_validate(result).model_dump()
        except ValidationError as exc:
            print(f"Warning: invalid enrichment response for {item.id}: {exc}")
            return {"queries": len(queries), "search_results": len(all_results), "enriched": False}

        # Combine structured sub-fields into per-language detailed_summary
        for lang in ("en", "zh"):
            if result.get(f"title_{lang}"):
                item.metadata[f"title_{lang}"] = str(result[f"title_{lang}"])

            parts = []
            for field in ("whats_new", "why_it_matters", "key_details"):
                text = result.get(f"{field}_{lang}", "").strip()
                if text:
                    parts.append(text)
            if parts:
                item.metadata[f"detailed_summary_{lang}"] = " ".join(parts)

            if result.get(f"background_{lang}"):
                item.metadata[f"background_{lang}"] = str(result[f"background_{lang}"])

            if result.get(f"community_discussion_{lang}"):
                item.metadata[f"community_discussion_{lang}"] = str(result[f"community_discussion_{lang}"])

        # Store citation sources — only URLs that actually came from our search results
        if result.get("sources") and available_urls:
            valid = [
                {"url": u, "title": available_urls[u]}
                for u in result["sources"]
                if u in available_urls
            ]
            if valid:
                item.metadata["sources"] = valid

        # Backward-compatible fallback fields (English as default)
        item.metadata["detailed_summary"] = item.metadata.get("detailed_summary_en", "")
        item.metadata["background"] = item.metadata.get("background_en", "")
        item.metadata["community_discussion"] = item.metadata.get("community_discussion_en", "")
        return {"queries": len(queries), "search_results": len(all_results), "enriched": True}

    def _verbose_event(self, name: str, **fields) -> None:
        if self.verbose_reporter is not None:
            self.verbose_reporter.event(name, **fields)

    def _progress_console(self):
        return getattr(self.verbose_reporter, "console", None)

    def _client_meta(self) -> dict:
        config = getattr(self.client, "config", None)
        provider = getattr(config, "provider", "unknown")
        provider_value = getattr(provider, "value", str(provider))
        return {
            "provider": provider_value,
            "model": getattr(config, "model", "unknown"),
        }
