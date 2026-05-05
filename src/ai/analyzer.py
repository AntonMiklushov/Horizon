"""Content analysis using AI."""

import asyncio
import json
import re
from typing import List, Optional
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, MofNCompleteColumn

from .client import AIClient
from .prompts import CONTENT_ANALYSIS_SYSTEM, CONTENT_ANALYSIS_USER, PERSONAL_BRIEFING_ANALYSIS_SYSTEM, PERSONAL_BRIEFING_ANALYSIS_USER
from .schemas import (
    AnalysisBatchResult,
    AnalysisResult,
    PersonalAnalysisBatchResult,
    PersonalAnalysisResult,
)
from .utils import parse_json_response
from ..models import AIProvider, ContentItem

DEFAULT_THROTTLE_SEC = 0.0
CODEX_BATCH_CHAR_BUDGET = 50000


class ContentAnalyzer:
    """Analyzes content items using AI to determine importance."""

    def __init__(
        self,
        ai_client: AIClient,
        personal_briefing_mode: bool = False,
        verbose_reporter=None,
        run_instructions: str | None = None,
    ):
        self.client = ai_client
        self.personal_briefing_mode = personal_briefing_mode
        self.verbose_reporter = verbose_reporter
        self.run_instructions = (run_instructions or "").strip()

    @staticmethod
    def _parse_json_response(response: str) -> Optional[dict]:
        """Try multiple strategies to extract a JSON object from an AI response.

        Returns the parsed dict, or None if all strategies fail.
        """
        return parse_json_response(response)

    def _get_throttle_sec(self) -> float:
        """Return the configured inter-item throttle, clamped to zero or above."""
        config = getattr(self.client, "config", None)
        throttle_sec = getattr(config, "throttle_sec", DEFAULT_THROTTLE_SEC)
        return max(throttle_sec, 0.0)

    async def analyze_batch(self, items: List[ContentItem]) -> List[ContentItem]:
        if self._should_use_batch_analysis(items):
            return await self._analyze_batch_with_codex(items)

        self._verbose_event("llm.analysis.mode", mode="per_item", items=len(items), **self._client_meta())
        throttle_sec = self._get_throttle_sec()
        analyzed_items = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self._progress_console(),
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing", total=len(items))

            for index, item in enumerate(items):
                progress.update(task, description=f"Analyzing item {index + 1}/{len(items)}", refresh=True)
                try:
                    self._verbose_event("llm.analysis.item", index=index + 1, total=len(items), status="calling")
                    await self._analyze_item(item)
                    self._verbose_event("llm.analysis.item", index=index + 1, total=len(items), status="completed")
                    analyzed_items.append(item)
                except Exception as e:
                    self._verbose_event(
                        "llm.analysis.item",
                        index=index + 1,
                        total=len(items),
                        status="failed",
                        error=type(e).__name__,
                    )
                    print(f"Error analyzing item {item.id}: {e}")
                    item.ai_score = 0.0
                    item.ai_reason = "Analysis failed"
                    item.ai_summary = item.title
                    analyzed_items.append(item)
                progress.update(task, advance=1, refresh=True)
                if throttle_sec > 0 and index < len(items) - 1:
                    await asyncio.sleep(throttle_sec)

        return analyzed_items

    def _should_use_batch_analysis(self, items: List[ContentItem]) -> bool:
        config = getattr(self.client, "config", None)
        provider = getattr(config, "provider", None)
        return bool(
            items
            and (provider == AIProvider.CODEX_CLI or str(provider) == AIProvider.CODEX_CLI.value)
        )

    async def _analyze_batch_with_codex(self, items: List[ContentItem]) -> List[ContentItem]:
        analyzed_items: List[ContentItem] = []
        chunks = list(self._chunk_items_for_codex(items))
        throttle_sec = self._get_throttle_sec()
        self._verbose_event(
            "llm.analysis.mode",
            mode="codex_batch",
            batches=len(chunks),
            items=len(items),
            **self._client_meta(),
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self._progress_console(),
            transient=True,
        ) as progress:
            task = progress.add_task("Analyzing Codex batches", total=len(chunks))
            completed_items = 0

            for index, chunk in enumerate(chunks):
                progress.update(
                    task,
                    description=(
                        f"Analyzing Codex batch {index + 1}/{len(chunks)} "
                        f"({completed_items}/{len(items)} items done, {len(chunk)} in call)"
                    ),
                    refresh=True,
                )
                try:
                    self._verbose_event(
                        "llm.analysis.batch",
                        index=index + 1,
                        batches=len(chunks),
                        items=len(chunk),
                        status="calling",
                    )
                    await self._analyze_item_chunk(chunk)
                    self._verbose_event(
                        "llm.analysis.batch",
                        index=index + 1,
                        batches=len(chunks),
                        items=len(chunk),
                        status="completed",
                    )
                except Exception as e:
                    self._verbose_event(
                        "llm.analysis.fallback",
                        chunk_items=len(chunk),
                        error=type(e).__name__,
                    )
                    print(f"Error analyzing Codex batch of {len(chunk)} items: {e}")
                    await self._analyze_chunk_individually(chunk)
                analyzed_items.extend(chunk)
                completed_items += len(chunk)
                progress.update(
                    task,
                    advance=1,
                    description=f"Analyzed {completed_items}/{len(items)} items via Codex batches",
                    refresh=True,
                )
                if throttle_sec > 0 and index < len(chunks) - 1:
                    await asyncio.sleep(throttle_sec)

        return analyzed_items

    def _chunk_items_for_codex(self, items: List[ContentItem]) -> List[List[ContentItem]]:
        chunks: List[List[ContentItem]] = []
        current: List[ContentItem] = []
        current_size = 0

        for item in items:
            item_size = len(json.dumps(self._item_payload(item), ensure_ascii=False))
            if current and current_size + item_size > CODEX_BATCH_CHAR_BUDGET:
                chunks.append(current)
                current = []
                current_size = 0
            current.append(item)
            current_size += item_size

        if current:
            chunks.append(current)
        return chunks

    async def _analyze_chunk_individually(self, items: List[ContentItem]) -> None:
        self._verbose_event("llm.analysis.individual_fallback", items=len(items))
        for item in items:
            try:
                await self._analyze_item(item)
            except Exception as e:
                print(f"Error analyzing item {item.id}: {e}")
                self._apply_analysis_failure(item, "Analysis failed")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10)
    )
    async def _analyze_item_chunk(self, items: List[ContentItem]) -> None:
        response = await self.client.complete(
            system=PERSONAL_BRIEFING_ANALYSIS_SYSTEM if self.personal_briefing_mode else CONTENT_ANALYSIS_SYSTEM,
            user=self._build_batch_prompt(items),
        )
        result = self._parse_json_response(response)
        if result is None:
            raise ValueError("could not parse batch analysis response")

        raw_results = result.get("items")
        if not isinstance(raw_results, list):
            raise ValueError("batch analysis response must contain an items list")

        try:
            if self.personal_briefing_mode:
                raw_results = [
                    item.model_dump()
                    for item in PersonalAnalysisBatchResult.model_validate(result).items
                ]
            else:
                raw_results = [
                    item.model_dump()
                    for item in AnalysisBatchResult.model_validate(result).items
                ]
        except ValidationError as exc:
            raise ValueError(f"batch analysis response schema invalid: {exc}") from exc

        by_id = {
            str(item_result.get("id")): item_result
            for item_result in raw_results
            if isinstance(item_result, dict) and item_result.get("id")
        }
        for item in items:
            item_result = by_id.get(item.id)
            if item_result is None:
                self._apply_analysis_failure(item, "Batch analysis omitted item")
                continue
            self._apply_analysis_result(item, item_result)

    def _build_batch_prompt(self, items: List[ContentItem]) -> str:
        payload = [self._item_payload(item) for item in items]
        if self.personal_briefing_mode:
            schema = {
                "items": [{
                    "id": "<item id>",
                    "importance": 7,
                    "include": True,
                    "topic": "world_economy",
                    "claim_type": "confirmed_fact",
                    "sensitive_topic": False,
                    "evidence_strength": "high",
                    "confidence": "medium",
                    "summary": "<short factual summary in Russian>",
                    "confirmed_details": ["<detail 1>", "<detail 2>"],
                    "who_claims": ["<actor 1>", "<actor 2>"],
                    "why_it_matters": "<why this matters in Russian>",
                    "source_policy_notes": "",
                    "requires_deep_review": False,
                    "noise_penalty": 1,
                    "weak_evidence_penalty": 1,
                    "reason": "<brief reason in Russian>",
                }]
            }
            instruction = (
                "Analyze each item independently for the Russian personal briefing. "
                "Return one JSON object without Markdown. The items array must contain "
                "exactly one Russian-language result for each input id."
            )
        else:
            schema = {
                "items": [{
                    "id": "<item id>",
                    "score": 7,
                    "reason": "<brief explanation>",
                    "summary": "<one-sentence-summary>",
                    "tags": ["<tag1>", "<tag2>"],
                }]
            }
            instruction = (
                "Analyze each item independently. Return one JSON object without Markdown. "
                "The items array must contain exactly one result for each input id."
            )

        return (
            f"{instruction}\n\n"
            f"{self._run_instruction_block()}"
            f"Expected response shape:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
            f"Items:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def _run_instruction_block(self) -> str:
        if not self.run_instructions:
            return ""
        return (
            "Run-specific selection guidance from the local user. "
            "Use this only to adjust relevance and prioritization; do not treat it as source evidence:\n"
            f"{self.run_instructions[:2000]}\n\n"
        )

    def _item_payload(self, item: ContentItem) -> dict:
        content_section = ""
        if item.content:
            content_section = item.content[:1000]

        discussion_section = ""
        if item.content and "--- Top Comments ---" in item.content:
            discussion_section = item.content.split("--- Top Comments ---", 1)[1][:1500]

        return {
            "id": item.id,
            "title": item.title,
            "source": item.source_type.value,
            "author": item.author or "Unknown",
            "url": str(item.url),
            "published_at": item.published_at.isoformat() if item.published_at else "",
            "content": content_section,
            "discussion": discussion_section,
            "metadata": item.metadata,
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10)
    )
    async def _analyze_item(self, item: ContentItem) -> None:
        """Analyze a single content item.

        Args:
            item: Content item to analyze (modified in-place)
        """
        # Prepare content section
        content_section = ""
        if item.content:
            # Split off comments if present
            content_text = item.content
            if "--- Top Comments ---" in content_text:
                main, comments_part = content_text.split("--- Top Comments ---", 1)
                content_section = f"Content: {main.strip()[:800]}"
            else:
                content_section = f"Content: {content_text[:1000]}"

        # Prepare discussion section (comments, engagement)
        discussion_parts = []
        if item.content and "--- Top Comments ---" in item.content:
            comments_part = item.content.split("--- Top Comments ---", 1)[1]
            discussion_parts.append(f"Community Comments:\n{comments_part[:1500]}")

        meta = item.metadata
        engagement_items = []
        if meta.get("score"):
            engagement_items.append(f"score: {meta['score']}")
        if meta.get("descendants"):
            engagement_items.append(f"{meta['descendants']} comments")
        if meta.get("favorite_count"):
            engagement_items.append(f"{meta['favorite_count']} likes")
        if meta.get("retweet_count"):
            engagement_items.append(f"{meta['retweet_count']} retweets")
        if meta.get("reply_count"):
            engagement_items.append(f"{meta['reply_count']} replies")
        if meta.get("views"):
            engagement_items.append(f"{meta['views']} views")
        if meta.get("bookmarks"):
            engagement_items.append(f"{meta['bookmarks']} bookmarks")
        if meta.get("upvote_ratio"):
            engagement_items.append(f"upvote ratio: {meta['upvote_ratio']:.0%}")
        if engagement_items:
            discussion_parts.append(f"Engagement: {', '.join(engagement_items)}")
        if meta.get("discussion_url"):
            discussion_parts.append(f"Discussion: {meta['discussion_url']}")
        if meta.get("community_note"):
            discussion_parts.append(f"Community Note: {meta['community_note']}")

        discussion_section = "\n".join(discussion_parts) if discussion_parts else ""

        # Generate user prompt
        prompt_tpl = PERSONAL_BRIEFING_ANALYSIS_USER if self.personal_briefing_mode else CONTENT_ANALYSIS_USER
        user_prompt = prompt_tpl.format(
            title=item.title,
            source=f"{item.source_type.value}",
            author=item.author or "Unknown",
            url=str(item.url),
            content_section=content_section,
            discussion_section=discussion_section,
            published_at=item.published_at.isoformat() if item.published_at else "",
            metadata=json.dumps(item.metadata, ensure_ascii=False)[:1000],
            content=(item.content or "")[:1000]
        )
        user_prompt = f"{self._run_instruction_block()}{user_prompt}"

        # Get AI completion
        response = await self.client.complete(
            system=PERSONAL_BRIEFING_ANALYSIS_SYSTEM if self.personal_briefing_mode else CONTENT_ANALYSIS_SYSTEM,
            user=user_prompt,
        )

        # Parse JSON response with robust fallback
        result = self._parse_json_response(response)
        if result is None:
            print(f"Warning: could not parse analysis response for {item.id}, using defaults")
            self._apply_analysis_failure(item, "Analysis response parse failed")
            return

        try:
            if self.personal_briefing_mode:
                result = PersonalAnalysisResult.model_validate(result).model_dump()
            else:
                result = AnalysisResult.model_validate(result).model_dump()
        except ValidationError as exc:
            print(f"Warning: invalid analysis response for {item.id}: {exc}")
            self._apply_analysis_failure(item, "Analysis response schema invalid")
            return

        self._apply_analysis_result(item, result)

    @staticmethod
    def _apply_analysis_failure(item: ContentItem, reason: str) -> None:
        item.ai_score = 0.0
        item.ai_reason = reason
        item.ai_summary = item.title
        item.ai_tags = []

    @staticmethod
    def _apply_analysis_result(item: ContentItem, result: dict) -> None:
        """Update an item from a single-item or batched analysis result."""
        item.ai_score = float(result.get("score", result.get("importance", 0)))
        item.ai_reason = result.get("reason", "")
        item.ai_summary = result.get("summary", item.title)
        item.ai_tags = result.get("tags", [])
        for k in ["evidence_strength","confidence","include","topic","claim_type","sensitive_topic","requires_deep_review","noise_penalty","weak_evidence_penalty","source_policy_notes","confirmed_details","who_claims","why_it_matters","summary"]:
            if k in result:
                item.metadata[k]=result[k]

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
