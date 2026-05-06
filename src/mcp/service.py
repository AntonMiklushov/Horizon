"""Application service for staged Horizon Brief pipeline execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from inspect import signature
from pathlib import Path
from typing import Any

from .errors import HorizonMcpError
from .horizon_adapter import (
    apply_source_filter,
    dicts_to_items,
    get_enabled_sources,
    get_source_counts,
    items_to_dicts,
    load_config,
    load_runtime,
    make_orchestrator,
    make_storage,
    resolve_config_path,
    resolve_horizon_path,
)
from .run_store import RunStore
from ..services.webhook import WebhookNotifier
from ..models import AIProvider
from ..horizon_ext.personal import (
    drop_items_flagged_by_critic,
    EvidenceChecker,
    select_personal_important_items,
)
from ..horizon_ext.mcp import local_only_config, normalize_run_instructions, redact_runtime_payload
from ..horizon_ext.pipeline import apply_source_diversity


def _default_runs_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "mcp-runs"


@dataclass
class PipelineContext:
    """Resolved execution context per call."""

    horizon_path: Path
    config_path: Path
    runtime: Any
    config: Any


class HorizonPipelineService:
    """High-level staged pipeline service."""

    def __init__(self, runs_root: Path | None = None):
        self.runs_root = Path(runs_root).resolve() if runs_root else _default_runs_root().resolve()
        self._run_store: RunStore | None = None

    @property
    def run_store(self) -> RunStore:
        if self._run_store is None:
            self._run_store = RunStore(self.runs_root)
        return self._run_store

    def list_runs(self, limit: int = 20) -> dict[str, Any]:
        """List recent runs and stage availability."""

        runs = self.run_store.list_runs(limit=limit)
        items = []
        for run in runs:
            run_id = run["run_id"]
            stages = {}
            for stage in ("raw", "scored", "filtered", "enriched"):
                stages[stage] = self.run_store.has_stage(run_id, stage)
            items.append(
                {
                    "run_id": run_id,
                    "created_at": run.get("created_at"),
                    "updated_at": run.get("updated_at"),
                    "stages": stages,
                    "meta": run.get("meta", {}),
                }
            )
        return {"count": len(items), "items": items}

    def get_run_meta(self, run_id: str) -> dict[str, Any]:
        """Read run metadata."""

        try:
            meta = self.run_store.load_meta(run_id)
        except FileNotFoundError as exc:
            raise HorizonMcpError(
                code="HZ_RUN_NOT_FOUND",
                message=f"run_id={run_id} does not exist.",
                details={"run_id": run_id},
            ) from exc
        return {"run_id": run_id, "meta": meta}

    def get_run_stage(
        self,
        run_id: str,
        stage: str,
        max_items: int = 200,
    ) -> dict[str, Any]:
        """Read staged item payload (JSON)."""

        if max_items <= 0:
            raise HorizonMcpError(code="HZ_INVALID_INPUT", message="max_items must be greater than 0.")
        try:
            items = self.run_store.load_items(run_id, stage)
        except ValueError as exc:
            raise HorizonMcpError(
                code="HZ_INVALID_STAGE",
                message=str(exc),
                details={"stage": stage},
            ) from exc
        except FileNotFoundError as exc:
            raise HorizonMcpError(
                code="HZ_STAGE_NOT_FOUND",
                message=f"run_id={run_id} is missing stage artifact: {stage}",
                details={"run_id": run_id, "stage": stage},
            ) from exc

        return {
            "run_id": run_id,
            "stage": stage,
            "count": len(items),
            "items": items[:max_items],
            "truncated": len(items) > max_items,
        }

    def get_run_summary(self, run_id: str, language: str = "zh") -> dict[str, Any]:
        """Read generated markdown summary for a run."""

        try:
            markdown = self.run_store.load_summary(run_id, language)
        except FileNotFoundError as exc:
            raise HorizonMcpError(
                code="HZ_SUMMARY_NOT_FOUND",
                message=f"run_id={run_id} is missing summary for language={language}.",
                details={"run_id": run_id, "language": language},
            ) from exc
        return {
            "run_id": run_id,
            "language": language,
            "summary": markdown,
        }

    def get_effective_config(
        self,
        horizon_path: str | None = None,
        config_path: str | None = None,
        sources: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return effective config after optional source filtering."""

        ctx, selected_sources, unknown_sources = self._build_context(
            horizon_path=horizon_path,
            config_path=config_path,
            sources=sources,
        )
        return {
            "horizon_path": str(ctx.horizon_path),
            "config_path": str(ctx.config_path),
            "selected_sources": selected_sources,
            "unknown_sources": unknown_sources,
            "config": self._redact_config(ctx.config.model_dump(mode="json")),
        }

    async def validate_config(
        self,
        horizon_path: str | None = None,
        config_path: str | None = None,
        sources: list[str] | None = None,
        check_env: bool = True,
    ) -> dict[str, Any]:
        ctx, selected_sources, unknown_sources = self._build_context(
            horizon_path=horizon_path,
            config_path=config_path,
            sources=sources,
        )

        warnings: list[str] = []
        missing_env: list[str] = []

        if check_env:
            required = []
            if ctx.config.ai.provider != AIProvider.CODEX_CLI and ctx.config.ai.api_key_env:
                required.append(ctx.config.ai.api_key_env)
            for key in required:
                if not os.getenv(key):
                    missing_env.append(key)

            if ctx.config.sources.github and not os.getenv("GITHUB_TOKEN"):
                warnings.append("GITHUB_TOKEN is not set; GitHub fetching may hit strict rate limits.")

            if getattr(ctx.config, "email", None) and ctx.config.email and ctx.config.email.enabled:
                pwd_key = ctx.config.email.password_env
                if not os.getenv(pwd_key):
                    missing_env.append(pwd_key)

            if getattr(ctx.config, "webhook", None) and ctx.config.webhook and ctx.config.webhook.enabled:
                if ctx.config.webhook.url_env and not os.getenv(ctx.config.webhook.url_env):
                    missing_env.append(ctx.config.webhook.url_env)

        return {
            "horizon_path": str(ctx.horizon_path),
            "config_path": str(ctx.config_path),
            "ai": {
                "provider": ctx.config.ai.provider.value,
                "model": ctx.config.ai.model,
                "languages": list(ctx.config.ai.languages),
                "api_key_env": ctx.config.ai.api_key_env,
            },
            "filtering": {
                "ai_score_threshold": ctx.config.filtering.ai_score_threshold,
                "time_window_hours": ctx.config.filtering.time_window_hours,
            },
            "enabled_sources": get_enabled_sources(ctx.config),
            "selected_sources": selected_sources,
            "unknown_sources": unknown_sources,
            "missing_env": missing_env,
            "warnings": warnings,
        }

    async def fetch_items(
        self,
        hours: int = 24,
        run_id: str | None = None,
        horizon_path: str | None = None,
        config_path: str | None = None,
        sources: list[str] | None = None,
        run_instructions: str | None = None,
        local_only: bool = False,
    ) -> dict[str, Any]:
        if hours <= 0:
            raise HorizonMcpError(code="HZ_INVALID_INPUT", message="hours must be greater than 0.")

        ctx, selected_sources, unknown_sources = self._build_context(
            horizon_path=horizon_path,
            config_path=config_path,
            sources=sources,
            local_only=local_only,
        )

        storage = make_storage(ctx.runtime, ctx.config_path)
        orchestrator = make_orchestrator(ctx.runtime, ctx.config, storage)

        run_id = self.run_store.create_run(run_id)
        since = datetime.now(timezone.utc) - timedelta(hours=hours)

        raw_items = await orchestrator.fetch_all_sources(since)
        personal_excluded: list[dict[str, str]] = []
        candidate_items = raw_items
        if self._personal_briefing_enabled(ctx.config):
            orchestrator._classify_personal_source_metadata(raw_items)
            candidate_items, personal_excluded = orchestrator._prefilter_personal_candidates(raw_items)
        merged_items = orchestrator.merge_cross_source_duplicates(candidate_items)

        self.run_store.save_items(run_id, "raw", items_to_dicts(merged_items))
        meta = self.run_store.update_meta(
            run_id,
            {
                "horizon_path": str(ctx.horizon_path),
                "config_path": str(ctx.config_path),
                "hours": hours,
                "since": since.isoformat(),
                "source_selection": selected_sources,
                "unknown_sources": unknown_sources,
                "raw_count_before_merge": len(raw_items),
                "raw_count_after_personal_prefilter": len(candidate_items),
                "raw_count": len(merged_items),
                "personal_prefilter_excluded": personal_excluded[:50],
                "run_instructions": self._normalize_instructions(run_instructions),
                "local_only": local_only,
            },
        )

        return {
            "run_id": run_id,
            "fetched": len(merged_items),
            "raw_before_merge": len(raw_items),
            "source_counts": get_source_counts(merged_items),
            "artifact": str((self.run_store.run_dir(run_id) / "raw_items.json").resolve()),
            "meta": meta,
        }

    async def score_items(
        self,
        run_id: str,
        source_stage: str = "raw",
        horizon_path: str | None = None,
        config_path: str | None = None,
        max_items: int | None = None,
        run_instructions: str | None = None,
        local_only: bool = False,
    ) -> dict[str, Any]:
        items, ctx = self._load_stage_items(
            run_id=run_id,
            stage=source_stage,
            horizon_path=horizon_path,
            config_path=config_path,
            local_only=local_only,
        )

        if not items:
            raise HorizonMcpError(code="HZ_EMPTY_INPUT", message="No items available for scoring.")
        scored_input, limit_meta = self._limit_items(items, max_items, "scoring")
        effective_instructions = self._run_instructions(run_id, run_instructions)

        ai_client = ctx.runtime.create_ai_client(ctx.config.ai)
        analyzer = self._make_content_analyzer(
            ctx.runtime.ContentAnalyzer,
            ai_client,
            personal_briefing_mode=self._personal_briefing_enabled(ctx.config),
            run_instructions=effective_instructions,
        )
        scored_items = await analyzer.analyze_batch(scored_input)

        self.run_store.save_items(run_id, "scored", items_to_dicts(scored_items))
        score_threshold = ctx.config.filtering.ai_score_threshold
        above_threshold = [x for x in scored_items if x.ai_score and x.ai_score >= score_threshold]

        meta = self.run_store.update_meta(
            run_id,
            {
                "scored_count": len(scored_items),
                "scored_input_count": len(scored_input),
                "scored_source_count": len(items),
                **limit_meta,
                "scored_threshold": score_threshold,
                "scored_above_threshold": len(above_threshold),
                "run_instructions": effective_instructions,
                "local_only": local_only,
            },
        )

        return {
            "run_id": run_id,
            "scored": len(scored_items),
            "source_items": len(items),
            "items_used": len(scored_input),
            "skipped_by_limit": max(0, len(items) - len(scored_input)),
            "above_threshold": len(above_threshold),
            "score_distribution": self._score_distribution(scored_items),
            "artifact": str((self.run_store.run_dir(run_id) / "scored_items.json").resolve()),
            "meta": meta,
        }

    async def filter_items(
        self,
        run_id: str,
        threshold: float | None = None,
        source_stage: str = "scored",
        topic_dedup: bool = True,
        horizon_path: str | None = None,
        config_path: str | None = None,
        run_instructions: str | None = None,
        local_only: bool = False,
    ) -> dict[str, Any]:
        items, ctx = self._load_stage_items(
            run_id=run_id,
            stage=source_stage,
            horizon_path=horizon_path,
            config_path=config_path,
            local_only=local_only,
        )

        effective_threshold = threshold if threshold is not None else ctx.config.filtering.ai_score_threshold
        effective_instructions = self._run_instructions(run_id, run_instructions)

        personal_excluded: list[dict[str, str]] = []
        if self._personal_briefing_enabled(ctx.config):
            storage = make_storage(ctx.runtime, ctx.config_path)
            orchestrator = make_orchestrator(ctx.runtime, ctx.config, storage)
            orchestrator._classify_personal_source_metadata(items)
            important_items, personal_excluded = select_personal_important_items(
                items,
                checker=EvidenceChecker(ctx.config.filtering.time_window_hours),
                min_importance=ctx.config.personal_briefing.min_importance,
                min_importance_priority_topics=ctx.config.personal_briefing.min_importance_priority_topics,
                require_dates=ctx.config.personal_briefing.require_dates,
                priority_topics=self._priority_topics(ctx.config),
            )
        else:
            important_items = [item for item in items if item.ai_score and item.ai_score >= effective_threshold]
        important_items.sort(key=lambda x: x.ai_score or 0, reverse=True)

        before_dedup = len(important_items)
        if topic_dedup and important_items:
            storage = make_storage(ctx.runtime, ctx.config_path)
            orchestrator = make_orchestrator(ctx.runtime, ctx.config, storage)
            important_items = await orchestrator.merge_topic_duplicates(important_items)
            if self._personal_briefing_enabled(ctx.config):
                important_items, retracked = select_personal_important_items(
                    important_items,
                    checker=EvidenceChecker(ctx.config.filtering.time_window_hours),
                    min_importance=ctx.config.personal_briefing.min_importance,
                    min_importance_priority_topics=ctx.config.personal_briefing.min_importance_priority_topics,
                    require_dates=ctx.config.personal_briefing.require_dates,
                    priority_topics=self._priority_topics(ctx.config),
                )
                personal_excluded.extend(retracked)
        after_topic_dedup = len(important_items)

        important_items, diversity_excluded = apply_source_diversity(
            important_items,
            max_items_per_source=ctx.config.filtering.max_items_per_source,
        )
        personal_excluded.extend(diversity_excluded)

        self.run_store.save_items(run_id, "filtered", items_to_dicts(important_items))
        meta = self.run_store.update_meta(
            run_id,
            {
                "filtered_count": len(important_items),
                "filter_threshold": effective_threshold,
                "topic_dedup_enabled": topic_dedup,
                "topic_dedup_removed": before_dedup - after_topic_dedup,
                "source_diversity_excluded": diversity_excluded[:50],
                "personal_filter_excluded": personal_excluded[:50],
                "run_instructions": effective_instructions,
                "local_only": local_only,
            },
        )

        return {
            "run_id": run_id,
            "kept": len(important_items),
            "threshold": effective_threshold,
            "removed_by_topic_dedup": before_dedup - after_topic_dedup,
            "source_counts": get_source_counts(important_items),
            "artifact": str((self.run_store.run_dir(run_id) / "filtered_items.json").resolve()),
            "meta": meta,
        }

    async def enrich_items(
        self,
        run_id: str,
        source_stage: str = "filtered",
        horizon_path: str | None = None,
        config_path: str | None = None,
        max_items: int | None = None,
        local_only: bool = False,
    ) -> dict[str, Any]:
        items, ctx = self._load_stage_items(
            run_id=run_id,
            stage=source_stage,
            horizon_path=horizon_path,
            config_path=config_path,
            local_only=local_only,
        )

        if not items:
            raise HorizonMcpError(code="HZ_EMPTY_INPUT", message="No items available for enrichment.")
        enrichment_input, limit_meta = self._limit_items(items, max_items, "enrichment")

        ai_client = ctx.runtime.create_ai_client(ctx.config.ai)
        enricher = ctx.runtime.ContentEnricher(ai_client)
        await enricher.enrich_batch(enrichment_input)

        self.run_store.save_items(run_id, "enriched", items_to_dicts(enrichment_input))

        citation_count = 0
        for item in enrichment_input:
            citation_count += len(item.metadata.get("sources", []))

        meta = self.run_store.update_meta(
            run_id,
            {
                "enriched_count": len(enrichment_input),
                "enrichment_input_count": len(enrichment_input),
                "enrichment_source_count": len(items),
                **limit_meta,
                "citation_count": citation_count,
                "local_only": local_only,
            },
        )

        return {
            "run_id": run_id,
            "enriched": len(enrichment_input),
            "source_items": len(items),
            "items_used": len(enrichment_input),
            "skipped_by_limit": max(0, len(items) - len(enrichment_input)),
            "citation_count": citation_count,
            "artifact": str((self.run_store.run_dir(run_id) / "enriched_items.json").resolve()),
            "meta": meta,
        }

    async def generate_summary(
        self,
        run_id: str,
        language: str = "zh",
        source_stage: str | None = None,
        horizon_path: str | None = None,
        config_path: str | None = None,
        save_to_horizon_data: bool = False,
        max_items: int | None = None,
        local_only: bool = False,
    ) -> dict[str, Any]:
        stage = source_stage or self._pick_summary_stage(run_id)
        items, ctx = self._load_stage_items(
            run_id=run_id,
            stage=stage,
            horizon_path=horizon_path,
            config_path=config_path,
            local_only=local_only,
        )
        summary_items, limit_meta = self._limit_items(items, max_items, "summary")

        total_fetched = self._total_fetched(run_id, fallback=len(items))
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if self._uses_personal_summary(ctx.config, language):
            renderer = ctx.runtime.PersonalBriefingRenderer()
            summary = renderer.render(date_str, summary_items, tracked=[])
            critic_config = ctx.config.personal_briefing.critic_pass
            if critic_config.enabled:
                critic = ctx.runtime.run_briefing_critic(summary, summary_items)
                if (not critic.passed) and critic_config.auto_revise_once:
                    revised_items = drop_items_flagged_by_critic(summary_items, critic)
                    if len(revised_items) < len(summary_items):
                        summary_items = revised_items
                        summary = renderer.render(date_str, summary_items, tracked=[])
                        critic = ctx.runtime.run_briefing_critic(summary, summary_items)
                if not critic.passed:
                    failed = summary + "\n\n## Предупреждения аудита\n" + "\n".join(
                        [f"- {issue}" for issue in critic.critical_issues]
                    )
                    failed_path = self.run_store.save_summary(run_id, f"{language}-audit-failed", failed)
                    self.run_store.update_meta(
                        run_id,
                        {
                            "summary_language": language,
                            "summary_audit_failed": True,
                            "summary_audit_artifact": str(failed_path.resolve()),
                            "summary_audit_issues": critic.critical_issues,
                        },
                    )
                    raise HorizonMcpError(
                        code="HZ_CRITIC_FAILED",
                        message="Personal briefing failed the deterministic critic.",
                        details={"run_id": run_id, "language": language, "artifact": str(failed_path.resolve())},
                    )
        else:
            summarizer = ctx.runtime.DailySummarizer()
            summary = await summarizer.generate_summary(
                summary_items,
                date_str,
                total_fetched,
                language=language,
            )

        run_summary_path = self.run_store.save_summary(run_id, language, summary)
        published_path = None
        if save_to_horizon_data:
            storage = make_storage(ctx.runtime, ctx.config_path)
            published_path = storage.save_daily_summary(date_str, summary, language=language)

        summary_meta = {
            "summary_stage": stage,
            "summary_language": language,
            "summary_generated_at": datetime.now(timezone.utc).isoformat(),
            "summary_artifact": str(run_summary_path.resolve()),
            "summary_source_count": len(items),
            "summary_items_used": len(summary_items),
            **limit_meta,
            "local_only": local_only,
        }
        if published_path:
            summary_meta["summary_published_path"] = str(Path(published_path).resolve())
        meta = self.run_store.update_meta(run_id, summary_meta)

        return {
            "run_id": run_id,
            "language": language,
            "source_stage": stage,
            "total_fetched": total_fetched,
            "source_items": len(items),
            "items_used": len(summary_items),
            "skipped_by_limit": max(0, len(items) - len(summary_items)),
            "summary_path": str(run_summary_path.resolve()),
            "published_path": str(Path(published_path).resolve()) if published_path else None,
            "preview": summary[:1200],
            "meta": meta,
        }

    async def run_pipeline(
        self,
        hours: int = 24,
        run_id: str | None = None,
        languages: list[str] | None = None,
        threshold: float | None = None,
        horizon_path: str | None = None,
        config_path: str | None = None,
        sources: list[str] | None = None,
        enrich: bool = True,
        topic_dedup: bool = True,
        save_to_horizon_data: bool = False,
        max_raw_items: int | None = None,
        max_filtered_items: int | None = None,
        run_instructions: str | None = None,
        local_only: bool = False,
    ) -> dict[str, Any]:
        fetch_result = await self.fetch_items(
            hours=hours,
            run_id=run_id,
            horizon_path=horizon_path,
            config_path=config_path,
            sources=sources,
            run_instructions=run_instructions,
            local_only=local_only,
        )
        run_id = fetch_result["run_id"]

        score_result = await self.score_items(
            run_id=run_id,
            horizon_path=horizon_path,
            config_path=config_path,
            max_items=max_raw_items,
            run_instructions=run_instructions,
            local_only=local_only,
        )

        filter_result = await self.filter_items(
            run_id=run_id,
            threshold=threshold,
            topic_dedup=topic_dedup,
            horizon_path=horizon_path,
            config_path=config_path,
            run_instructions=run_instructions,
            local_only=local_only,
        )

        enrich_result: dict[str, Any] | None = None
        stage_for_summary = "filtered"
        if enrich:
            enrich_result = await self.enrich_items(
                run_id=run_id,
                source_stage="filtered",
                horizon_path=horizon_path,
                config_path=config_path,
                max_items=max_filtered_items,
                local_only=local_only,
            )
            stage_for_summary = "enriched"

        ctx, _, _ = self._build_context(
            horizon_path=horizon_path,
            config_path=config_path,
            sources=sources,
            local_only=local_only,
        )
        final_languages = languages if languages else list(ctx.config.ai.languages)
        if not languages and self._personal_briefing_enabled(ctx.config):
            personal = ctx.config.personal_briefing
            if not personal.generate_standard_summaries:
                final_languages = [personal.language]
            elif personal.language not in final_languages:
                final_languages.append(personal.language)

        summaries = []
        for lang in final_languages:
            summary_result = await self.generate_summary(
                run_id=run_id,
                language=lang,
                source_stage=stage_for_summary,
                horizon_path=horizon_path,
                config_path=config_path,
                save_to_horizon_data=False if local_only else save_to_horizon_data,
                max_items=max_filtered_items if not enrich else None,
                local_only=local_only,
            )
            summaries.append(summary_result)

        return {
            "run_id": run_id,
            "fetch": fetch_result,
            "score": score_result,
            "filter": filter_result,
            "enrich": enrich_result,
            "summaries": summaries,
            "limits": {
                "max_raw_items": max_raw_items,
                "max_filtered_items": max_filtered_items,
            },
            "run_instructions": self._normalize_instructions(run_instructions),
            "local_only": local_only,
            "meta": self.run_store.load_meta(run_id),
        }

    def _build_context(
        self,
        horizon_path: str | None,
        config_path: str | None,
        sources: list[str] | None,
        local_only: bool = False,
    ) -> tuple[PipelineContext, list[str], list[str]]:
        resolved_horizon = resolve_horizon_path(horizon_path)
        runtime = load_runtime(resolved_horizon)
        resolved_config = resolve_config_path(resolved_horizon, config_path)
        config = load_config(runtime, resolved_config)
        effective_config, selected_sources, unknown_sources = apply_source_filter(config, sources)
        if local_only:
            effective_config = self._local_only_config(effective_config)

        return (
            PipelineContext(
                horizon_path=resolved_horizon,
                config_path=resolved_config,
                runtime=runtime,
                config=effective_config,
            ),
            selected_sources,
            unknown_sources,
        )

    def _load_stage_items(
        self,
        run_id: str,
        stage: str,
        horizon_path: str | None,
        config_path: str | None,
        local_only: bool = False,
    ) -> tuple[list[Any], PipelineContext]:
        ctx, _, _ = self._build_context(
            horizon_path=horizon_path,
            config_path=config_path,
            sources=None,
            local_only=local_only,
        )
        try:
            payload = self.run_store.load_items(run_id, stage)
        except FileNotFoundError as exc:
            raise HorizonMcpError(
                code="HZ_STAGE_NOT_FOUND",
                message=f"run_id={run_id} is missing stage artifact: {stage}",
                details={"run_id": run_id, "stage": stage},
            ) from exc
        items = dicts_to_items(ctx.runtime, payload)
        return items, ctx

    def _pick_summary_stage(self, run_id: str) -> str:
        for stage in ("enriched", "filtered", "scored", "raw"):
            if self.run_store.has_stage(run_id, stage):
                return stage
        raise HorizonMcpError(
            code="HZ_STAGE_NOT_FOUND",
            message=f"run_id={run_id} has no usable stage for summary generation.",
            details={"run_id": run_id},
        )

    def _total_fetched(self, run_id: str, fallback: int) -> int:
        try:
            raw = self.run_store.load_items(run_id, "raw")
            return len(raw)
        except Exception:
            return fallback

    @staticmethod
    def _score_distribution(items: list[Any]) -> dict[str, int]:
        buckets = {"0-2": 0, "3-4": 0, "5-6": 0, "7-8": 0, "9-10": 0}
        for item in items:
            score = float(item.ai_score or 0.0)
            if score < 3:
                buckets["0-2"] += 1
            elif score < 5:
                buckets["3-4"] += 1
            elif score < 7:
                buckets["5-6"] += 1
            elif score < 9:
                buckets["7-8"] += 1
            else:
                buckets["9-10"] += 1
        return buckets

    @staticmethod
    def _personal_briefing_enabled(config: Any) -> bool:
        personal = getattr(config, "personal_briefing", None)
        return bool(getattr(personal, "enabled", False))

    @staticmethod
    def _priority_topics(config: Any) -> set[str]:
        personal = getattr(config, "personal_briefing", None)
        return {str(topic) for topic in getattr(personal, "priority_topics", [])}

    @staticmethod
    def _uses_personal_summary(config: Any, language: str) -> bool:
        personal = getattr(config, "personal_briefing", None)
        return bool(
            getattr(personal, "enabled", False)
            and language == getattr(personal, "language", None)
        )

    @staticmethod
    def _normalize_instructions(run_instructions: str | None) -> str:
        return normalize_run_instructions(run_instructions)

    def _run_instructions(self, run_id: str, explicit: str | None) -> str:
        normalized = self._normalize_instructions(explicit)
        if normalized:
            return normalized
        try:
            return self._normalize_instructions(self.run_store.load_meta(run_id).get("run_instructions"))
        except Exception:
            return ""

    @staticmethod
    def _make_content_analyzer(
        analyzer_cls: Any,
        ai_client: Any,
        personal_briefing_mode: bool,
        run_instructions: str,
    ) -> Any:
        kwargs = {"personal_briefing_mode": personal_briefing_mode}
        try:
            params = signature(analyzer_cls).parameters
        except (TypeError, ValueError):
            params = {}
        if "run_instructions" in params:
            kwargs["run_instructions"] = run_instructions
        return analyzer_cls(ai_client, **kwargs)

    @staticmethod
    def _local_only_config(config: Any) -> Any:
        return local_only_config(config)

    @staticmethod
    def _limit_items(items: list[Any], max_items: int | None, stage_name: str) -> tuple[list[Any], dict[str, Any]]:
        if max_items is None:
            return items, {
                f"{stage_name}_limit": None,
                f"{stage_name}_skipped_by_limit": 0,
            }
        if max_items <= 0:
            raise HorizonMcpError(
                code="HZ_INVALID_INPUT",
                message=f"max_items for {stage_name} must be greater than 0.",
                details={"stage": stage_name, "max_items": max_items},
            )
        limited = items[:max_items]
        return limited, {
            f"{stage_name}_limit": max_items,
            f"{stage_name}_skipped_by_limit": max(0, len(items) - len(limited)),
        }

    @classmethod
    def _redact_config(cls, value: Any) -> Any:
        return redact_runtime_payload(value)

    async def send_webhook(
        self,
        date: str,
        language: str = "zh",
        important_items: int = 0,
        all_items: int = 0,
        result: str = "success",
        summary: str = "",
        horizon_path: str | None = None,
        config_path: str | None = None,
    ) -> dict[str, Any]:
        """Send a webhook notification using the configured webhook settings."""

        ctx, _, _ = self._build_context(
            horizon_path=horizon_path,
            config_path=config_path,
            sources=None,
        )

        webhook_config = ctx.config.webhook
        if not webhook_config or not webhook_config.enabled:
            return {
                "sent": False,
                "reason": "Webhook is not enabled in configuration.",
            }

        notifier = WebhookNotifier(webhook_config)
        variables = {
            "date": date,
            "language": language,
            "important_items": important_items,
            "all_items": all_items,
            "result": result,
            "timestamp": str(int(datetime.now(timezone.utc).timestamp())),
            "message_title": f"Horizon Brief {date} webhook",
            "message_kind": "manual",
            "summary": summary,
        }

        await notifier.notify(variables)

        return {
            "sent": True,
            "variables": {k: (v if k != "summary" else f"<{len(v)} chars>") for k, v in variables.items()},
        }
