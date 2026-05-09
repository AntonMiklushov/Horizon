"""Application service for staged Horizon Brief pipeline execution."""

from __future__ import annotations

import asyncio
import os
import time
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
    apply_personal_selection_caps,
    CorroborationGate,
    drop_items_flagged_by_critic,
    EvidenceChecker,
    select_personal_important_items,
    source_policy_decisions,
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


class RunTraceReporter:
    """Persist safe, observable run activity without raw prompts or model thoughts."""

    def __init__(self, run_store: RunStore, run_id: str):
        self.run_store = run_store
        self.run_id = run_id
        self.console = None
        self._disabled = False

    def start(self, stage: str, **fields: Any) -> float:
        started_at = time.monotonic()
        message = fields.pop("message", None)
        self._emit(stage, status="running", fields=fields, message=message)
        return started_at

    def end(self, stage: str, started_at: float | None = None, **fields: Any) -> None:
        payload = dict(fields)
        status = str(payload.pop("status", "completed"))
        message = payload.pop("message", None)
        if started_at is not None:
            payload["duration_sec"] = f"{time.monotonic() - started_at:.2f}"
        self._emit(stage, status=status, fields=payload, message=message)

    def event(self, name: str, **fields: Any) -> None:
        payload = dict(fields)
        status = str(payload.pop("status", "event"))
        message = payload.pop("message", None)
        self._emit(name, status=status, fields=payload, message=message)

    def _emit(self, name: str, status: str, fields: dict[str, Any], message: Any = None) -> None:
        if self._disabled:
            return

        safe_fields = self._json_safe(self._redact_trace_fields(redact_runtime_payload(fields)))
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "stage": self._stage_name(name),
            "status": status,
            "message": str(message) if message else self._message(name, status, safe_fields),
            "progress": self._progress(safe_fields),
            "fields": safe_fields,
        }
        try:
            self.run_store.append_trace_event(self.run_id, event)
        except (OSError, ValueError, TypeError):
            self._disabled = True

    @staticmethod
    def _stage_name(name: str) -> str:
        if name.startswith(("source.", "sources.")) or name == "fetch":
            return "fetch"
        if name.startswith("llm.analysis") or name == "score":
            return "analyze"
        if name.startswith(("filter", "dedup.", "selection.", "llm.topic_dedup")):
            return "filter"
        if name.startswith("llm.enrichment") or name == "enrich":
            return "enrich"
        if name.startswith(("summary", "summaries")):
            return "summary"
        if name.startswith("pipeline."):
            return "pipeline"
        return name.split(".", 1)[0]

    @classmethod
    def _message(cls, name: str, status: str, fields: dict[str, Any]) -> str:
        if name == "pipeline.start":
            return "Run started"
        if name == "pipeline.finish":
            return "Report ready"
        if name == "pipeline.error":
            return "Run failed"
        if name == "pipeline.cancelled":
            return "Run cancelled"
        if name == "fetch":
            if status == "cancelled":
                return "Fetch cancelled"
            if status == "failed":
                return "Fetch failed"
            if status == "completed":
                return f"Fetched {fields.get('fetched', fields.get('total_items', 0))} items"
            return "Fetching sources"
        if name.startswith("source."):
            source_name = name.replace("source.", "", 1)
            if status == "cancelled":
                return f"Cancelled while fetching {source_name}"
            if status == "failed":
                return f"Fetch failed for {source_name}"
            return f"Fetching from {source_name}" if status == "running" else f"Fetched {source_name}"
        if name == "score":
            if status == "completed":
                return f"Scored {fields.get('scored', 0)} items"
            return "Analyzing items with the model"
        if name == "filter":
            if status == "completed":
                return f"Kept {fields.get('kept', 0)} items after filtering"
            return "Filtering model-ranked items"
        if name == "enrich":
            if status == "skipped":
                return "Enrichment skipped"
            if status == "completed":
                return f"Enriched {fields.get('enriched', 0)} items"
            return "Enriching selected items"
        if name == "summary":
            if status == "completed":
                language = fields.get("language", "")
                return f"Generated {language} summary".strip()
            return "Generating summary"
        if name == "llm.analysis.mode":
            mode = fields.get("mode", "analysis")
            items = fields.get("items", 0)
            batches = fields.get("batches")
            if batches:
                return f"Preparing {batches} analysis batches for {items} items"
            return f"Preparing {mode} analysis for {items} items"
        if name == "llm.analysis.item":
            return cls._indexed_message("Analyzing item", fields, "total", status)
        if name == "llm.analysis.batch":
            return cls._indexed_message("Analyzing batch", fields, "batches", status)
        if name == "llm.analysis.fallback":
            return "Analysis batch failed; falling back to individual items"
        if name == "llm.analysis.individual_fallback":
            return "Analyzing fallback items individually"
        if name == "llm.topic_dedup":
            return f"Topic deduplication {status}"
        if name == "llm.enrichment.mode":
            return f"Preparing enrichment for {fields.get('items', 0)} items"
        if name == "llm.enrichment.item":
            return cls._indexed_message("Enriching item", fields, "total", status)
        if name == "llm.enrichment.result":
            return f"Enrichment completed: {fields.get('enriched', 0)} enriched"
        if name == "summary.render":
            return f"Rendering {fields.get('language', '')} summary".strip()
        if name == "summary.saved":
            return f"Saved {fields.get('language', '')} summary".strip()
        return name.replace(".", " ").capitalize()

    @staticmethod
    def _indexed_message(prefix: str, fields: dict[str, Any], total_key: str, status: str) -> str:
        index = fields.get("index")
        total = fields.get(total_key)
        suffix = f" ({status})" if status not in {"event", "calling"} else ""
        if index is not None and total is not None:
            return f"{prefix} {index}/{total}{suffix}"
        return f"{prefix}{suffix}"

    @staticmethod
    def _progress(fields: dict[str, Any]) -> dict[str, Any] | None:
        current = fields.get("index")
        total = fields.get("total", fields.get("batches"))
        if current is None or total is None:
            return None
        return {"current": current, "total": total}

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value if len(value) <= 500 else value[:497] + "..."
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(item) for item in value]
        return str(value)

    @classmethod
    def _redact_trace_fields(cls, value: Any) -> Any:
        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                key_l = str(key).lower()
                if any(token in key_l for token in ("prompt", "completion", "response", "system_message", "user_message")):
                    redacted[key] = "<redacted>"
                else:
                    redacted[key] = cls._redact_trace_fields(item)
            return redacted
        if isinstance(value, list):
            return [cls._redact_trace_fields(item) for item in value]
        return value


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

    def create_trace_reporter(self, run_id: str) -> RunTraceReporter:
        """Create a safe trace reporter for a known run id."""

        self.run_store.create_run(run_id)
        return RunTraceReporter(self.run_store, run_id)

    def get_run_trace(self, run_id: str, max_events: int = 200) -> dict[str, Any]:
        """Read safe observable activity events for a run."""

        try:
            events = self.run_store.load_trace_events(run_id, limit=max_events)
        except FileNotFoundError as exc:
            raise HorizonMcpError(
                code="HZ_RUN_NOT_FOUND",
                message=f"run_id={run_id} does not exist.",
                details={"run_id": run_id},
            ) from exc
        return {
            "run_id": run_id,
            "count": len(events),
            "events": self._redact_config(events),
        }

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
        trace_reporter: Any | None = None,
    ) -> dict[str, Any]:
        if hours <= 0:
            raise HorizonMcpError(code="HZ_INVALID_INPUT", message="hours must be greater than 0.")

        if trace_reporter is not None:
            trace_reporter.event("fetch", status="running", hours=hours, sources=sources or "enabled")

        ctx, selected_sources, unknown_sources = self._build_context(
            horizon_path=horizon_path,
            config_path=config_path,
            sources=sources,
            local_only=local_only,
        )

        storage = make_storage(ctx.runtime, ctx.config_path)
        orchestrator = make_orchestrator(ctx.runtime, ctx.config, storage)
        self._attach_trace_reporter(orchestrator, trace_reporter)

        run_id = self.run_store.create_run(run_id)
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        self.run_store.update_meta(run_id, {"status": "running", "current_stage": "fetch"})

        try:
            raw_items = await orchestrator.fetch_all_sources(since)
        except asyncio.CancelledError:
            self.run_store.update_meta(
                run_id,
                {
                    "status": "cancelled",
                    "current_stage": "fetch",
                    "cancelled_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            if trace_reporter is not None:
                trace_reporter.event("fetch", status="cancelled")
            raise
        except Exception:
            self.run_store.update_meta(
                run_id,
                {
                    "status": "failed",
                    "current_stage": "fetch",
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            if trace_reporter is not None:
                trace_reporter.event("fetch", status="failed")
            raise
        personal_excluded: list[dict[str, str]] = []
        candidate_items = raw_items
        if self._personal_briefing_enabled(ctx.config):
            orchestrator._classify_personal_source_metadata(raw_items)
            candidate_items, personal_excluded = orchestrator._prefilter_personal_candidates(raw_items)
        merged_items = orchestrator.merge_cross_source_duplicates(candidate_items)
        if self._personal_briefing_enabled(ctx.config):
            self.run_store.write_json(
                run_id,
                "source_policy_decisions.json",
                source_policy_decisions(merged_items, personal_excluded),
            )

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
                "status": "completed",
                "current_stage": "raw",
            },
        )

        if trace_reporter is not None:
            trace_reporter.event(
                "fetch",
                status="completed",
                fetched=len(merged_items),
                raw_before_merge=len(raw_items),
                source_counts=get_source_counts(merged_items),
                selected_sources=selected_sources,
                unknown_sources=unknown_sources,
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
        trace_reporter: Any | None = None,
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
        self.run_store.update_meta(run_id, {"status": "running", "current_stage": "analyze"})
        if trace_reporter is not None:
            trace_reporter.event(
                "score",
                status="running",
                source_stage=source_stage,
                source_items=len(items),
                items_used=len(scored_input),
            )

        ai_client = ctx.runtime.create_ai_client(ctx.config.ai)
        analyzer = self._make_content_analyzer(
            ctx.runtime.ContentAnalyzer,
            ai_client,
            personal_briefing_mode=self._personal_briefing_enabled(ctx.config),
            run_instructions=effective_instructions,
            trace_reporter=trace_reporter,
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
                "status": "completed",
                "current_stage": "scored",
            },
        )

        if trace_reporter is not None:
            trace_reporter.event(
                "score",
                status="completed",
                scored=len(scored_items),
                above_threshold=len(above_threshold),
                threshold=score_threshold,
                score_distribution=self._score_distribution(scored_items),
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
        trace_reporter: Any | None = None,
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
        self.run_store.update_meta(run_id, {"status": "running", "current_stage": "filter"})
        if trace_reporter is not None:
            trace_reporter.event(
                "filter",
                status="running",
                source_stage=source_stage,
                source_items=len(items),
                threshold=effective_threshold,
                topic_dedup=topic_dedup,
            )

        personal_excluded: list[dict[str, str]] = []
        if self._personal_briefing_enabled(ctx.config):
            storage = make_storage(ctx.runtime, ctx.config_path)
            orchestrator = make_orchestrator(ctx.runtime, ctx.config, storage)
            self._attach_trace_reporter(orchestrator, trace_reporter)
            orchestrator._classify_personal_source_metadata(items)
            important_items, personal_excluded = select_personal_important_items(
                items,
                checker=EvidenceChecker(
                    ctx.config.filtering.time_window_hours,
                    high_confidence_requires_supporting_source=self._high_confidence_requires_supporting_source(ctx.config),
                ),
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
            self._attach_trace_reporter(orchestrator, trace_reporter)
            important_items = await orchestrator.merge_topic_duplicates(important_items)
            if self._personal_briefing_enabled(ctx.config):
                important_items, retracked = select_personal_important_items(
                    important_items,
                    checker=EvidenceChecker(
                        ctx.config.filtering.time_window_hours,
                        high_confidence_requires_supporting_source=self._high_confidence_requires_supporting_source(ctx.config),
                    ),
                    min_importance=ctx.config.personal_briefing.min_importance,
                    min_importance_priority_topics=ctx.config.personal_briefing.min_importance_priority_topics,
                    require_dates=ctx.config.personal_briefing.require_dates,
                    priority_topics=self._priority_topics(ctx.config),
                )
                personal_excluded.extend(retracked)
        after_topic_dedup = len(important_items)

        if self._personal_briefing_enabled(ctx.config):
            CorroborationGate(getattr(ctx.config.personal_briefing, "corroboration", None)).apply(important_items)

        important_items, diversity_excluded = apply_source_diversity(
            important_items,
            max_items_per_source=ctx.config.filtering.max_items_per_source,
        )
        personal_excluded.extend(diversity_excluded)
        if self._personal_briefing_enabled(ctx.config):
            important_items, cap_excluded = apply_personal_selection_caps(
                important_items,
                getattr(ctx.config.personal_briefing, "selection_caps", None),
            )
            personal_excluded.extend(cap_excluded)

        if self._personal_briefing_enabled(ctx.config):
            self.run_store.write_json(
                run_id,
                "source_policy_decisions.json",
                source_policy_decisions(important_items, personal_excluded),
            )

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
                "status": "completed",
                "current_stage": "filtered",
            },
        )

        if trace_reporter is not None:
            trace_reporter.event(
                "filter",
                status="completed",
                kept=len(important_items),
                source_items=len(items),
                removed_by_topic_dedup=before_dedup - after_topic_dedup,
                source_counts=get_source_counts(important_items),
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
        trace_reporter: Any | None = None,
    ) -> dict[str, Any]:
        items, ctx = self._load_stage_items(
            run_id=run_id,
            stage=source_stage,
            horizon_path=horizon_path,
            config_path=config_path,
            local_only=local_only,
        )

        enrichment_input, limit_meta = self._limit_items(items, max_items, "enrichment")
        if not items:
            self.run_store.save_items(run_id, "enriched", [])
            meta = self.run_store.update_meta(
                run_id,
                {
                    "enriched_count": 0,
                    "enrichment_input_count": 0,
                    "enrichment_source_count": 0,
                    **limit_meta,
                    "citation_count": 0,
                    "enrichment_skipped_reason": "empty_input",
                    "local_only": local_only,
                    "status": "completed",
                    "current_stage": "enriched",
                },
            )
            if trace_reporter is not None:
                trace_reporter.event(
                    "enrich",
                    status="skipped",
                    source_stage=source_stage,
                    source_items=0,
                    items_used=0,
                    reason="empty_input",
                )
            return {
                "run_id": run_id,
                "enriched": 0,
                "source_items": 0,
                "items_used": 0,
                "skipped_by_limit": 0,
                "citation_count": 0,
                "skipped": True,
                "skip_reason": "empty_input",
                "artifact": str((self.run_store.run_dir(run_id) / "enriched_items.json").resolve()),
                "meta": meta,
            }
        self.run_store.update_meta(run_id, {"status": "running", "current_stage": "enrich"})
        if trace_reporter is not None:
            trace_reporter.event(
                "enrich",
                status="running",
                source_stage=source_stage,
                source_items=len(items),
                items_used=len(enrichment_input),
            )

        items_to_enrich = enrichment_input
        if self._personal_briefing_enabled(ctx.config) and ctx.config.personal_briefing.enrichment.disable_for_sensitive_topics:
            items_to_enrich = []
            for item in enrichment_input:
                if item.metadata.get("sensitive_topic") or item.metadata.get("sensitive_topic_auto"):
                    item.metadata["enrichment_skipped"] = "sensitive topic"
                else:
                    items_to_enrich.append(item)

        if items_to_enrich:
            ai_client = ctx.runtime.create_ai_client(ctx.config.ai)
            search_result_filter = None
            if (
                self._personal_briefing_enabled(ctx.config)
                and ctx.config.personal_briefing.enrichment.filter_search_results_by_source_policy
            ):
                storage = make_storage(ctx.runtime, ctx.config_path)
                orchestrator = make_orchestrator(ctx.runtime, ctx.config, storage)
                search_result_filter = getattr(orchestrator, "_personal_search_result_allowed", None)
            enricher = self._make_content_enricher(
                ctx.runtime.ContentEnricher,
                ai_client,
                trace_reporter,
                search_result_filter=search_result_filter,
            )
            await enricher.enrich_batch(items_to_enrich)

        self.run_store.save_items(run_id, "enriched", items_to_dicts(enrichment_input))

        citation_count = 0
        for item in enrichment_input:
            citation_count += len(item.metadata.get("sources", []))

        meta = self.run_store.update_meta(
            run_id,
            {
                "enriched_count": len(enrichment_input),
                "enriched_llm_count": len(items_to_enrich),
                "enrichment_input_count": len(enrichment_input),
                "enrichment_source_count": len(items),
                **limit_meta,
                "citation_count": citation_count,
                "local_only": local_only,
                "status": "completed",
                "current_stage": "enriched",
            },
        )

        if trace_reporter is not None:
            trace_reporter.event(
                "enrich",
                status="completed",
                enriched=len(enrichment_input),
                source_items=len(items),
                citation_count=citation_count,
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
        trace_reporter: Any | None = None,
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
        self.run_store.update_meta(run_id, {"status": "running", "current_stage": "summary"})
        if trace_reporter is not None:
            trace_reporter.event(
                "summary",
                status="running",
                language=language,
                source_stage=stage,
                source_items=len(items),
                items_used=len(summary_items),
            )

        if self._uses_personal_summary(ctx.config, language):
            renderer = ctx.runtime.PersonalBriefingRenderer()
            summary_context = {
                "total_fetched": total_fetched,
                "source_items": len(items),
                "selected_count": len(summary_items),
                "threshold": getattr(getattr(ctx.config, "filtering", None), "ai_score_threshold", None),
            }
            summary = self._render_personal_summary(renderer, date_str, summary_items, tracked=[], context=summary_context)
            critic_config = ctx.config.personal_briefing.critic_pass
            if critic_config.enabled:
                critic = ctx.runtime.run_briefing_critic(
                    summary,
                    summary_items,
                    high_confidence_requires_supporting_source=self._high_confidence_requires_supporting_source(ctx.config),
                )
                if (not critic.passed) and critic_config.auto_revise_once:
                    revised_items = drop_items_flagged_by_critic(summary_items, critic)
                    if len(revised_items) < len(summary_items):
                        summary_items = revised_items
                        summary_context["selected_count"] = len(summary_items)
                        summary = self._render_personal_summary(
                            renderer,
                            date_str,
                            summary_items,
                            tracked=[],
                            context=summary_context,
                        )
                        critic = ctx.runtime.run_briefing_critic(
                            summary,
                            summary_items,
                            high_confidence_requires_supporting_source=self._high_confidence_requires_supporting_source(ctx.config),
                        )
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
            "status": "completed",
            "current_stage": "summary",
        }
        if published_path:
            summary_meta["summary_published_path"] = str(Path(published_path).resolve())
        meta = self.run_store.update_meta(run_id, summary_meta)

        if trace_reporter is not None:
            trace_reporter.event(
                "summary",
                status="completed",
                language=language,
                source_stage=stage,
                items_used=len(summary_items),
                skipped_by_limit=max(0, len(items) - len(summary_items)),
            )

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

    @staticmethod
    def _render_personal_summary(
        renderer: Any,
        date: str,
        items: list[Any],
        *,
        tracked: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> str:
        if "context" in signature(renderer.render).parameters:
            return renderer.render(date, items, tracked=tracked, context=context)
        return renderer.render(date, items, tracked=tracked)

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
        trace_reporter: Any | None = None,
    ) -> dict[str, Any]:
        if trace_reporter is not None:
            trace_reporter.event(
                "pipeline.start",
                status="running",
                hours=hours,
                languages=languages or "config",
                sources=sources or "enabled",
                local_only=local_only,
            )

        active_run_id = run_id
        try:
            fetch_result = await self.fetch_items(
                hours=hours,
                run_id=run_id,
                horizon_path=horizon_path,
                config_path=config_path,
                sources=sources,
                run_instructions=run_instructions,
                local_only=local_only,
                trace_reporter=trace_reporter,
            )
            run_id = fetch_result["run_id"]
            active_run_id = run_id

            score_result = await self.score_items(
                run_id=run_id,
                horizon_path=horizon_path,
                config_path=config_path,
                max_items=max_raw_items,
                run_instructions=run_instructions,
                local_only=local_only,
                trace_reporter=trace_reporter,
            )

            filter_result = await self.filter_items(
                run_id=run_id,
                threshold=threshold,
                topic_dedup=topic_dedup,
                horizon_path=horizon_path,
                config_path=config_path,
                run_instructions=run_instructions,
                local_only=local_only,
                trace_reporter=trace_reporter,
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
                    trace_reporter=trace_reporter,
                )
                stage_for_summary = "enriched"
            elif trace_reporter is not None:
                trace_reporter.event("enrich", status="skipped")

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
                    trace_reporter=trace_reporter,
                )
                summaries.append(summary_result)

            result = {
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
                "meta": self.run_store.update_meta(
                    run_id,
                    {
                        "status": "completed",
                        "current_stage": "completed",
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    },
                ),
            }
            if trace_reporter is not None:
                trace_reporter.event("pipeline.finish", status="completed", run_id=run_id)
            return result
        except asyncio.CancelledError:
            if active_run_id:
                try:
                    self.run_store.update_meta(
                        active_run_id,
                        {
                            "status": "cancelled",
                            "cancelled_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                except (OSError, ValueError, TypeError, FileNotFoundError):
                    pass
            if trace_reporter is not None:
                trace_reporter.event("pipeline.cancelled", status="cancelled")
            raise
        except Exception as exc:
            if active_run_id:
                try:
                    self.run_store.update_meta(
                        active_run_id,
                        {
                            "status": "failed",
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                            "error_type": type(exc).__name__,
                        },
                    )
                except (OSError, ValueError, TypeError, FileNotFoundError):
                    pass
            if trace_reporter is not None:
                trace_reporter.event("pipeline.error", status="failed", error=type(exc).__name__)
            raise

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
    def _high_confidence_requires_supporting_source(config: Any) -> bool:
        personal = getattr(config, "personal_briefing", None)
        corroboration = getattr(personal, "corroboration", None)
        return bool(getattr(corroboration, "high_confidence_requires_supporting_source", True))

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
        trace_reporter: Any | None = None,
    ) -> Any:
        kwargs = {"personal_briefing_mode": personal_briefing_mode}
        try:
            params = signature(analyzer_cls).parameters
        except (TypeError, ValueError):
            params = {}
        if "run_instructions" in params:
            kwargs["run_instructions"] = run_instructions
        if trace_reporter is not None and "verbose_reporter" in params:
            kwargs["verbose_reporter"] = trace_reporter
        return analyzer_cls(ai_client, **kwargs)

    @staticmethod
    def _make_content_enricher(
        enricher_cls: Any,
        ai_client: Any,
        trace_reporter: Any | None = None,
        search_result_filter: Any | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {}
        try:
            params = signature(enricher_cls).parameters
        except (TypeError, ValueError):
            params = {}
        if trace_reporter is not None and "verbose_reporter" in params:
            kwargs["verbose_reporter"] = trace_reporter
        if search_result_filter is not None and "search_result_filter" in params:
            kwargs["search_result_filter"] = search_result_filter
        return enricher_cls(ai_client, **kwargs)

    @staticmethod
    def _attach_trace_reporter(target: Any, trace_reporter: Any | None) -> None:
        if trace_reporter is not None and hasattr(target, "verbose_reporter"):
            target.verbose_reporter = trace_reporter

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
