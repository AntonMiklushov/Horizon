"""FastAPI application for the local Horizon dashboard."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import markdown
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import ValidationError

from ...mcp.errors import HorizonMcpError
from ...mcp.horizon_adapter import VALID_SOURCES
from ...mcp.service import HorizonPipelineService
from ...storage.manager import StorageManager
from .config_forms import (
    ConfigFormError,
    apply_basic_settings,
    apply_source_settings,
    source_summary,
    web_default_hours,
)


TEMPLATE_DIR = Path(__file__).with_name("templates")


@dataclass
class WebRunState:
    run_id: str
    status: str = "queued"
    message: str = "Queued"
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


def create_app(
    config_path: str | None = None,
    data_dir: str = "data",
    service: HorizonPipelineService | None = None,
) -> FastAPI:
    """Create the local dashboard app."""

    app = FastAPI(title="Horizon Web", docs_url=None, redoc_url=None)
    app.state.config_path = str(Path(config_path).expanduser().resolve()) if config_path else None
    app.state.data_dir = str(Path(data_dir).expanduser().resolve())
    app.state.storage = StorageManager(data_dir=app.state.data_dir, config_path=app.state.config_path)
    app.state.service = service or HorizonPipelineService(runs_root=Path(app.state.data_dir) / "mcp-runs")
    app.state.runs = {}
    app.state.tasks = set()
    app.state.templates = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        config, error = _load_config(request)
        validation = None
        if config is not None:
            try:
                validation = await request.app.state.service.validate_config(
                    config_path=str(request.app.state.storage.config_path),
                    check_env=True,
                )
            except Exception as exc:
                validation = {"warnings": [str(exc)], "missing_env": []}
        return _render(
            request,
            "dashboard.html",
            config=config,
            config_error=error,
            validation=validation,
            source_counts=source_summary(config) if config else {},
            default_hours=web_default_hours(config) if config else 24,
            runs=request.app.state.service.list_runs(limit=12)["items"],
            active_runs=sorted(request.app.state.runs.values(), key=lambda item: item.started_at, reverse=True),
            sources=sorted(VALID_SOURCES),
        )

    @app.post("/runs")
    async def create_run(request: Request) -> RedirectResponse:
        form = await request.form()
        run_id = f"web-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        state = WebRunState(run_id=run_id, status="running", message="Run started")
        request.app.state.runs[run_id] = state
        params = _run_params_from_form(form, run_id, request)
        task = asyncio.create_task(_run_pipeline(request.app, state, params))
        request.app.state.tasks.add(task)
        task.add_done_callback(request.app.state.tasks.discard)
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: str) -> HTMLResponse:
        run_state = request.app.state.runs.get(run_id)
        meta = _safe_meta(request.app.state.service, run_id)
        stages = _safe_stages(request.app.state.service, run_id)
        summaries = _summary_payloads(request.app.state.service, run_id)
        rendered_summary = _markdown_to_html(summaries[0]["markdown"]) if summaries else ""
        stage_payloads = _stage_payloads(request.app.state.service, run_id, stages, limit=80)
        items = _stage_items(request.app.state.service, run_id, _best_stage(stages), limit=80)
        return _render(
            request,
            "run.html",
            run_id=run_id,
            run_state=run_state,
            meta=meta,
            stages=stages,
            summaries=summaries,
            rendered_summary=rendered_summary,
            stage_payloads=stage_payloads,
            items=items,
        )

    @app.get("/api/runs/{run_id}")
    async def run_api(request: Request, run_id: str) -> JSONResponse:
        run_state = request.app.state.runs.get(run_id)
        payload = {
            "run_id": run_id,
            "state": asdict(run_state) if run_state else None,
            "meta": _safe_meta(request.app.state.service, run_id),
            "stages": _safe_stages(request.app.state.service, run_id),
        }
        return JSONResponse(payload)

    @app.get("/settings/basic", response_class=HTMLResponse)
    async def basic_settings(request: Request) -> HTMLResponse:
        config, error = _load_config(request)
        return _render(
            request,
            "settings_basic.html",
            config=config,
            config_error=error,
            default_hours=web_default_hours(config) if config else 24,
            error=None,
            saved=False,
        )

    @app.post("/settings/basic", response_class=HTMLResponse)
    async def save_basic_settings(request: Request) -> HTMLResponse:
        form = await request.form()
        config, load_error = _load_config(request)
        if config is None:
            return _render(
                request,
                "settings_basic.html",
                config=None,
                config_error=load_error,
                default_hours=24,
                error=load_error,
                saved=False,
            )
        try:
            updated = apply_basic_settings(config, form)
            request.app.state.storage.save_config(updated, backup=True)
            return _render(
                request,
                "settings_basic.html",
                config=updated,
                config_error=None,
                default_hours=web_default_hours(updated),
                error=None,
                saved=True,
            )
        except (ConfigFormError, ValidationError) as exc:
            return _render(
                request,
                "settings_basic.html",
                config=config,
                config_error=None,
                default_hours=web_default_hours(config),
                error=str(exc),
                saved=False,
                status_code=400,
            )

    @app.get("/settings/sources", response_class=HTMLResponse)
    async def source_settings(request: Request) -> HTMLResponse:
        config, error = _load_config(request)
        return _render(
            request,
            "settings_sources.html",
            config=config,
            config_error=error,
            error=None,
            saved=False,
        )

    @app.post("/settings/sources", response_class=HTMLResponse)
    async def save_source_settings(request: Request) -> HTMLResponse:
        form = await request.form()
        config, load_error = _load_config(request)
        if config is None:
            return _render(
                request,
                "settings_sources.html",
                config=None,
                config_error=load_error,
                error=load_error,
                saved=False,
            )
        try:
            updated = apply_source_settings(config, form)
            request.app.state.storage.save_config(updated, backup=True)
            return _render(
                request,
                "settings_sources.html",
                config=updated,
                config_error=None,
                error=None,
                saved=True,
            )
        except (ConfigFormError, ValidationError) as exc:
            return _render(
                request,
                "settings_sources.html",
                config=config,
                config_error=None,
                error=str(exc),
                saved=False,
                status_code=400,
            )

    return app


def _run_params_from_form(form: Any, run_id: str, request: Request) -> dict[str, Any]:
    languages = [part.strip() for part in str(form.get("languages", "")).split(",") if part.strip()]
    sources = form.getlist("sources") if hasattr(form, "getlist") else form.get("sources", [])
    if isinstance(sources, str):
        sources = [sources]
    threshold_text = str(form.get("threshold", "")).strip()
    return {
        "run_id": run_id,
        "hours": int(form.get("hours") or 24),
        "languages": languages or None,
        "threshold": float(threshold_text) if threshold_text else None,
        "sources": list(sources) or None,
        "enrich": "enrich" in form,
        "topic_dedup": "topic_dedup" in form,
        "run_instructions": str(form.get("instructions") or "").strip(),
        "config_path": str(request.app.state.storage.config_path),
        "save_to_horizon_data": False,
        "local_only": True,
    }


async def _run_pipeline(app: FastAPI, state: WebRunState, params: dict[str, Any]) -> None:
    try:
        state.message = "Fetching and analyzing sources"
        state.result = await app.state.service.run_pipeline(**params)
        state.status = "completed"
        state.message = "Report ready"
    except Exception as exc:
        state.status = "failed"
        if isinstance(exc, HorizonMcpError):
            state.error = f"{exc.code}: {exc.message}"
        else:
            state.error = f"{type(exc).__name__}: {exc}"
        state.message = "Run failed"
    finally:
        state.completed_at = datetime.now(timezone.utc).isoformat()


def _load_config(request: Request) -> tuple[Any | None, str | None]:
    try:
        return request.app.state.storage.load_config(), None
    except Exception as exc:
        return None, str(exc)


def _render(request: Request, template_name: str, status_code: int = 200, **context: Any) -> HTMLResponse:
    template = request.app.state.templates.get_template(template_name)
    body = template.render(request=request, **context)
    return HTMLResponse(body, status_code=status_code)


def _safe_meta(service: HorizonPipelineService, run_id: str) -> dict[str, Any] | None:
    try:
        return service.get_run_meta(run_id)["meta"]
    except Exception:
        return None


def _safe_stages(service: HorizonPipelineService, run_id: str) -> dict[str, bool]:
    try:
        run = service.get_run_meta(run_id)
        stages = {}
        for stage in ("raw", "scored", "filtered", "enriched"):
            stages[stage] = service.run_store.has_stage(run_id, stage)
        if run["meta"].get("summary_artifact"):
            stages["summary"] = True
        else:
            stages["summary"] = bool(_summary_payloads(service, run_id))
        return stages
    except Exception:
        return {"raw": False, "scored": False, "filtered": False, "enriched": False, "summary": False}


def _summary_payloads(service: HorizonPipelineService, run_id: str) -> list[dict[str, str]]:
    try:
        run_dir = service.run_store.run_dir(run_id)
    except Exception:
        return []
    summaries = []
    for path in sorted(run_dir.glob("summary-*.md")):
        language = path.stem.replace("summary-", "", 1)
        summaries.append({"language": language, "markdown": path.read_text(encoding="utf-8")})
    return summaries


def _stage_items(service: HorizonPipelineService, run_id: str, stage: str | None, limit: int) -> list[dict[str, Any]]:
    if not stage:
        return []
    try:
        return service.get_run_stage(run_id, stage, max_items=limit)["items"]
    except Exception:
        return []


def _stage_payloads(
    service: HorizonPipelineService,
    run_id: str,
    stages: dict[str, bool],
    limit: int,
) -> list[dict[str, Any]]:
    payloads = []
    for stage in ("raw", "scored", "filtered", "enriched"):
        if not stages.get(stage):
            continue
        items = service._redact_config(_stage_items(service, run_id, stage, limit=limit))
        payloads.append(
            {
                "stage": stage,
                "items": items,
                "json": json.dumps(items, indent=2, sort_keys=True),
            }
        )
    return payloads


def _best_stage(stages: dict[str, bool]) -> str | None:
    for stage in ("enriched", "filtered", "scored", "raw"):
        if stages.get(stage):
            return stage
    return None


def _markdown_to_html(markdown_text: str) -> str:
    return markdown.markdown(markdown_text, extensions=["extra", "sane_lists", "toc"])


def app_from_env() -> FastAPI:
    """Default app object for ASGI runners."""

    return create_ap