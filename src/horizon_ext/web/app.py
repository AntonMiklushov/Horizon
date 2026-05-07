"""FastAPI application for the local Horizon Brief dashboard."""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import markdown
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import ValidationError

from ...mcp.errors import HorizonMcpError
from ...mcp.horizon_adapter import VALID_SOURCES, get_enabled_sources
from ...mcp.service import HorizonPipelineService
from ...models import Config
from ...storage.manager import StorageManager
from .config_forms import (
    ConfigFormError,
    apply_basic_settings,
    apply_policy_settings,
    apply_source_settings,
    source_summary,
    web_default_hours,
)


TEMPLATE_DIR = Path(__file__).with_name("templates")
LOCAL_LLM_DEFAULT_AI_ITEM_CAP = 8


@dataclass
class WebRunState:
    run_id: str
    status: str = "queued"
    message: str = "Queued"
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    cancel_requested_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


def create_app(
    config_path: str | None = None,
    data_dir: str = "data",
    service: HorizonPipelineService | None = None,
) -> FastAPI:
    """Create the local dashboard app."""

    app = FastAPI(title="Horizon Brief Web", docs_url=None, redoc_url=None)
    resolved_data_dir = Path(data_dir).expanduser().resolve()
    source_config_path = (
        Path(config_path).expanduser().resolve()
        if config_path
        else (resolved_data_dir / "config.json").resolve()
    )
    active_config_path = _resolve_web_config_path(source_config_path)
    app.state.config_source_path = str(source_config_path)
    app.state.config_fallback_active = active_config_path != source_config_path
    app.state.config_path = str(active_config_path)
    app.state.data_dir = str(resolved_data_dir)
    app.state.storage = StorageManager(data_dir=app.state.data_dir, config_path=app.state.config_path)
    app.state.runs_root = str(_resolve_runs_root(Path(app.state.data_dir)))
    app.state.service = service or HorizonPipelineService(runs_root=Path(app.state.runs_root))
    app.state.runs = {}
    app.state.tasks = set()
    app.state.run_tasks = {}
    app.state.templates = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        return _render(
            request,
            "dashboard.html",
            **await _dashboard_context(request),
        )

    @app.post("/runs")
    async def create_run(request: Request) -> Any:
        form = await request.form()
        run_id = f"web-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        try:
            params = _run_params_from_form(form, run_id, request)
        except ConfigFormError as exc:
            return _render(
                request,
                "dashboard.html",
                status_code=400,
                **await _dashboard_context(request, run_error=str(exc)),
            )

        state = WebRunState(run_id=run_id, status="running", message="Run started")
        request.app.state.runs[run_id] = state
        task = asyncio.create_task(_run_pipeline(request.app, state, params))
        request.app.state.tasks.add(task)
        request.app.state.run_tasks[run_id] = task
        task.add_done_callback(lambda completed, app=request.app, state=state: _cleanup_run_task(app, state, completed))
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)

    @app.post("/runs/{run_id}/cancel")
    async def cancel_run(request: Request, run_id: str) -> RedirectResponse:
        run_state = request.app.state.runs.get(run_id)
        task = request.app.state.run_tasks.get(run_id)
        if run_state is not None and run_state.status in {"queued", "running", "cancelling"}:
            run_state.status = "cancelling"
            run_state.message = "Cancellation requested"
            run_state.cancel_requested_at = datetime.now(timezone.utc).isoformat()
            if task is None or task.done():
                run_state.status = "cancelled"
                run_state.message = "Run cancelled"
                run_state.completed_at = datetime.now(timezone.utc).isoformat()
            else:
                task.cancel()
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: str) -> HTMLResponse:
        run_state = request.app.state.runs.get(run_id)
        meta = _safe_meta(request.app.state.service, run_id)
        stages = _safe_stages(request.app.state.service, run_id)
        trace_events = _safe_trace_events(request.app.state.service, run_id, limit=80)
        summaries = _summary_payloads(request.app.state.service, run_id)
        rendered_summary = _markdown_to_html(summaries[0]["markdown"]) if summaries else ""
        stage_payloads = _stage_payloads(request.app.state.service, run_id, stages, limit=80)
        items = _stage_items(request.app.state.service, run_id, _best_stage(stages), limit=80)
        return _render(
            request,
            "run.html",
            status_code=404 if run_state is None and meta is None else 200,
            run_id=run_id,
            run_state=run_state,
            meta=meta,
            stages=stages,
            trace_events=trace_events,
            recent_trace_events=trace_events[-8:],
            activity=_activity_snapshot(stages, trace_events, run_state),
            activity_stages=_activity_stages(stages, trace_events, run_state),
            summaries=summaries,
            rendered_summary=rendered_summary,
            stage_payloads=stage_payloads,
            items=items,
            run_failure_diagnostic=_run_failure_diagnostic(trace_events, run_state),
            personal_prefilter_note=_personal_prefilter_note(meta),
            personal_prefilter_exclusions=_personal_prefilter_exclusions(meta),
            source_policy_decisions=_source_policy_decisions(request.app.state.service, run_id),
        )

    @app.get("/api/runs/{run_id}")
    async def run_api(request: Request, run_id: str) -> JSONResponse:
        run_state = request.app.state.runs.get(run_id)
        stages = _safe_stages(request.app.state.service, run_id)
        trace_events = _safe_trace_events(request.app.state.service, run_id, limit=120)
        activity = _activity_snapshot(stages, trace_events, run_state)
        payload = {
            "run_id": run_id,
            "state": _run_state_payload(run_state, activity),
            "meta": _safe_meta(request.app.state.service, run_id),
            "stages": stages,
            "trace": trace_events,
            "activity": activity,
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
            _save_web_config(request.app, updated)
            return _render(
                request,
                "settings_basic.html",
                config=updated,
                config_error=None,
                default_hours=web_default_hours(updated),
                error=None,
                saved=True,
            )
        except (ConfigFormError, ValidationError, OSError) as exc:
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
            source_counts=source_summary(config) if config else {},
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
            _save_web_config(request.app, updated)
            return _render(
                request,
                "settings_sources.html",
                config=updated,
                config_error=None,
                source_counts=source_summary(updated),
                error=None,
                saved=True,
            )
        except (ConfigFormError, ValidationError, OSError) as exc:
            return _render(
                request,
                "settings_sources.html",
                config=config,
                config_error=None,
                source_counts=source_summary(config),
                error=str(exc),
                saved=False,
                status_code=400,
            )

    @app.get("/settings/policy", response_class=HTMLResponse)
    async def policy_settings(request: Request) -> HTMLResponse:
        config, error = _load_config(request)
        return _render(
            request,
            "settings_policy.html",
            config=config,
            config_error=error,
            error=None,
            saved=False,
            policy_preset=_policy_preset(config),
        )

    @app.post("/settings/policy", response_class=HTMLResponse)
    async def save_policy_settings(request: Request) -> HTMLResponse:
        form = await request.form()
        config, load_error = _load_config(request)
        if config is None:
            return _render(
                request,
                "settings_policy.html",
                config=None,
                config_error=load_error,
                error=load_error,
                saved=False,
                policy_preset="custom",
            )
        try:
            updated = apply_policy_settings(config, form)
            _save_web_config(request.app, updated)
            return _render(
                request,
                "settings_policy.html",
                config=updated,
                config_error=None,
                error=None,
                saved=True,
                policy_preset=_policy_preset(updated),
            )
        except (ConfigFormError, ValidationError, OSError) as exc:
            return _render(
                request,
                "settings_policy.html",
                config=config,
                config_error=None,
                error=str(exc),
                saved=False,
                policy_preset=_policy_preset(config),
                status_code=400,
            )

    @app.get("/settings/config", response_class=HTMLResponse)
    async def config_editor(request: Request) -> HTMLResponse:
        config, error = _load_config(request)
        return _render(
            request,
            "settings_config.html",
            config=config,
            config_error=error,
            config_json=_config_editor_json(config),
            error=None,
            saved=False,
            saved_target=None,
        )

    @app.post("/settings/config", response_class=HTMLResponse)
    async def save_config_editor(request: Request) -> HTMLResponse:
        form = await request.form()
        raw_json = str(form.get("config_json") or "")
        target = str(form.get("save_target") or "primary")
        try:
            payload = json.loads(raw_json)
            updated = Config.model_validate(payload)
            if target == "runtime":
                saved_path = request.app.state.storage.save_config(updated, backup=True)
            else:
                saved_path = _save_source_config(request.app, updated)
            return _render(
                request,
                "settings_config.html",
                config=updated,
                config_error=None,
                config_json=_config_editor_json(updated),
                error=None,
                saved=True,
                saved_target=str(saved_path),
            )
        except (json.JSONDecodeError, ValidationError, OSError) as exc:
            config, load_error = _load_config(request)
            return _render(
                request,
                "settings_config.html",
                config=config,
                config_error=load_error,
                config_json=raw_json,
                error=str(exc),
                saved=False,
                saved_target=None,
                status_code=400,
            )

    return app


def _run_params_from_form(form: Any, run_id: str, request: Request) -> dict[str, Any]:
    languages = [part.strip() for part in str(form.get("languages", "")).split(",") if part.strip()]
    sources = form.getlist("sources") if hasattr(form, "getlist") else form.get("sources", [])
    if isinstance(sources, str):
        sources = [sources]
    threshold_text = str(form.get("threshold", "")).strip()
    max_ai_items = _max_ai_items_from_form(form, request)
    params = {
        "run_id": run_id,
        "hours": _bounded_int(form, "hours", "Hours", default=24, minimum=1, maximum=720),
        "languages": languages or None,
        "threshold": _bounded_float_text(
            threshold_text,
            "Threshold",
            minimum=0.0,
            maximum=10.0,
        ) if threshold_text else None,
        "sources": list(sources) or None,
        "enrich": "enrich" in form,
        "topic_dedup": "topic_dedup" in form,
        "run_instructions": str(form.get("instructions") or "").strip(),
        "config_path": str(request.app.state.storage.config_path),
        "save_to_horizon_data": False,
        "local_only": True,
    }
    if max_ai_items is not None:
        params["max_raw_items"] = max_ai_items
        params["max_filtered_items"] = max_ai_items
    return params


def _max_ai_items_from_form(form: Any, request: Request) -> int | None:
    value = str(form.get("max_ai_items", "")).strip()
    if not value:
        config, _ = _load_config(request)
        return _default_ai_item_cap(config)
    return _bounded_int_text(value, "AI Item Cap", minimum=1, maximum=200)


def _bounded_int(
    form: Any,
    key: str,
    label: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = str(form.get(key, "")).strip()
    if not value:
        return default
    return _bounded_int_text(value, label, minimum=minimum, maximum=maximum)


def _bounded_int_text(value: str, label: str, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise ConfigFormError(f"{label} must be an integer.") from exc
    if number < minimum:
        raise ConfigFormError(f"{label} must be at least {minimum}.")
    if number > maximum:
        raise ConfigFormError(f"{label} must be at most {maximum}.")
    return number


def _bounded_float_text(value: str, label: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ConfigFormError(f"{label} must be a number.") from exc
    if number < minimum:
        raise ConfigFormError(f"{label} must be at least {minimum:g}.")
    if number > maximum:
        raise ConfigFormError(f"{label} must be at most {maximum:g}.")
    return number


def _default_ai_item_cap(config: Any | None) -> int | None:
    if _is_local_openai_config(config):
        return LOCAL_LLM_DEFAULT_AI_ITEM_CAP
    return None


def _is_local_openai_config(config: Any | None) -> bool:
    if config is None:
        return False
    provider = getattr(getattr(config, "ai", None), "provider", None)
    provider_value = getattr(provider, "value", str(provider))
    if provider_value != "openai":
        return False
    base_url = getattr(config.ai, "base_url", None)
    if not base_url:
        return False
    return urlparse(str(base_url)).hostname in {"127.0.0.1", "localhost", "::1"}


def _resolve_web_config_path(config_path: Path) -> Path:
    if not config_path.exists() or _can_write_file(config_path):
        return config_path
    fallback_path = _web_config_fallback_path(config_path)
    _copy_config_for_web(config_path, fallback_path)
    return fallback_path


def _can_write_file(path: Path) -> bool:
    if not path.exists():
        return _can_write_directory(path.parent)
    try:
        with path.open("r+", encoding="utf-8"):
            pass
        return True
    except OSError:
        return False


def _web_config_fallback_path(config_path: Path) -> Path:
    repo_root = config_path.parent.parent
    candidates = [
        Path(os.getenv("LOCALAPPDATA") or tempfile.gettempdir()) / "HorizonBrief" / "web-config",
        Path(tempfile.gettempdir()) / "HorizonBrief" / "web-config",
        repo_root / ".codex-tmp-web" / "web-config",
    ]
    fallback_base = next((path for path in candidates if _can_write_directory(path)), candidates[-1])
    digest = hashlib.sha256(str(config_path).encode("utf-8")).hexdigest()[:12]
    return fallback_base / f"{digest}-config.json"


def _copy_config_for_web(source_path: Path, fallback_path: Path) -> None:
    if fallback_path.exists():
        return
    payload = json.loads(source_path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        _make_web_config_paths_absolute(payload, source_path)
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallback_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _save_web_config(app: FastAPI, config: Config) -> Path:
    source_path = Path(app.state.config_source_path)
    if bool(getattr(app.state, "config_fallback_active", False)) and _can_write_file(source_path):
        return _save_source_config(app, config)
    return app.state.storage.save_config(config, backup=True)


def _save_source_config(app: FastAPI, config: Config) -> Path:
    source_path = Path(app.state.config_source_path)
    source_storage = StorageManager(data_dir=app.state.data_dir, config_path=str(source_path))
    saved_path = source_storage.save_config(config, backup=True)
    _activate_config_path(app, saved_path)
    return saved_path


def _activate_config_path(app: FastAPI, config_path: Path) -> None:
    app.state.config_path = str(config_path)
    app.state.config_fallback_active = Path(app.state.config_source_path) != config_path
    app.state.storage = StorageManager(data_dir=app.state.data_dir, config_path=str(config_path))


def _config_editor_json(config: Any | None) -> str:
    if config is None:
        return ""
    return json.dumps(config.model_dump(mode="json"), indent=2, ensure_ascii=False)


def _make_web_config_paths_absolute(payload: dict[str, Any], source_path: Path) -> None:
    repo_root = source_path.parent.parent
    personal = payload.get("personal_briefing")
    if isinstance(personal, dict):
        personal["source_policy_file"] = _absolute_runtime_path(
            personal.get("source_policy_file"),
            repo_root,
        )
    publishing = payload.get("publishing")
    if isinstance(publishing, dict):
        publishing["docs_dir"] = _absolute_runtime_path(publishing.get("docs_dir"), repo_root)
    rendering = payload.get("rendering")
    if isinstance(rendering, dict) and rendering.get("template_dir"):
        rendering["template_dir"] = _absolute_runtime_path(rendering.get("template_dir"), repo_root)


def _absolute_runtime_path(value: Any, repo_root: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


def _resolve_runs_root(data_dir: Path) -> Path:
    fallback_base = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    candidates = [
        data_dir / "mcp-runs",
        fallback_base / "HorizonBrief" / "mcp-runs",
        Path(tempfile.gettempdir()) / "HorizonBrief" / "mcp-runs",
    ]
    for candidate in candidates:
        if _can_write_directory(candidate):
            return candidate
    return candidates[0]


def _can_write_directory(path: Path) -> bool:
    probe = path / f".write-test-{uuid4().hex}"
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe.mkdir()
        return True
    except OSError:
        return False
    finally:
        try:
            if probe.exists():
                probe.rmdir()
        except OSError:
            pass


async def _run_pipeline(app: FastAPI, state: WebRunState, params: dict[str, Any]) -> None:
    trace_reporter = None
    try:
        if hasattr(app.state.service, "create_trace_reporter"):
            trace_reporter = app.state.service.create_trace_reporter(state.run_id)
        state.message = "Fetching and analyzing sources"
        if trace_reporter is not None:
            params = {**params, "trace_reporter": trace_reporter}
        state.result = await app.state.service.run_pipeline(**params)
        state.status = "completed"
        state.message = "Report ready"
    except asyncio.CancelledError:
        state.status = "cancelled"
        state.message = "Run cancelled"
        if trace_reporter is not None:
            trace_reporter.event("pipeline.cancelled", status="cancelled", message="Run cancelled")
    except Exception as exc:
        state.status = "failed"
        if isinstance(exc, HorizonMcpError):
            state.error = f"{exc.code}: {exc.message}"
        else:
            state.error = f"{type(exc).__name__}: {exc}"
        state.message = "Run failed"
    finally:
        state.completed_at = datetime.now(timezone.utc).isoformat()


def _cleanup_run_task(app: FastAPI, state: WebRunState, task: asyncio.Task[Any]) -> None:
    app.state.tasks.discard(task)
    app.state.run_tasks.pop(state.run_id, None)
    if task.cancelled() and state.status in {"queued", "running", "cancelling"}:
        state.status = "cancelled"
        state.message = "Run cancelled"
        state.completed_at = datetime.now(timezone.utc).isoformat()


def _run_state_payload(run_state: WebRunState | None, activity: dict[str, Any]) -> dict[str, Any] | None:
    if run_state is None:
        return None
    payload = asdict(run_state)
    if run_state.status in {"queued", "running", "cancelling"} and activity.get("message"):
        payload["message"] = activity["message"]
    return payload


def _load_config(request: Request) -> tuple[Any | None, str | None]:
    try:
        return request.app.state.storage.load_config(), None
    except Exception as exc:
        return None, str(exc)


async def _dashboard_context(request: Request, run_error: str | None = None) -> dict[str, Any]:
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
    return {
        "config": config,
        "config_error": error,
        "validation": validation,
        "run_error": run_error,
        "source_counts": source_summary(config) if config else {},
        "enabled_sources": get_enabled_sources(config) if config else [],
        "provider_label": _provider_label(config) if config else "",
        "default_hours": web_default_hours(config) if config else 24,
        "default_ai_item_cap": _default_ai_item_cap(config),
        "personal_prefilter_active": _personal_prefilter_enabled(config),
        "runs": request.app.state.service.list_runs(limit=12)["items"],
        "active_runs": sorted(
            (
                run
                for run in request.app.state.runs.values()
                if run.status in {"queued", "running", "cancelling"}
            ),
            key=lambda item: item.started_at,
            reverse=True,
        ),
        "sources": sorted(VALID_SOURCES),
    }


def _provider_label(config: Any) -> str:
    if _is_local_openai_config(config):
        return "LM Studio"
    provider = getattr(getattr(config, "ai", None), "provider", None)
    provider_value = getattr(provider, "value", str(provider))
    labels = {
        "codex_cli": "Codex CLI",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "gemini": "Gemini",
        "ali": "Ali DashScope",
        "doubao": "Doubao",
        "minimax": "MiniMax",
    }
    return labels.get(provider_value, provider_value)


def _personal_prefilter_enabled(config: Any | None) -> bool:
    personal = getattr(config, "personal_briefing", None)
    return bool(getattr(personal, "enabled", False))


def _policy_preset(config: Any | None) -> str:
    personal = getattr(config, "personal_briefing", None)
    path = str(getattr(personal, "source_policy_file", "") or "")
    normalized = path.replace("\\", "/")
    if normalized.endswith("source-policy.personal-media.example.json"):
        return "personal-media"
    if normalized.endswith("config.personal-news.example.json"):
        return "personal-news"
    return "custom"


def _personal_prefilter_note(meta: dict[str, Any] | None) -> str | None:
    if not meta:
        return None
    raw_before = meta.get("raw_count_before_merge")
    raw_after = meta.get("raw_count_after_personal_prefilter")
    excluded = meta.get("personal_prefilter_excluded") or []
    if not isinstance(raw_before, int) or not isinstance(raw_after, int):
        return None
    if raw_after >= raw_before:
        return None
    if raw_after == 0 and raw_before > 0:
        return f"Personal briefing prefilter excluded all {raw_before} fetched items before scoring."
    if excluded:
        return f"Personal briefing prefilter kept {raw_after} of {raw_before} fetched items before scoring."
    return None


def _personal_prefilter_exclusions(meta: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not meta:
        return []
    excluded = meta.get("personal_prefilter_excluded") or meta.get("personal_filter_excluded") or []
    return excluded if isinstance(excluded, list) else []


def _source_policy_decisions(service: HorizonPipelineService, run_id: str) -> dict[str, Any] | None:
    try:
        payload = service.run_store.read_json(run_id, "source_policy_decisions.json")
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _render(request: Request, template_name: str, status_code: int = 200, **context: Any) -> HTMLResponse:
    template = request.app.state.templates.get_template(template_name)
    context = {
        "config_fallback_active": bool(getattr(request.app.state, "config_fallback_active", False)),
        "config_source_path": getattr(request.app.state, "config_source_path", ""),
        "config_active_path": getattr(request.app.state, "config_path", ""),
        **context,
    }
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


def _safe_trace_events(service: HorizonPipelineService, run_id: str, limit: int) -> list[dict[str, Any]]:
    try:
        return service.get_run_trace(run_id, max_events=limit)["events"]
    except Exception:
        return []


def _run_failure_diagnostic(
    trace_events: list[dict[str, Any]],
    run_state: WebRunState | None,
) -> str | None:
    if run_state is None or run_state.status != "failed":
        return None
    for status in ("failed", "warning"):
        for event in reversed(trace_events):
            if event.get("name") == "pipeline.error" or event.get("status") != status:
                continue
            message = _trace_event_diagnostic_message(event)
            if not message or (run_state.error and message == run_state.error):
                continue
            return _redact_diagnostic_text(message)
    return None


def _trace_event_diagnostic_message(event: dict[str, Any]) -> str | None:
    fields = event.get("fields")
    if event.get("name") == "fetch" and not fields:
        return None
    message = str(event.get("message") or event.get("name") or "Activity failed").strip()
    error = fields.get("error") if isinstance(fields, dict) else None
    detail = fields.get("detail") if isinstance(fields, dict) else None
    diagnostics = fields.get("diagnostics") if isinstance(fields, dict) else None
    if error:
        message = f"{message}: {error}"
    if detail:
        message = f"{message}: {detail}"
    elif diagnostics_summary := _trace_diagnostics_summary(diagnostics):
        message = f"{message}: {diagnostics_summary}"
    return message


def _trace_diagnostics_summary(diagnostics: Any) -> str | None:
    if not isinstance(diagnostics, list) or not diagnostics:
        return None
    first = diagnostics[0]
    if not isinstance(first, dict):
        return f"{len(diagnostics)} source diagnostics"
    source = str(first.get("source") or "source").strip()
    error = str(first.get("error") or first.get("status") or "warning").strip()
    message = str(first.get("message") or "").strip()
    summary = f"{len(diagnostics)} source diagnostics; first: {source}: {error}"
    if message:
        summary = f"{summary}: {message}"
    return summary


def _redact_diagnostic_text(value: str) -> str:
    redacted = re.sub(
        r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
        r"\1<redacted>",
        value,
    )
    redacted = re.sub(
        r"(?i)\b(api[_-]?key|token|secret|password)(\s*[:=]\s*)[^\s,;]+",
        r"\1\2<redacted>",
        redacted,
    )
    redacted = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "sk-<redacted>", redacted)
    if len(redacted) > 500:
        return redacted[:497] + "..."
    return redacted


def _activity_snapshot(
    stages: dict[str, bool],
    trace_events: list[dict[str, Any]],
    run_state: WebRunState | None,
) -> dict[str, Any]:
    latest = trace_events[-1] if trace_events else {}
    stage = str(latest.get("stage") or _current_activity_stage(stages, run_state))
    message = str(latest.get("message") or "")
    if not message and run_state is not None:
        message = run_state.message
    if not message:
        message = "Saved run artifacts are available" if any(stages.values()) else "No activity is available yet"
    return {
        "stage": stage,
        "message": message,
        "progress_percent": _activity_progress_percent(stages, latest, run_state),
    }


def _activity_stages(
    stages: dict[str, bool],
    trace_events: list[dict[str, Any]],
    run_state: WebRunState | None,
) -> list[dict[str, Any]]:
    current = _activity_snapshot(stages, trace_events, run_state)["stage"]
    labels = [
        ("fetch", "Fetch", stages.get("raw", False)),
        ("analyze", "Analyze", stages.get("scored", False)),
        ("filter", "Filter", stages.get("filtered", False)),
        ("enrich", "Enrich", stages.get("enriched", False)),
        ("summary", "Summary", stages.get("summary", False)),
    ]
    return [
        {"key": key, "label": label, "complete": bool(complete), "active": key == current}
        for key, label, complete in labels
    ]


def _current_activity_stage(stages: dict[str, bool], run_state: WebRunState | None) -> str:
    if run_state and run_state.status == "completed":
        return "summary"
    if not stages.get("raw"):
        return "fetch"
    if not stages.get("scored"):
        return "analyze"
    if not stages.get("filtered"):
        return "filter"
    if not stages.get("enriched") and not stages.get("summary"):
        return "enrich"
    return "summary"


def _activity_progress_percent(
    stages: dict[str, bool],
    latest: dict[str, Any],
    run_state: WebRunState | None,
) -> int:
    if run_state and run_state.status == "completed":
        return 100
    stage_order = ["fetch", "analyze", "filter", "enrich", "summary"]
    completed = {
        "fetch": stages.get("raw", False),
        "analyze": stages.get("scored", False),
        "filter": stages.get("filtered", False),
        "enrich": stages.get("enriched", False),
        "summary": stages.get("summary", False),
    }
    progress = sum(1 for stage in stage_order if completed[stage]) / len(stage_order)
    event_stage = latest.get("stage")
    event_progress = latest.get("progress")
    if event_stage in stage_order and isinstance(event_progress, dict):
        current = event_progress.get("current")
        total = event_progress.get("total")
        if isinstance(current, (int, float)) and isinstance(total, (int, float)) and total > 0:
            stage_index = stage_order.index(event_stage)
            progress = max(progress, (stage_index + min(max(current / total, 0), 1)) / len(stage_order))
    return int(min(max(progress * 100, 0), 100))


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
    escaped = html.escape(markdown_text)
    return markdown.markdown(escaped, extensions=["extra", "sane_lists", "toc"])


def app_from_env() -> FastAPI:
    """Default app object for ASGI runners."""

    return create_app()
