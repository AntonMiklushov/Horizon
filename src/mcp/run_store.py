"""Run artifact persistence for Horizon Brief MCP."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


STAGES = {
    "raw": "raw_items.json",
    "scored": "scored_items.json",
    "filtered": "filtered_items.json",
    "enriched": "enriched_items.json",
}
TRACE_FILE = "trace.jsonl"
SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass
class RunStore:
    """Store intermediate artifacts per pipeline run."""

    root: Path

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(self, run_id: str | None = None) -> str:
        if run_id is None:
            run_id = self._make_run_id()
        self._validate_slug(run_id, "run_id")
        run_dir = self.root / run_id
        self._assert_under_root(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            self.write_json(run_id, "meta.json", {"run_id": run_id, "created_at": self._utc_now()})
        return run_id

    def run_dir(self, run_id: str) -> Path:
        self._validate_slug(run_id, "run_id")
        path = self.root / run_id
        self._assert_under_root(path)
        if not path.exists():
            raise FileNotFoundError(f"Run not found: {run_id}")
        return path

    def has_stage(self, run_id: str, stage: str) -> bool:
        return (self.run_dir(run_id) / self._stage_file(stage)).exists()

    def save_items(self, run_id: str, stage: str, items: list[dict[str, Any]]) -> Path:
        return self.write_json(run_id, self._stage_file(stage), items)

    def load_items(self, run_id: str, stage: str) -> list[dict[str, Any]]:
        return self.read_json(run_id, self._stage_file(stage))

    def save_summary(self, run_id: str, language: str, markdown: str) -> Path:
        self._validate_slug(language, "language")
        filename = f"summary-{language}.md"
        path = self.run_dir(run_id) / filename
        self._write_text(path, markdown)
        return path

    def load_summary(self, run_id: str, language: str) -> str:
        self._validate_slug(language, "language")
        path = self.run_dir(run_id) / f"summary-{language}.md"
        self._assert_under_root(path)
        if not path.exists():
            raise FileNotFoundError(f"Summary not found: run={run_id} lang={language}")
        return path.read_text(encoding="utf-8")

    def update_meta(self, run_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        meta = self.read_json(run_id, "meta.json")
        meta.update(updates)
        meta["updated_at"] = self._utc_now()
        self.write_json(run_id, "meta.json", meta)
        return meta

    def load_meta(self, run_id: str) -> dict[str, Any]:
        return self.read_json(run_id, "meta.json")

    def append_trace_event(self, run_id: str, event: dict[str, Any]) -> Path:
        """Append one observable activity event to the run trace."""

        path = self.run_dir(run_id) / TRACE_FILE
        self._assert_under_root(path)
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        return path

    def load_trace_events(self, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """Load recent run trace events, ignoring malformed partial JSONL lines."""

        if limit <= 0:
            return []

        path = self.run_dir(run_id) / TRACE_FILE
        self._assert_under_root(path)
        if not path.exists():
            return []

        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events[-limit:]

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """List runs sorted by create/update time descending."""

        entries: list[dict[str, Any]] = []
        for run_dir in self.root.iterdir():
            if not run_dir.is_dir():
                continue
            meta_path = run_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            created = meta.get("created_at") or ""
            updated = meta.get("updated_at") or created
            entries.append(
                {
                    "run_id": meta.get("run_id", run_dir.name),
                    "created_at": created,
                    "updated_at": updated,
                    "meta": meta,
                }
            )

        entries.sort(key=lambda x: x["updated_at"] or x["created_at"], reverse=True)
        return entries[: max(0, limit)]

    def write_json(self, run_id: str, filename: str, payload: Any) -> Path:
        path = self.run_dir(run_id) / filename
        self._assert_under_root(path)
        self._write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
        return path

    def read_json(self, run_id: str, filename: str) -> Any:
        path = self.run_dir(run_id) / filename
        self._assert_under_root(path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: run={run_id} file={filename}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _stage_file(stage: str) -> str:
        if stage not in STAGES:
            supported = ", ".join(sorted(STAGES))
            raise ValueError(f"Unsupported stage '{stage}', expected one of: {supported}")
        return STAGES[stage]

    @staticmethod
    def _make_run_id() -> str:
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"run-{now}-{uuid4().hex[:8]}"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _validate_slug(value: str, field: str) -> None:
        if not SLUG_RE.fullmatch(value):
            raise ValueError(f"Invalid {field}: {value!r}")

    def _assert_under_root(self, path: Path) -> None:
        resolved = path.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError(f"Path escapes run store root: {resolved}")

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        last_error: PermissionError | None = None
        for attempt in range(3):
            tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            try:
                tmp_path.write_text(text, encoding="utf-8")
                os.replace(tmp_path, path)
                return
            except PermissionError as exc:
                last_error = exc
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except OSError:
                    pass
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                try:
                    path.write_text(text, encoding="utf-8")
                    return
                except PermissionError:
                    raise last_error
        if last_error is not None:
            raise last_error
