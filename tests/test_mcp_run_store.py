from __future__ import annotations

from pathlib import Path

import pytest

from src.mcp.run_store import RunStore


def test_create_run_writes_meta(tmp_path: Path) -> None:
    store = RunStore(tmp_path)

    run_id = store.create_run()
    meta = store.load_meta(run_id)

    assert run_id.startswith("run-")
    assert meta["run_id"] == run_id
    assert "created_at" in meta


def test_save_and_load_stage_items(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run_id = store.create_run("run-fixed")
    items = [{"title": "foo"}, {"title": "bar"}]

    path = store.save_items(run_id, "raw", items)
    loaded = store.load_items(run_id, "raw")

    assert path.name == "raw_items.json"
    assert loaded == items
    assert store.has_stage(run_id, "raw") is True


def test_update_meta_sets_updated_at(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run_id = store.create_run("run-meta")

    meta = store.update_meta(run_id, {"status": "done"})

    assert meta["status"] == "done"
    assert "updated_at" in meta


def test_trace_events_append_and_load_recent_in_order(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run_id = store.create_run("run-trace")

    path = store.append_trace_event(run_id, {"message": "first"})
    store.append_trace_event(run_id, {"message": "second"})
    store.append_trace_event(run_id, {"message": "third"})

    assert path.name == "trace.jsonl"
    assert store.load_trace_events(run_id, limit=2) == [
        {"message": "second"},
        {"message": "third"},
    ]


def test_trace_events_ignore_malformed_partial_lines(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run_id = store.create_run("run-trace-partial")
    trace_path = store.run_dir(run_id) / "trace.jsonl"
    trace_path.write_text(
        '{"message":"first"}\n{"message":\n{"message":"second"}\n',
        encoding="utf-8",
    )

    assert store.load_trace_events(run_id, limit=10) == [
        {"message": "first"},
        {"message": "second"},
    ]


def test_trace_event_limit_must_be_positive(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run_id = store.create_run("run-trace-empty")
    store.append_trace_event(run_id, {"message": "hidden"})

    assert store.load_trace_events(run_id, limit=0) == []


def test_write_json_falls_back_when_atomic_replace_is_denied(
    tmp_path: Path, monkeypatch
) -> None:
    def deny_replace(src, dst):
        raise PermissionError("replace denied")

    monkeypatch.setattr("src.mcp.run_store.os.replace", deny_replace)
    store = RunStore(tmp_path)

    run_id = store.create_run("run-fallback")
    store.update_meta(run_id, {"status": "done"})

    assert store.load_meta(run_id)["status"] == "done"


def test_save_and_load_summary(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run_id = store.create_run("run-summary")

    saved = store.save_summary(run_id, "zh", "# 摘要")
    content = store.load_summary(run_id, "zh")

    assert saved.name == "summary-zh.md"
    assert content == "# 摘要"


def test_unsupported_stage_raises(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run_id = store.create_run("run-invalid-stage")

    with pytest.raises(ValueError, match="Unsupported stage"):
        store.save_items(run_id, "unknown", [])


def test_missing_run_raises(tmp_path: Path) -> None:
    store = RunStore(tmp_path)

    with pytest.raises(FileNotFoundError, match="Run not found"):
        store.run_dir("missing-run")


@pytest.mark.parametrize("bad_run_id", ["../outside", "..\\outside", "/abs", "run/child", ""])
def test_run_id_must_be_slug(tmp_path: Path, bad_run_id: str) -> None:
    store = RunStore(tmp_path)

    with pytest.raises(ValueError, match="Invalid run_id"):
        store.create_run(bad_run_id)


@pytest.mark.parametrize("bad_language", ["../ru", "ru/../../x", ""])
def test_summary_language_must_be_slug(tmp_path: Path, bad_language: str) -> None:
    store = RunStore(tmp_path)
    run_id = store.create_run("run-summary")

    with pytest.raises(ValueError, match="Invalid language"):
        store.save_summary(run_id, bad_language, "# summary")


def test_missing_artifact_raises(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run_id = store.create_run("run-missing-file")

    with pytest.raises(FileNotFoundError, match="Artifact not found"):
        store.read_json(run_id, "does-not-exist.json")


def test_list_runs_returns_desc_order(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    run1 = store.create_run("run-1")
    store.update_meta(run1, {"seq": 1})
    run2 = store.create_run("run-2")
    store.update_meta(run2, {"seq": 2})

    runs = store.list_runs(limit=10)

    assert runs[0]["run_id"] == "run-2"
    assert runs[1]["run_id"] == "run-1"
