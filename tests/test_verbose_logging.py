from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from src.main import build_parser
from src.models import AIConfig, AIProvider, Config, FilteringConfig, SourcesConfig
from src.orchestrator import HorizonOrchestrator
from src.storage.manager import StorageManager


class CaptureConsole:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def print(self, *objects, **kwargs) -> None:
        self.messages.append(" ".join(str(obj) for obj in objects))


class CaptureReporter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def start(self, stage: str, **fields):
        self.calls.append(("start", stage, fields))
        return 1.0

    def end(self, stage: str, started_at=None, **fields):
        self.calls.append(("end", stage, fields))

    def event(self, name: str, **fields):
        self.calls.append(("event", name, fields))


def _config() -> Config:
    return Config(
        version="1",
        ai=AIConfig(
            provider=AIProvider.CODEX_CLI,
            model="codex-cli",
            languages=["ru"],
        ),
        sources=SourcesConfig(),
        filtering=FilteringConfig(),
    )


def test_cli_parser_accepts_verbose_flag() -> None:
    args = build_parser().parse_args(["--hours", "1", "--config", "data/config.json", "--verbose"])

    assert args.verbose is True
    assert args.hours == 1
    assert args.config == "data/config.json"


def test_cli_parser_accepts_short_verbose_flag() -> None:
    args = build_parser().parse_args(["-v"])

    assert args.verbose is True


def test_orchestrator_verbose_false_suppresses_verbose_messages() -> None:
    orchestrator = HorizonOrchestrator(
        _config(),
        StorageManager(data_dir="data"),
        verbose=False,
    )
    console = CaptureConsole()
    orchestrator.verbose_reporter.console = console

    orchestrator.verbose_reporter.event("stage.test", count=1)

    assert console.messages == []


def test_orchestrator_verbose_true_prints_stage_counters_and_redacts() -> None:
    orchestrator = HorizonOrchestrator(
        _config(),
        StorageManager(data_dir="data"),
        verbose=True,
    )
    console = CaptureConsole()
    orchestrator.verbose_reporter.console = console

    started_at = orchestrator.verbose_reporter.start("stage.test", count=2)
    orchestrator.verbose_reporter.event("stage.secret", api_key="secret-value", token="token-value")
    orchestrator.verbose_reporter.end("stage.test", started_at, saved=1)

    output = "\n".join(console.messages)
    assert "VERBOSE START stage.test" in output
    assert "VERBOSE stage.secret" in output
    assert "count=2" in output
    assert "saved=1" in output
    assert "<redacted>" in output
    assert "secret-value" not in output
    assert "token-value" not in output


def test_fetch_with_progress_records_failed_source(tmp_path) -> None:
    class BrokenScraper:
        fetch_diagnostics = [
            {"source": "Feed", "error": "ConnectError", "message": "Connection failed"},
        ]

        async def fetch(self, since):  # type: ignore[no-untyped-def]
            raise RuntimeError("source down")

    orchestrator = HorizonOrchestrator(
        _config(),
        StorageManager(data_dir=str(tmp_path / "data")),
    )
    reporter = CaptureReporter()
    orchestrator.verbose_reporter = reporter  # type: ignore[assignment]
    orchestrator.console = CaptureConsole()

    with pytest.raises(RuntimeError, match="source down"):
        asyncio.run(orchestrator._fetch_with_progress("RSS Feeds", BrokenScraper(), datetime.now(timezone.utc)))

    assert reporter.calls[-1][0] == "end"
    assert reporter.calls[-1][1] == "source.RSS Feeds"
    assert reporter.calls[-1][2]["status"] == "failed"
    assert reporter.calls[-1][2]["message"] == "Fetch failed for RSS Feeds"
    assert reporter.calls[-1][2]["diagnostic_count"] == 1
    assert reporter.calls[-1][2]["diagnostics"][0]["source"] == "Feed"


def test_fetch_with_progress_records_source_warnings(tmp_path) -> None:
    class WarningScraper:
        fetch_diagnostics = [
            {"source": "Feed", "error": "HTTPStatusError", "message": "HTTP 500"},
        ]

        async def fetch(self, since):  # type: ignore[no-untyped-def]
            return []

    orchestrator = HorizonOrchestrator(
        _config(),
        StorageManager(data_dir=str(tmp_path / "data")),
    )
    reporter = CaptureReporter()
    orchestrator.verbose_reporter = reporter  # type: ignore[assignment]
    orchestrator.console = CaptureConsole()

    items = asyncio.run(orchestrator._fetch_with_progress("RSS Feeds", WarningScraper(), datetime.now(timezone.utc)))

    assert items == []
    assert reporter.calls[-1][0] == "end"
    assert reporter.calls[-1][1] == "source.RSS Feeds"
    assert reporter.calls[-1][2]["status"] == "warning"
    assert reporter.calls[-1][2]["message"] == "Fetched RSS Feeds with source warnings"
    assert reporter.calls[-1][2]["diagnostic_count"] == 1
