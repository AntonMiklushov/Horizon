"""Tests for the Codex CLI AI provider."""

from __future__ import annotations

import subprocess
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.ai.client import CodexCliClient, OpenAIClient, _LocalTempDir, create_ai_client
from src.models import AIConfig, AIProvider


def _make_config(**overrides) -> AIConfig:
    defaults = {
        "provider": AIProvider.CODEX_CLI,
        "model": "codex-cli",
        "languages": ["ru"],
        "codex_command": "codex",
        "codex_timeout_sec": 180,
        "codex_extra_args": [],
        "codex_use_output_last_message": True,
        "codex_use_json": False,
    }
    defaults.update(overrides)
    return AIConfig(**defaults)


class FakeLocalTempDir:
    def __init__(self, *args, **kwargs):
        self.name = "fake-temp-dir"

    def cleanup(self):
        return None


def test_codex_cli_provider_does_not_require_api_key_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = create_ai_client(_make_config())

    assert isinstance(client, CodexCliClient)


def test_command_uses_codex_exec_and_passes_prompt(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="completion\n", stderr="")

    monkeypatch.setattr("src.ai.client.subprocess.run", fake_run)
    monkeypatch.setattr("src.ai.client._LocalTempDir", FakeLocalTempDir)
    client = CodexCliClient(_make_config(codex_use_output_last_message=False))
    client._supported_flags = {"--color", "--sandbox", "--skip-git-repo-check", "-c"}

    result = client._run_codex_exec(
        client._build_prompt(system="system prompt", user="user prompt")
    )

    assert result == "completion"
    args, kwargs = calls[0]
    assert args[1] == "exec"
    command_name = args[0].lower()
    assert command_name.endswith("codex") or command_name.endswith("codex.exe")
    assert args.count("--skip-git-repo-check") == 1
    assert args[args.index("-c") + 1] == 'model_reasoning_effort="medium"'
    assert args[-1] == "-"
    assert kwargs["input"].startswith("SYSTEM:\nsystem prompt")
    assert "\n\nUSER:\nuser prompt" in kwargs["input"]
    assert "Do not edit files." in kwargs["input"]
    assert kwargs["timeout"] == 180
    assert kwargs["check"] is False


def test_extra_args_do_not_duplicate_managed_codex_flags(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="completion\n", stderr="")

    monkeypatch.setattr("src.ai.client.subprocess.run", fake_run)
    client = CodexCliClient(
        _make_config(
            codex_use_output_last_message=False,
            codex_extra_args=[
                "--skip-git-repo-check",
                "-c",
                'model_reasoning_effort="high"',
            ],
        )
    )
    client._supported_flags = {"--skip-git-repo-check", "-c"}

    result = client._run_codex_exec(client._build_prompt(system="s", user="u"))

    assert result == "completion"
    args, _ = calls[0]
    assert args.count("--skip-git-repo-check") == 1
    assert args.count("-c") == 1
    assert 'model_reasoning_effort="high"' in args
    assert 'model_reasoning_effort="medium"' not in args


def test_stdout_returned_when_no_artifact_exists(monkeypatch):
    def fake_run(args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="stdout answer\n", stderr="")

    monkeypatch.setattr("src.ai.client.subprocess.run", fake_run)
    client = CodexCliClient(_make_config(codex_use_output_last_message=False))
    client._supported_flags = set()

    result = client._run_codex_exec(client._build_prompt(system="s", user="u"))

    assert result == "stdout answer"


def test_output_last_message_artifact_wins_over_stdout(monkeypatch):
    def fake_run(args, **kwargs):
        assert args[args.index("--output-last-message") + 1].endswith("last-message.txt")
        return SimpleNamespace(returncode=0, stdout="stdout answer\n", stderr="")

    monkeypatch.setattr("src.ai.client.subprocess.run", fake_run)
    monkeypatch.setattr("src.ai.client._LocalTempDir", FakeLocalTempDir)
    monkeypatch.setattr(
        CodexCliClient,
        "_read_artifact",
        staticmethod(lambda output_path: "artifact answer"),
    )
    client = CodexCliClient(_make_config())
    client._supported_flags = {"--output-last-message"}

    result = client._run_codex_exec(client._build_prompt(system="s", user="u"))

    assert result == "artifact answer"


def test_codex_temp_dir_is_not_created_in_workspace(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    temp_dir = _LocalTempDir(prefix="horizon-test-")

    try:
        assert Path(temp_dir.name).name.startswith("horizon-test-")
        assert not Path(temp_dir.name).resolve().is_relative_to(tmp_path.resolve())
    finally:
        temp_dir.cleanup()


def test_codex_temp_dir_cleanup_ignores_os_errors(monkeypatch):
    class RaisingTempDir:
        name = "fake-temp-dir"

        def cleanup(self):
            raise PermissionError("cleanup denied")

    monkeypatch.setattr(
        "src.ai.client.tempfile.TemporaryDirectory",
        lambda **kwargs: RaisingTempDir(),
    )

    _LocalTempDir(prefix="horizon-test-").cleanup()


def test_non_zero_exit_raises_clear_error(monkeypatch):
    def fake_run(args, **kwargs):
        return SimpleNamespace(returncode=2, stdout="partial stdout", stderr="bad stderr")

    monkeypatch.setattr("src.ai.client.subprocess.run", fake_run)
    client = CodexCliClient(_make_config(codex_use_output_last_message=False))
    client._supported_flags = set()

    with pytest.raises(RuntimeError, match="Codex CLI failed with exit code 2"):
        client._run_codex_exec(client._build_prompt(system="s", user="u"))


def test_timeout_raises_clear_error(monkeypatch):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr("src.ai.client.subprocess.run", fake_run)
    client = CodexCliClient(
        _make_config(codex_timeout_sec=30, codex_use_output_last_message=False)
    )
    client._supported_flags = set()

    with pytest.raises(TimeoutError, match="Codex CLI timed out after 30 seconds"):
        client._run_codex_exec(client._build_prompt(system="s", user="u"))


def test_existing_openai_provider_still_uses_openai_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = create_ai_client(
        AIConfig(
            provider=AIProvider.OPENAI,
            model="gpt-4",
            api_key_env="OPENAI_API_KEY",
        )
    )

    assert isinstance(client, OpenAIClient)


def test_openai_compatible_local_provider_does_not_require_api_key_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = create_ai_client(
        AIConfig(
            provider=AIProvider.OPENAI,
            model="local-model",
            base_url="http://127.0.0.1:1234/v1",
            api_key_env=None,
        )
    )

    assert isinstance(client, OpenAIClient)
    assert client.model == "local-model"


def test_openai_compatible_local_provider_uses_text_response_format(monkeypatch):
    class FakeCompletions:
        async def create(self, **kwargs):  # type: ignore[no-untyped-def]
            calls.append(kwargs)
            return SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
            )

    calls = []
    monkeypatch.setattr(
        "src.ai.client.AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )
    client = OpenAIClient(
        AIConfig(
            provider=AIProvider.OPENAI,
            model="local-model",
            base_url="http://127.0.0.1:1234/v1",
            api_key_env=None,
        )
    )

    result = asyncio.run(client.complete("system", "user", max_tokens=32))

    assert result == '{"ok": true}'
    assert calls[0]["response_format"] == {"type": "text"}
