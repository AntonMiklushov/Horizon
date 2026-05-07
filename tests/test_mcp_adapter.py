from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.mcp.errors import HorizonMcpError
from src.mcp.horizon_adapter import (
    _load_mcp_secrets,
    apply_source_filter,
    load_config,
    load_runtime,
    resolve_config_path,
    resolve_horizon_path,
)


def test_resolve_horizon_path_accepts_explicit_repo() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert resolve_horizon_path(str(repo_root)) == repo_root.resolve()


def test_resolve_config_path_defaults_to_repo_data_config(tmp_path: Path) -> None:
    repo_root = tmp_path
    config_path = repo_root / "data" / "config.json"
    config_path.parent.mkdir()
    config_path.write_text("{}", encoding="utf-8")

    assert resolve_config_path(repo_root) == config_path.resolve()


def test_resolve_horizon_path_rejects_other_repo_by_default(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fake'\n", encoding="utf-8")

    with pytest.raises(HorizonMcpError, match="disabled by default"):
        resolve_horizon_path(str(tmp_path))


def test_source_filter_supports_twitter() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runtime = load_runtime(repo_root)
    config = load_config(runtime, repo_root / "data" / "config.example.json")

    filtered, selected, unknown = apply_source_filter(config, ["twitter"])

    assert selected == ["twitter"]
    assert unknown == []
    assert filtered.sources.twitter is not None
    assert filtered.sources.github == []


def test_mcp_load_config_accepts_utf8_bom(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runtime = load_runtime(repo_root)
    config_path = tmp_path / "config.json"
    payload = (repo_root / "data" / "config.example.json").read_text(encoding="utf-8")
    config_path.write_text(payload, encoding="utf-8-sig")

    config = load_config(runtime, config_path)

    assert config.version == "1.0"


def test_load_mcp_secrets_loads_generic_env_keys(tmp_path: Path, monkeypatch) -> None:
    secrets_path = tmp_path / "mcp.secrets.json"
    secrets_path.write_text(
        json.dumps(
            {
                "env": {
                    "ANTHROPIC_API_KEY": "sk-ant-test",
                    "CUSTOM_TOKEN": "token-123",
                    "lowercase": "ignored",
                }
            }
        ),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("HORIZON_MCP_SECRETS_PATH", str(secrets_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CUSTOM_TOKEN", raising=False)

    _load_mcp_secrets(repo_root, override=False)

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test"
    assert os.environ["CUSTOM_TOKEN"] == "token-123"
    assert "lowercase" not in os.environ
