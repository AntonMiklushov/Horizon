from __future__ import annotations

import tomllib
import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_config_is_ignored_by_git_and_docker() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")

    for pattern in [
        ".env.local",
        ".env.*.local",
        "data/config.json",
        "data/*.bak",
        "data/mcp-runs/",
        "data/*secrets*.json",
    ]:
        assert pattern in gitignore
        assert pattern in dockerignore


def test_sdist_excludes_runtime_data() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excludes = set(pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"])

    assert "/data/config.json" in excludes
    assert "/data/*.bak" in excludes
    assert "/data/mcp-runs" in excludes
    assert "/data/*secrets*.json" in excludes
    assert "/.env" in excludes
    assert "/.env.local" in excludes
    assert "/.env.*.local" in excludes


def test_extension_templates_are_packaged() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    included = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert "src/horizon_ext/rendering/templates" in included
    assert "src/horizon_ext/web/templates" in included


def test_legacy_import_paths_remain_available() -> None:
    for module_name in [
        "src.pipeline",
        "src.rendering",
        "src.ai.personal_briefing",
        "src.web.app",
    ]:
        assert importlib.import_module(module_name)


def test_dockerfile_copies_only_safe_data_files() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY data ./data" not in dockerfile
    assert "data/config.example.json" in dockerfile
    assert "data/config.personal-news.example.json" in dockerfile
    assert "data/presets.json" in dockerfile


def test_daily_workflow_materializes_runtime_config_from_secret() -> None:
    workflow = (ROOT / ".github" / "workflows" / "daily-summary.yml").read_text(encoding="utf-8")

    assert "HORIZON_CONFIG_JSON: ${{ secrets.HORIZON_CONFIG_JSON }}" in workflow
    assert "printf '%s' \"$HORIZON_CONFIG_JSON\" > data/config.json" in workflow
    assert "python -m json.tool data/config.json > /dev/null" in workflow


def test_windows_web_launcher_opens_local_dashboard_without_auto_install() -> None:
    launcher = (ROOT / "start-horizon-web.cmd").read_text(encoding="utf-8")
    normalized = launcher.lower()

    assert "127.0.0.1" in launcher
    assert "8787" in launcher
    assert "%LOCALAPPDATA%\\HorizonBrief" in launcher
    assert "venv\\Scripts\\python.exe" in launcher
    assert "-m src.horizon_ext.web.main" in launcher
    assert 'start "" "%URL%"' in launcher
    assert "invoke-webrequest" in normalized
    assert "ready_timeout_seconds" in normalized

    for forbidden in ["uv sync", "pip install", "python -m pip", "pip3 install"]:
        assert forbidden not in normalized
