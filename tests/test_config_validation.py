from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import AIConfig, AIProvider, Config, FilteringConfig, SourcesConfig
from src.storage import manager as storage_manager
from src.storage.manager import StorageManager


def _base() -> dict:
    return Config(
        version="1",
        ai=AIConfig(provider=AIProvider.OPENAI, model="m", api_key_env="X"),
        sources=SourcesConfig(),
        filtering=FilteringConfig(),
    ).model_dump(mode="json")


def test_config_rejects_unknown_nested_keys():
    payload = _base()
    payload["filtering"]["ai_score_treshold"] = 7

    with pytest.raises(ValidationError):
        Config.model_validate(payload)


def test_config_rejects_invalid_thresholds():
    payload = _base()
    payload["filtering"]["ai_score_threshold"] = 99

    with pytest.raises(ValidationError):
        Config.model_validate(payload)


def test_storage_load_config_accepts_utf8_bom(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path = data_dir / "config.json"
    config_path.write_text(
        Config.model_validate(_base()).model_dump_json(),
        encoding="utf-8-sig",
    )

    loaded = StorageManager(data_dir=str(data_dir)).load_config()

    assert loaded.ai.languages == ["en"]


def test_storage_save_config_uses_timestamped_backup_when_fixed_backup_is_locked(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config_path = data_dir / "config.json"
    original = Config.model_validate(_base())
    config_path.write_text(original.model_dump_json(), encoding="utf-8")

    real_copy2 = storage_manager.shutil.copy2

    def copy2_with_locked_fixed_backup(src, dst, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(dst).endswith("config.json.bak"):
            raise PermissionError("fixed backup is locked")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(storage_manager.shutil, "copy2", copy2_with_locked_fixed_backup)
    updated = original.model_copy(deep=True)
    updated.notes["web_default_hours"] = "12"

    StorageManager(data_dir=str(data_dir)).save_config(updated, backup=True)

    assert not config_path.with_suffix(".json.bak").exists()
    fallback_backups = list(data_dir.glob("config.json.*.bak"))
    assert len(fallback_backups) == 1
    loaded = StorageManager(data_dir=str(data_dir)).load_config()
    assert loaded.notes["web_default_hours"] == "12"
