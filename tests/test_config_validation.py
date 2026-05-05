from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import AIConfig, AIProvider, Config, FilteringConfig, SourcesConfig


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
