"""Resolve OpenClaw provider settings into Horizon's native AIConfig."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import dotenv_values

from ..models import AIConfig, AIProvider


_OPENCLAW_SENTINEL_MODELS = {"", "openclaw", "default", "openclaw/default"}
_ENV_KEY_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

_API_KEY_ENV_BY_PROVIDER = {
    AIProvider.OPENAI: "OPENAI_API_KEY",
    AIProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    AIProvider.GEMINI: "GOOGLE_API_KEY",
    AIProvider.ALI: "DASHSCOPE_API_KEY",
    AIProvider.DOUBAO: "DOUBAO_API_KEY",
    AIProvider.MINIMAX: "MINIMAX_API_KEY",
    AIProvider.AZURE: "AZURE_OPENAI_API_KEY",
}

_OPENAI_COMPATIBLE_DEFAULT_BASE_URLS = {
    AIProvider.ALI: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    AIProvider.DOUBAO: "https://ark.cn-beijing.volces.com/api/v3",
    AIProvider.MINIMAX: "https://api.minimax.io/v1",
}

_PROVIDER_ALIASES = {
    "openai": AIProvider.OPENAI,
    "openai_compatible": AIProvider.OPENAI,
    "lm_studio": AIProvider.OPENAI,
    "anthropic": AIProvider.ANTHROPIC,
    "claude": AIProvider.ANTHROPIC,
    "google": AIProvider.GEMINI,
    "gemini": AIProvider.GEMINI,
    "azure": AIProvider.AZURE,
    "azure_openai": AIProvider.AZURE,
    "ali": AIProvider.ALI,
    "aliyun": AIProvider.ALI,
    "dashscope": AIProvider.ALI,
    "qwen": AIProvider.ALI,
    "doubao": AIProvider.DOUBAO,
    "minimax": AIProvider.MINIMAX,
}

_API_PROVIDER_BY_OPENCLAW_ADAPTER = {
    "openai-completions": AIProvider.OPENAI,
    "openai-responses": AIProvider.OPENAI,
    "anthropic-messages": AIProvider.ANTHROPIC,
    "google-generative-ai": AIProvider.GEMINI,
}


class OpenClawConfigSnapshot:
    def __init__(self, text: str = ""):
        self.text = text

    @classmethod
    def from_path(cls, path: Path | None) -> "OpenClawConfigSnapshot":
        if path is None:
            return cls()
        text = path.read_text(encoding="utf-8-sig")
        snapshot = cls(text)
        snapshot.load_env_block()
        return snapshot

    def load_env_block(self) -> None:
        env_block = self._object_body_after_key("env")
        if not env_block:
            return
        for key, value in _string_pairs(env_block):
            if _ENV_KEY_RE.fullmatch(key) and value:
                os.environ.setdefault(key, value)

    def primary_model(self, agent_id: str | None = None) -> str | None:
        if agent_id:
            agent_body = self._agent_body(agent_id)
            if agent_body:
                model = _model_from_body(agent_body)
                if model:
                    return model
        defaults_body = self._object_body_for_path(("agents", "defaults"))
        if defaults_body:
            model = _model_from_body(defaults_body)
            if model:
                return model
        return self._first_model_string()

    def provider_body(self, provider_id: str) -> str | None:
        providers_body = self._object_body_for_path(("models", "providers"))
        if not providers_body:
            return None
        return _object_body_after_key_in_text(providers_body, provider_id)

    def _agent_body(self, agent_id: str) -> str | None:
        list_body = self._array_body_for_path(("agents", "list"))
        if not list_body:
            return None
        for body in _top_level_object_bodies(list_body):
            if _string_value(body, "id") == agent_id:
                return body
        return None

    def _object_body_for_path(self, keys: tuple[str, ...]) -> str | None:
        body = self.text
        for key in keys:
            body = _object_body_after_key_in_text(body, key)
            if body is None:
                return None
        return body

    def _array_body_for_path(self, keys: tuple[str, ...]) -> str | None:
        body = self.text
        for key in keys[:-1]:
            body = _object_body_after_key_in_text(body, key)
            if body is None:
                return None
        return _array_body_after_key_in_text(body, keys[-1])

    def _object_body_after_key(self, key: str) -> str | None:
        return _object_body_after_key_in_text(self.text, key)

    def _first_model_string(self) -> str | None:
        match = re.search(r"\bmodel\s*:\s*['\"]([^'\"]+)['\"]", self.text)
        return match.group(1).strip() if match else None


def resolve_openclaw_ai_config(config: AIConfig) -> AIConfig:
    """Return a concrete Horizon AIConfig for OpenClaw shared-provider mode.

    Horizon still runs its own pipeline. OpenClaw is used only as the source of
    provider/model/env settings; no OpenClaw agent session or prompt context is
    involved in Horizon's scoring, enrichment, or summary generation.
    """

    _load_openclaw_env(config.openclaw_env_path)
    snapshot = OpenClawConfigSnapshot.from_path(_resolve_openclaw_config_path(config.openclaw_config_path))
    model_spec = _resolve_model_spec(config, snapshot)
    provider_name, model_name = _split_model_spec(model_spec)
    provider, api_key_env, base_url = _provider_settings(provider_name, config, snapshot)

    return config.model_copy(
        update={
            "provider": provider,
            "model": model_name,
            "api_key_env": api_key_env,
            "base_url": base_url,
        }
    )


def _resolve_model_spec(config: AIConfig, snapshot: OpenClawConfigSnapshot) -> str:
    model = (config.model or "").strip()
    if model.lower() not in _OPENCLAW_SENTINEL_MODELS:
        return model

    env_model = _first_env("HORIZON_OPENCLAW_MODEL", "OPENCLAW_MODEL")
    if env_model:
        return env_model

    model_from_config = snapshot.primary_model(
        agent_id=config.openclaw_agent_id
        or _first_env("HORIZON_OPENCLAW_AGENT_ID", "OPENCLAW_AGENT_ID")
    )
    if model_from_config:
        return model_from_config

    raise ValueError(
        "OpenClaw provider mode needs a model spec such as 'openai/gpt-5.5', "
        "HORIZON_OPENCLAW_MODEL/OPENCLAW_MODEL, or an OpenClaw config with agents.defaults.model.primary."
    )


def _load_openclaw_env(explicit_path: str | None) -> None:
    for path in _candidate_env_paths(explicit_path):
        if not path.exists():
            continue
        values = dotenv_values(path)
        for key, value in values.items():
            if not key or not value or not _ENV_KEY_RE.fullmatch(key):
                continue
            os.environ.setdefault(key, value)
        return


def _candidate_env_paths(explicit_path: str | None) -> list[Path]:
    candidates: list[Path] = []
    for value in (
        explicit_path,
        os.getenv("HORIZON_OPENCLAW_ENV_PATH"),
        os.getenv("OPENCLAW_ENV_PATH"),
        "~/.openclaw/openclaw.env",
        "/etc/openclaw/openclaw.env",
    ):
        if value:
            candidates.append(Path(value).expanduser())
    return candidates


def _resolve_openclaw_config_path(explicit_path: str | None) -> Path | None:
    for value in (
        explicit_path,
        os.getenv("HORIZON_OPENCLAW_CONFIG_PATH"),
        os.getenv("OPENCLAW_CONFIG_PATH"),
        "~/.openclaw/openclaw.json",
        "/etc/openclaw/openclaw.json5",
    ):
        if not value:
            continue
        path = Path(value).expanduser()
        if path.exists():
            return path
    return None


def _split_model_spec(model_spec: str) -> tuple[str | None, str]:
    normalized = model_spec.strip()
    if "/" not in normalized:
        return None, normalized
    provider, model = normalized.split("/", 1)
    provider = provider.strip().lower().replace("-", "_")
    model = model.strip()
    if not model:
        raise ValueError(f"OpenClaw model spec has an empty model name: {model_spec!r}")
    return provider, model


def _provider_settings(
    provider_name: str | None,
    config: AIConfig,
    snapshot: OpenClawConfigSnapshot,
) -> tuple[AIProvider, str | None, str | None]:
    provider_alias = (provider_name or "openai").lower().replace("-", "_")

    if provider_alias == "ollama":
        return (
            AIProvider.OPENAI,
            _api_key_env_from_config(config, None, None),
            config.base_url
            or _first_env(
                "HORIZON_OPENCLAW_BASE_URL",
                "OPENCLAW_LLM_BASE_URL",
                "OLLAMA_OPENAI_BASE_URL",
                "OPENAI_BASE_URL",
            )
            or _ollama_openai_base_url(),
        )

    provider_body = snapshot.provider_body(provider_alias)
    provider = _provider_from_openclaw_provider(provider_alias, provider_body)
    api_key_env = _api_key_env_from_config(config, provider, provider_body)
    base_url = (
        config.base_url
        or _base_url_from_provider(provider_body)
        or _first_env("HORIZON_OPENCLAW_BASE_URL", "OPENCLAW_LLM_BASE_URL", _base_url_env(provider))
        or _OPENAI_COMPATIBLE_DEFAULT_BASE_URLS.get(provider)
    )
    return provider, api_key_env, base_url


def _provider_from_openclaw_provider(alias: str, provider_body: str | None) -> AIProvider:
    if provider_body:
        api = (_string_value(provider_body, "api") or "").strip().lower()
        if api:
            try:
                return _API_PROVIDER_BY_OPENCLAW_ADAPTER[api]
            except KeyError as exc:
                raise ValueError(f"Unsupported OpenClaw provider adapter for {alias!r}: {api!r}") from exc
        if _string_value(provider_body, "baseUrl"):
            return AIProvider.OPENAI
    try:
        return _PROVIDER_ALIASES[alias]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported OpenClaw provider prefix {alias!r}. Add models.providers.{alias}.api/baseUrl "
            "or use a Horizon-supported provider prefix."
        ) from exc


def _api_key_env_from_config(config: AIConfig, provider: AIProvider | None, provider_body: str | None) -> str | None:
    if config.api_key_env:
        return config.api_key_env
    marker = _string_value(provider_body or "", "apiKey")
    env_key = _env_key_from_secret_marker(marker)
    if env_key:
        return env_key
    if marker and marker.strip():
        generated_key = "HORIZON_OPENCLAW_" + re.sub(r"[^A-Z0-9_]", "_", (provider.value if provider else "provider").upper()) + "_API_KEY"
        os.environ.setdefault(generated_key, marker.strip())
        return generated_key
    if provider:
        return _API_KEY_ENV_BY_PROVIDER.get(provider)
    return None


def _env_key_from_secret_marker(value: str | None) -> str | None:
    if not value:
        return None
    marker = value.strip()
    braced = re.fullmatch(r"\$\{([A-Z_][A-Z0-9_]*)\}", marker)
    if braced:
        return braced.group(1)
    if _ENV_KEY_RE.fullmatch(marker):
        return marker
    return None


def _base_url_from_provider(provider_body: str | None) -> str | None:
    return _string_value(provider_body or "", "baseUrl")


def _base_url_env(provider: AIProvider) -> str:
    if provider == AIProvider.ANTHROPIC:
        return "ANTHROPIC_BASE_URL"
    if provider == AIProvider.AZURE:
        return "AZURE_OPENAI_ENDPOINT"
    return "OPENAI_BASE_URL"


def _ollama_openai_base_url() -> str:
    host = _first_env("OLLAMA_HOST")
    if not host:
        return "http://127.0.0.1:11434/v1"
    host = host.rstrip("/")
    if host.endswith("/v1"):
        return host
    return f"{host}/v1"


def _first_env(*keys: str) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value and value.strip():
            return value.strip()
    return None


def _model_from_body(body: str) -> str | None:
    model_object = _object_body_after_key_in_text(body, "model")
    if model_object:
        primary = _string_value(model_object, "primary")
        if primary:
            return primary
    return _string_value(body, "model") or _string_value(body, "primary")


def _string_value(text: str, key: str) -> str | None:
    key_pattern = _key_pattern(key)
    match = re.search(rf"{key_pattern}\s*:\s*['\"]([^'\"]+)['\"]", text)
    return match.group(1).strip() if match else None


def _string_pairs(text: str) -> list[tuple[str, str]]:
    return [
        (match.group(1) or match.group(2), match.group(3))
        for match in re.finditer(r"(?:['\"]([^'\"]+)['\"]|([A-Za-z_][\w-]*))\s*:\s*['\"]([^'\"]*)['\"]", text)
    ]


def _object_body_after_key_in_text(text: str, key: str) -> str | None:
    match = re.search(rf"{_key_pattern(key)}\s*:\s*\{{", text)
    if not match:
        return None
    return _balanced_body(text, match.end() - 1, "{", "}")


def _array_body_after_key_in_text(text: str, key: str) -> str | None:
    match = re.search(rf"{_key_pattern(key)}\s*:\s*\[", text)
    if not match:
        return None
    return _balanced_body(text, match.end() - 1, "[", "]")


def _top_level_object_bodies(text: str) -> list[str]:
    bodies: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        body, end = _balanced_body_with_end(text, index, "{", "}")
        bodies.append(body)
        index = end + 1
    return bodies


def _balanced_body(text: str, start: int, open_char: str, close_char: str) -> str | None:
    result = _balanced_body_with_end(text, start, open_char, close_char)
    return result[0] if result else None


def _balanced_body_with_end(text: str, start: int, open_char: str, close_char: str) -> tuple[str, int] | None:
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == open_char:
            depth += 1
            continue
        if char == close_char:
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index
    return None


def _key_pattern(key: str) -> str:
    escaped = re.escape(key)
    if re.fullmatch(r"[A-Za-z_][\w-]*", key):
        return rf"(?:\b{escaped}\b|['\"]{escaped}['\"])"
    return rf"['\"]{escaped}['\"]"