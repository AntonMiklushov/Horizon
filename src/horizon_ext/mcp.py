"""MCP helpers for fork-specific local run behavior."""

from __future__ import annotations

from typing import Any


def normalize_run_instructions(run_instructions: str | None) -> str:
    """Normalize one-off local run instructions before storing or prompting."""

    return (run_instructions or "").strip()[:2000]


def local_only_config(config: Any) -> Any:
    """Return a deep config clone with external delivery disabled."""

    clone = config.model_copy(deep=True)
    if getattr(clone, "email", None):
        clone.email.enabled = False
    if getattr(clone, "webhook", None):
        clone.webhook.enabled = False
    if getattr(clone, "publishing", None):
        clone.publishing.enabled = False
    return clone


def redact_runtime_payload(value: Any) -> Any:
    """Redact secret-bearing keys from payloads exposed through local tools."""

    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_l = str(key).lower()
            if any(token in key_l for token in ("password", "secret", "token", "api_key", "headers")):
                redacted[key] = "<redacted>"
            elif key_l in {"email_address", "request_body"}:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact_runtime_payload(item)
        return redacted
    if isinstance(value, list):
        return [redact_runtime_payload(item) for item in value]
    return value

