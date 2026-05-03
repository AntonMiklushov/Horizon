"""Console helpers for CLI entrypoints."""

from __future__ import annotations

import sys
from typing import Any

from rich.console import Console


def _stream_encoding() -> str:
    return getattr(sys.stdout, "encoding", None) or "utf-8"


def _supports_status_emoji(encoding: str) -> bool:
    try:
        "🌅✅❌⚠️".encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def _safe_text(text: str, encoding: str) -> str:
    try:
        text.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    return text


class HorizonConsole(Console):
    """Rich console that degrades literal emoji on legacy Windows encodings."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._horizon_encoding = _stream_encoding()
        self._horizon_unicode_safe = _supports_status_emoji(self._horizon_encoding)
        kwargs.setdefault("emoji", self._horizon_unicode_safe)
        super().__init__(*args, **kwargs)

    def print(self, *objects: Any, **kwargs: Any) -> None:
        if not self._horizon_unicode_safe:
            objects = tuple(
                _safe_text(obj, self._horizon_encoding) if isinstance(obj, str) else obj
                for obj in objects
            )
            kwargs.setdefault("emoji", False)
        super().print(*objects, **kwargs)


def make_console(*args: Any, **kwargs: Any) -> HorizonConsole:
    return HorizonConsole(*args, **kwargs)
