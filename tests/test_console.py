from src.console import _safe_text, _supports_status_emoji


def test_status_emoji_detection_rejects_legacy_encoding() -> None:
    assert _supports_status_emoji("cp1252") is False


def test_safe_text_replaces_unencodable_status_emoji() -> None:
    assert _safe_text("✅ done", "cp1252") == "? done"
