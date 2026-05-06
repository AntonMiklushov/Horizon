from __future__ import annotations

from datetime import datetime, timezone

from src.models import ContentItem, SourceType
from src.horizon_ext.pipeline import build_digest_document
from src.horizon_ext.rendering import DigestRenderer


def _digest():
    item = ContentItem(
        id="rss:test",
        source_type=SourceType.RSS,
        title="<script>alert(1)</script>",
        url="https://example.com/a?x=1",
        published_at=datetime(2026, 5, 3, tzinfo=timezone.utc),
        metadata={
            "source_name": "Reuters",
            "source_role": "fact_layer",
            "source_reliability_tier": "tier1",
            "summary": "Safe <b>summary</b>",
            "confidence": "medium",
            "evidence_strength": "high",
            "claim_type": "confirmed_fact",
            "why_it_matters": "Important for readers",
            "sources": [{"url": "https://example.com/ref", "title": "Ref"}],
        },
        ai_score=9.0,
        ai_tags=["world"],
    )
    return build_digest_document(
        date="2026-05-03",
        language="ru",
        title="Test Digest",
        items=[item],
        total_fetched=3,
        markdown="# Test",
    )


def test_html_renderer_escapes_hostile_content():
    html = DigestRenderer().render_html(_digest())

    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Safe &lt;b&gt;summary&lt;/b&gt;" in html
    assert "confidence: medium" in html


def test_email_renderer_outputs_html_and_plain_text():
    renderer = DigestRenderer(inline_email_css=False)
    digest = _digest()

    html = renderer.render_email_html(digest)
    text = renderer.render_plain_text(digest)

    assert "<html" in html
    assert "Test Digest" in text
    assert "https://example.com/a?x=1" in text
