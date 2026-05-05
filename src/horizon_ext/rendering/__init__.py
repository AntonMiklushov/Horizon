"""Digest rendering layer for Markdown, HTML, Jekyll, email, and webhooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..pipeline import DigestDocument


class DigestRenderer:
    """Render a DigestDocument into channel-specific artifacts."""

    def __init__(self, template_dir: str | Path | None = None, inline_email_css: bool = True):
        default_templates = Path(__file__).parent / "templates"
        search_paths = []
        if template_dir:
            search_paths.append(str(Path(template_dir).expanduser().resolve()))
        search_paths.append(str(default_templates))
        self.env = Environment(
            loader=FileSystemLoader(search_paths),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.inline_email_css = inline_email_css

    @classmethod
    def from_config(cls, config: Any, storage: Any | None = None) -> "DigestRenderer":
        rendering = getattr(config, "rendering", None)
        template_dir = getattr(rendering, "template_dir", None)
        if template_dir and storage is not None:
            template_dir = storage.resolve_runtime_path(template_dir)
        return cls(
            template_dir=template_dir,
            inline_email_css=bool(getattr(rendering, "inline_email_css", True)),
        )

    def render_markdown(self, digest: DigestDocument) -> str:
        return digest.markdown

    def render_html(self, digest: DigestDocument) -> str:
        return self.env.get_template("digest.html").render(digest=digest)

    def render_jekyll_post(self, digest: DigestDocument) -> str:
        markdown = digest.markdown
        first_line = markdown.strip().split("\n")[0] if markdown.strip() else ""
        if first_line.startswith("# "):
            parts = markdown.split("\n", 1)
            markdown = parts[1].strip() if len(parts) > 1 else ""
        return self.env.get_template("jekyll_post.md").render(digest=digest, markdown=markdown)

    def render_email_html(self, digest: DigestDocument) -> str:
        html = self.env.get_template("email.html").render(digest=digest)
        if not self.inline_email_css:
            return html
        try:
            from premailer import transform
        except Exception:
            return html
        return transform(html, allow_network=False)

    def render_plain_text(self, digest: DigestDocument) -> str:
        return self.env.get_template("email.txt").render(digest=digest)

    def render_webhook_markdown(self, digest: DigestDocument) -> str:
        return _format_markdown_for_webhook(digest.markdown)


def _format_markdown_for_webhook(value: str) -> str:
    """Flatten HTML constructs that chat/webhook Markdown often cannot render."""

    import html
    import re

    details_re = re.compile(
        r"<details>\s*<summary>(.*?)</summary>\s*(.*?)\s*</details>",
        re.IGNORECASE | re.DOTALL,
    )
    li_link_re = re.compile(
        r"<li>\s*<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>\s*</li>",
        re.IGNORECASE | re.DOTALL,
    )
    li_re = re.compile(r"<li>\s*(.*?)\s*</li>", re.IGNORECASE | re.DOTALL)
    anchor_id_re = re.compile(r"<a\s+[^>]*id=[\"'][^\"']+[\"'][^>]*>\s*</a>", re.IGNORECASE)
    html_tag_re = re.compile(r"<[^>]+>")

    def strip_tags(text: str) -> str:
        return html.unescape(html_tag_re.sub("", text)).strip()

    def replace_details(match: re.Match) -> str:
        title = strip_tags(match.group(1)) or "References"
        body = match.group(2)
        items: list[str] = []
        for href, label in li_link_re.findall(body):
            clean_label = strip_tags(label)
            clean_href = html.unescape(href).strip()
            if clean_label and clean_href:
                items.append(f"- [{clean_label}]({clean_href})")
        if not items:
            for item in li_re.findall(body):
                clean_item = strip_tags(item)
                if clean_item:
                    items.append(f"- {clean_item}")
        return f"**{title}**" + ("\n\n" + "\n".join(items) if items else "")

    value = anchor_id_re.sub("", value)
    return details_re.sub(replace_details, value)
