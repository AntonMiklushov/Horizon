"""Webhook notification service for Horizon Brief."""

import html
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, List, Optional, Union
from urllib.parse import urlsplit
import httpx

from ..models import ContentItem, WebhookConfig
from ..ai.summarizer import DailySummarizer

logger = logging.getLogger(__name__)


# Pattern: #{key} or #{key?param1=val1&param2=val2}
_PLACEHOLDER_RE = re.compile(r"#\{(\w+)(\?\w+=[^}]+)?\}")
_DETAILS_RE = re.compile(
    r"<details>\s*<summary>(.*?)</summary>\s*(.*?)\s*</details>",
    re.IGNORECASE | re.DOTALL,
)
_LI_LINK_RE = re.compile(
    r"<li>\s*<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>\s*</li>",
    re.IGNORECASE | re.DOTALL,
)
_LI_RE = re.compile(r"<li>\s*(.*?)\s*</li>", re.IGNORECASE | re.DOTALL)
_ANCHOR_ID_RE = re.compile(r"<a\s+[^>]*id=[\"'][^\"']+[\"'][^>]*>\s*</a>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SECRET_TEXT_RE = re.compile(
    r"(?i)(token|secret|password|key|authorization)([\"'\s:=]+)([^\"'\s,}]+)"
)


def _truncate(value: str, limit: int, split: str) -> str:
    """Truncate a string to at most *limit* characters by splitting on *split*.

    Segments are accumulated in order until adding the next one would
    exceed *limit* characters.  Remaining segments are dropped.

    Args:
        value: The full text to truncate
        limit: Maximum number of characters allowed
        split: Delimiter to split value into segments

    Returns:
        Truncated text
    """
    segments = value.split(split)
    kept: list[str] = []
    current_chars = 0

    for seg in segments:
        # +len(split) for the delimiter that will be re-joined
        seg_chars = len(seg) + (len(split) if kept else 0)
        if kept and current_chars + seg_chars > limit:
            break
        kept.append(seg)
        current_chars += seg_chars

    return split.join(kept)


def _redact_text(value: str) -> str:
    value = re.sub(r"([?&](?:token|key|secret|password)=)[^&#\s]+", r"\1<redacted>", value, flags=re.IGNORECASE)
    return _SECRET_TEXT_RE.sub(r"\1\2<redacted>", value)


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted = {}
    for key, value in headers.items():
        key_l = key.lower()
        if any(token in key_l for token in ("authorization", "token", "secret", "password", "api-key", "api_key")):
            redacted[key] = "<redacted>"
        else:
            redacted[key] = _redact_text(str(value))
    return redacted


def _redact_url(value: str) -> str:
    try:
        parts = urlsplit(value)
    except ValueError:
        return "<redacted-url>"
    if not parts.scheme or not parts.netloc:
        return "<redacted-url>"
    return f"{parts.scheme}://{parts.netloc}/<redacted>"


def _render(template: Union[str, dict, list], variables: dict) -> Union[str, dict, list]:
    """Replace #{key} and #{key?params} placeholders in a template.

    Supports strings, dicts, and lists.  For dicts/lists, walks all
    string values recursively and replaces placeholders.

    Parameterized syntax: #{key?limit=N&split=DELIM}
      - limit: maximum number of output characters
      - split: delimiter to split the value into segments before
               accumulating up to *limit* characters

    Args:
        template: Template with #{key} placeholders — str, dict, or list
        variables: Dict mapping placeholder keys to replacement values

    Returns:
        Same type as template, with placeholders replaced
    """
    if isinstance(template, dict):
        return {k: _render(v, variables) for k, v in template.items()}
    if isinstance(template, list):
        return [_render(item, variables) for item in template]
    if isinstance(template, str):
        def _replace(match: re.Match) -> str:
            key = match.group(1)
            params_str = match.group(2)  # e.g. "?limit=500&split=---"

            value = variables.get(key)
            if value is None:
                return match.group(0)  # leave placeholder unchanged

            if not params_str:
                return str(value)

            # Parse params: ?limit=500&split=---
            raw_params = params_str.lstrip("?")
            params: dict[str, str] = {}
            for pair in raw_params.split("&"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    params[k] = v

            limit = int(params.get("limit", "0")) if "limit" in params else 0
            split_delim = params.get("split", "---")

            if limit and split_delim:
                return _truncate(str(value), limit, split_delim)

            return str(value)

        return _PLACEHOLDER_RE.sub(_replace, template)
    # int, float, bool, None — return as-is
    return template


def _strip_html_tags(value: str) -> str:
    """Remove simple HTML tags and decode HTML entities."""
    return html.unescape(_HTML_TAG_RE.sub("", value)).strip()


def _convert_details_to_markdown(value: str) -> str:
    """Convert HTML details blocks into plain Markdown sections.

    Feishu card Markdown does not render HTML disclosure widgets, so references
    are flattened to a heading plus Markdown links before webhook delivery.
    """
    def _replace(match: re.Match) -> str:
        title = _strip_html_tags(match.group(1)) or "References"
        body = match.group(2)
        items: list[str] = []

        for href, label in _LI_LINK_RE.findall(body):
            clean_label = _strip_html_tags(label)
            clean_href = html.unescape(href).strip()
            if clean_label and clean_href:
                items.append(f"- [{clean_label}]({clean_href})")

        if not items:
            for item in _LI_RE.findall(body):
                clean_item = _strip_html_tags(item)
                if clean_item:
                    items.append(f"- {clean_item}")

        if not items:
            fallback = _strip_html_tags(body)
            return f"**{title}**\n\n{fallback}" if fallback else f"**{title}**"

        return f"**{title}**\n\n" + "\n".join(items)

    return _DETAILS_RE.sub(_replace, value)


def _format_markdown_for_webhook(value: str) -> str:
    """Flatten HTML constructs that chat/webhook Markdown often cannot render."""
    value = _ANCHOR_ID_RE.sub("", value)
    return _convert_details_to_markdown(value)


def _prepare_variables_for_body(raw_body: Union[str, dict, list, None], variables: dict) -> dict:
    """Apply webhook-safe variable formatting before body rendering."""
    if raw_body is None or "summary" not in variables:
        return variables

    prepared = dict(variables)
    prepared["summary"] = _format_markdown_for_webhook(str(variables["summary"]))
    return prepared


def _isjson(s: str) -> bool:
    """Return True if the string starts with a JSON open brace."""
    s = s.strip()
    return s.startswith("{") or s.startswith("[")


def _is_feishu_platform(platform: str) -> bool:
    """Return whether platform should use Feishu/Lark card rendering."""
    return platform.lower() in {"feishu", "lark"}


def _text(value: str) -> dict[str, str]:
    """Build a Feishu plain text object."""
    return {"tag": "plain_text", "content": value}


def _markdown(content: str) -> dict[str, str]:
    """Build a Feishu Markdown component."""
    return {"tag": "markdown", "content": content}


def _collapsible_panel(title: str, content: str) -> dict[str, Any]:
    """Build a Feishu Card JSON 2.0 collapsible panel."""
    return {
        "tag": "collapsible_panel",
        "expanded": False,
        "header": {
            "title": _text(title),
            "icon": {
                "tag": "standard_icon",
                "token": "down-small-ccm_outlined",
                "size": "16px 16px",
            },
            "icon_position": "right",
            "icon_expanded_angle": -180,
        },
        "border": {"color": "grey", "corner_radius": "5px"},
        "elements": [_markdown(content)],
    }


def _extract_headers(headers_str: Optional[str]) -> dict:
    """Parse custom headers from a multi-line "Key: Value" string.

    Args:
        headers_str: Multi-line string, each line "Key: Value"

    Returns:
        dict: Parsed headers as key-value pairs
    """
    if not headers_str:
        return {}

    headers = {}
    for line in headers_str.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(":", 1)
        if len(parts) != 2:
            logger.warning("Invalid webhook header line: %s", line)
            continue
        k, v = parts[0].strip(), parts[1].strip()
        headers[k] = v

    return headers


class WebhookNotifier:
    """Sends webhook notifications after pipeline completion or failure."""

    def __init__(self, config: WebhookConfig, console=None):
        self.config = config
        self.url = os.getenv(config.url_env or "") if config.url_env else None
        if console is None:
            try:
                from ..console import make_console
                self.console = make_console()
            except ImportError:
                class DummyConsole:
                    def print(self, *args, **kwargs):
                        print(*args, **kwargs)
                self.console = DummyConsole()
        else:
            self.console = console

    def _render_request_components(self, variables: dict) -> tuple[str, str | None, dict[str, str]]:
        """Render the final request URL, body, and headers for the given variables."""
        request_url = _render(self.url or "", variables)

        content_type = "application/x-www-form-urlencoded"
        body_content = None
        raw_body = variables.get("_request_body_override", self.config.request_body)
        body_variables = _prepare_variables_for_body(raw_body, variables)

        if raw_body:
            if isinstance(raw_body, (dict, list)):
                rendered_obj = _render(raw_body, body_variables)
                body_content = json.dumps(rendered_obj, ensure_ascii=False)
                content_type = "application/json"
            elif isinstance(raw_body, str) and raw_body.strip():
                if _isjson(raw_body):
                    try:
                        template_obj = json.loads(raw_body)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            "Webhook JSON request_body strings must be valid JSON before "
                            "placeholder rendering; use a dict/list request_body for structured payloads."
                        ) from exc
                    rendered_obj = _render(template_obj, body_variables)
                    body_content = json.dumps(rendered_obj, ensure_ascii=False)
                    content_type = "application/json"
                else:
                    body_content = _render(raw_body, body_variables)

        headers = _extract_headers(self.config.headers)
        headers["Content-Type"] = content_type
        return request_url, body_content, headers

    def _can_use_feishu_collapsible(self) -> bool:
        """Return whether this notifier should render Feishu collapsible cards."""
        platform = getattr(self.config, "platform", "generic")
        layout = getattr(self.config, "layout", "markdown")
        return _is_feishu_platform(platform) and layout == "collapsible"

    def _build_feishu_collapsible_overview(
        self,
        item_count: int,
        all_items_count: int,
        date: str,
        lang: str,
    ) -> str:
        """Build a non-redundant overview for a card that already lists item panels."""
        if lang == "zh":
            if item_count == 0:
                return (
                    f"# Horizon Brief 每日速递 - {date}\n\n"
                    f"> 已分析 {all_items_count} 条内容，暂无达到重要性阈值的资讯。"
                )
            return (
                f"# Horizon Brief 每日速递 - {date}\n\n"
                f"> 从 {all_items_count} 条内容中筛选出 {item_count} 条重要资讯。\n\n"
                "点击下方新闻面板即可在飞书内展开阅读全文。"
            )

        if item_count == 0:
            return (
                f"# Horizon Brief Daily - {date}\n\n"
                f"> Analyzed {all_items_count} items, but none met the importance threshold."
            )

        return (
            f"# Horizon Brief Daily - {date}\n\n"
            f"> Selected {item_count} important items from {all_items_count} fetched items.\n\n"
            "Expand the panels below to read the full briefing inside Feishu/Lark."
        )

    def _build_feishu_collapsible_body(
        self,
        important_items: List[ContentItem],
        all_items_count: int,
        date: str,
        lang: str,
        summarizer: DailySummarizer,
    ) -> dict[str, Any]:
        """Build a single Feishu Card JSON 2.0 message with collapsed item details."""
        overview = self._build_feishu_collapsible_overview(
            item_count=len(important_items),
            all_items_count=all_items_count,
            date=date,
            lang=lang,
        )
        elements: list[dict[str, Any]] = [_markdown(overview)]

        for item_index, item in enumerate(important_items, start=1):
            title = str(item.metadata.get(f"title_{lang}") or item.title)
            score = item.ai_score or "?"
            panel_title = f"{item_index}. {title} ⭐️ {score}/10"
            item_content = summarizer.generate_webhook_item(
                item,
                language=lang,
                index=item_index,
                total=len(important_items),
            )
            elements.append(_collapsible_panel(
                panel_title,
                _format_markdown_for_webhook(item_content),
            ))

        return {
            "msg_type": "interactive",
            "card": {
                "schema": "2.0",
                "config": {
                    "wide_screen_mode": True,
                    "update_multi": True,
                },
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": (
                            f"Horizon Brief {date} 折叠日报" if lang == "zh"
                            else f"Horizon Brief {date} Collapsible Daily"
                        ),
                    },
                    "template": "blue",
                },
                "body": {
                    "elements": elements,
                },
            },
        }

    def build_preview(self, variables: dict) -> dict[str, Any]:
        """Build the fully rendered request for dry-run preview."""
        request_url, body_content, headers = self._render_request_components(variables)
        return {
            "url": request_url,
            "body": body_content,
            "headers": headers,
        }

    def build_safe_preview(self, variables: dict) -> dict[str, Any]:
        """Build a dry-run preview with URL, body, and headers redacted."""
        preview = self.build_preview(variables)
        body = preview["body"]
        return {
            "url": _redact_url(preview["url"]),
            "body": _redact_text(body) if body is not None else None,
            "headers": _redact_headers(preview["headers"]),
        }

    def build_daily_summary_messages(
        self,
        summary: str,
        important_items: List[ContentItem],
        all_items_count: int,
        date: str,
        lang: str,
        summarizer: DailySummarizer,
    ) -> List[dict[str, Any]]:
        """Build the variables for all webhook messages for one language."""
        webhook_languages = getattr(self.config, "languages", None)
        if webhook_languages and lang not in webhook_languages:
            return []

        base_vars = {
            "date": date,
            "language": lang,
            "important_items": len(important_items),
            "all_items": all_items_count,
            "result": "success",
            "timestamp": str(int(datetime.now(timezone.utc).timestamp())),
        }

        if self._can_use_feishu_collapsible():
            return [{
                **base_vars,
                "message_title": (
                    f"Horizon Brief {date} 折叠日报" if lang == "zh"
                    else f"Horizon Brief {date} Collapsible Daily"
                ),
                "message_kind": "collapsible",
                "summary": self._build_feishu_collapsible_overview(
                    item_count=len(important_items),
                    all_items_count=all_items_count,
                    date=date,
                    lang=lang,
                ),
                "_request_body_override": self._build_feishu_collapsible_body(
                    important_items=important_items,
                    all_items_count=all_items_count,
                    date=date,
                    lang=lang,
                    summarizer=summarizer,
                ),
            }]

        delivery = getattr(self.config, "delivery", "summary")
        if delivery == "summary_and_items":
            item_messages: List[dict[str, Any]] = []
            overview = summarizer.generate_webhook_overview(
                important_items, date, all_items_count, language=lang,
            )
            overview_message = {
                **base_vars,
                "message_title": (
                    f"Horizon Brief {date} 总览" if lang == "zh"
                    else f"Horizon Brief {date} Overview"
                ),
                "message_kind": "overview",
                "summary": overview,
            }
            for item_index, item in enumerate(important_items, start=1):
                title = str(item.metadata.get(f"title_{lang}") or item.title)
                item_summary = summarizer.generate_webhook_item(
                    item, language=lang, index=item_index,
                    total=len(important_items),
                )
                item_messages.append({
                    **base_vars,
                    "message_title": f"{item_index}/{len(important_items)} {title}",
                    "message_kind": "item",
                    "item_index": item_index,
                    "item_count": len(important_items),
                    "item_title": title,
                    "item_url": str(item.url),
                    "item_score": item.ai_score or "",
                    "summary": item_summary,
                })

            if getattr(self.config, "overview_position", "first") == "last":
                return list(reversed(item_messages)) + [overview_message]

            return [overview_message] + item_messages

        return [{
            **base_vars,
            "message_title": (
                f"Horizon Brief {date} 日报" if lang == "zh"
                else f"Horizon Brief {date} Daily"
            ),
            "message_kind": "summary",
            "summary": summary,
        }]

    async def notify(self, variables: dict) -> None:
        """Send a webhook notification with template variable substitution.

        If request_body is empty, sends a GET request.
        If request_body is provided, sends a POST request with
        auto-detected content-type

        Args:
            variables: Dict of template variable values to replace
                       in URL, request_body, and headers.
        """
        if not self.config.enabled:
            return

        if not self.url:
            logger.warning("Webhook enabled but URL is empty (env var %s not set), skipping notification.", self.config.url_env)
            return

        method = "GET"
        raw_body = self.config.request_body
        request_url, body_content, headers = self._render_request_components(variables)
        safe_request_url = _redact_url(request_url)
        if raw_body:
            method = "POST"
            logger.debug(
                "Webhook POST body (%d chars): %s",
                len(body_content or ""),
                _redact_text(body_content or "")[:2000],
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                if method == "GET":
                    response = await client.get(request_url, headers=headers)
                else:
                    response = await client.post(
                        request_url,
                        content=body_content.encode("utf-8"),
                        headers=headers,
                    )

            if response.status_code == 200:
                logger.info(
                    "Webhook sent OK. URL: %s, body: %s",
                    safe_request_url,
                    response.text[:500],
                )
            else:
                self.console.print(
                    f"[red]Webhook failed! status={response.status_code} "
                    f"response={response.text[:500]}[/red]"
                )
                logger.error(
                    "Webhook failed! URL: %s, status: %d, body: %s",
                    safe_request_url,
                    response.status_code,
                    response.text[:500],
                )

        except Exception as e:
            self.console.print(f"[red]Webhook call failed! Exception: {e}[/red]")
            logger.error("Webhook call failed! URL: %s, exception: %s", safe_request_url, e)

    async def send_daily_summary(
        self,
        summary: str,
        important_items: List[ContentItem],
        all_items_count: int,
        date: str,
        lang: str,
        summarizer: DailySummarizer,
    ) -> None:
        """Send daily summary webhook notification.

        Handles language filtering, delivery mode (summary vs summary_and_items),
        and variable construction internally.

        Args:
            summary: Full markdown summary text
            important_items: List of important content items
            all_items_count: Total number of items fetched
            date: Date string (YYYY-MM-DD)
            lang: Language code ("en" or "zh")
            summarizer: DailySummarizer instance for generating webhook overviews
        """
        messages = self.build_daily_summary_messages(
            summary=summary,
            important_items=important_items,
            all_items_count=all_items_count,
            date=date,
            lang=lang,
            summarizer=summarizer,
        )
        if not messages:
            self.console.print(
                f"🔕 Skipping {lang.upper()} webhook notification "
                f"(filtered by webhook.languages)"
            )
            return

        self.console.print(f"🔔 Sending {lang.upper()} webhook notification...")
        for message in messages:
            await self.notify(message)

    async def send_failure(
        self,
        date: str,
        error_message: str,
    ) -> None:
        """Send webhook notification when the pipeline fails.

        Args:
            date: Date string (YYYY-MM-DD)
            error_message: Description of the failure
        """
        self.console.print("🔔 Sending webhook failure notification...")
        await self.notify({
            "date": date,
            "language": "",
            "important_items": 0,
            "all_items": 0,
            "result": "failed",
            "timestamp": str(int(datetime.now(timezone.utc).timestamp())),
            "message_title": "Horizon Brief generation failed",
            "message_kind": "failure",
            "summary": f"generation failed: {error_message}",
        })
