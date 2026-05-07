from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ..models import ContentItem


class ClaimType(str, Enum):
    CONFIRMED_FACT = "confirmed_fact"
    OFFICIAL_STATEMENT = "official_statement"
    PARTY_CLAIM = "party_claim"
    ANALYSIS = "analysis"
    MARKET_REACTION = "market_reaction"
    UNVERIFIED_REPORT = "unverified_report"
    CORRECTION_OR_UPDATE = "correction_or_update"
    PRIMARY_STATEMENT = "primary_statement"


class CriticResult(BaseModel):
    passed: bool = Field(alias="pass")
    critical_issues: List[str] = Field(default_factory=list)
    minor_issues: List[str] = Field(default_factory=list)
    required_edits: List[str] = Field(default_factory=list)


class SourcePolicy(BaseModel):
    default_language: str = "ru"
    timezone: str = "Europe/Paris"
    source_domain_rules: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    allowed_social_primary_actors: List[str] = Field(default_factory=list)


def conservative_default_policy() -> SourcePolicy:
    mapping = {
        "reuters.com": {"role": "fact_layer", "tier": "tier1", "name": "Reuters"},
        "apnews.com": {"role": "fact_layer", "tier": "tier1", "name": "Associated Press"},
        "ft.com": {"role": "context_layer", "tier": "tier1", "name": "Financial Times"},
        "economist.com": {"role": "context_layer", "tier": "tier1", "name": "The Economist"},
        "bbc.com": {"role": "fact_layer", "tier": "tier1", "name": "BBC"},
        "bbc.co.uk": {"role": "fact_layer", "tier": "tier1", "name": "BBC"},
        "theguardian.com": {"role": "context_layer", "tier": "tier2", "name": "The Guardian"},
        "federalreserve.gov": {"role": "official_primary_source", "tier": "tier1", "name": "Federal Reserve"},
        "ecb.europa.eu": {"role": "official_primary_source", "tier": "tier1", "name": "European Central Bank"},
        "cbr.ru": {"role": "official_primary_source", "tier": "tier1", "name": "Bank of Russia"},
        "sec.gov": {"role": "official_primary_source", "tier": "tier1", "name": "SEC"},
        "jpl.nasa.gov": {"role": "science_primary_source", "tier": "tier1", "name": "NASA JPL"},
        "nasa.gov": {"role": "science_primary_source", "tier": "tier1", "name": "NASA"},
        "esa.int": {"role": "science_primary_source", "tier": "tier1", "name": "ESA"},
        "cern.ch": {"role": "science_primary_source", "tier": "tier1", "name": "CERN"},
        "home.cern": {"role": "science_primary_source", "tier": "tier1", "name": "CERN"},
        "nsf.gov": {"role": "science_primary_source", "tier": "tier1", "name": "NSF"},
        "nih.gov": {"role": "science_primary_source", "tier": "tier1", "name": "NIH"},
        "noaa.gov": {"role": "science_primary_source", "tier": "tier1", "name": "NOAA"},
        "cdc.gov": {"role": "science_primary_source", "tier": "tier1", "name": "CDC"},
        "who.int": {"role": "science_primary_source", "tier": "tier1", "name": "WHO"},
        "interfax.ru": {"role": "russian_institutional_frame", "tier": "tier2", "name": "Interfax"},
        "kommersant.ru": {"role": "russian_institutional_frame", "tier": "tier2", "name": "Kommersant"},
        "rbc.ru": {"role": "russian_institutional_frame", "tier": "tier2", "name": "RBC"},
        "mos.ru": {"role": "official_primary_source", "tier": "tier2", "name": "mos.ru"},
        "nature.com": {"role": "science_primary_source", "tier": "tier1", "name": "Nature"},
        "science.org": {"role": "science_primary_source", "tier": "tier1", "name": "Science"},
        "pnas.org": {"role": "science_primary_source", "tier": "tier1", "name": "PNAS"},
        "cell.com": {"role": "science_primary_source", "tier": "tier1", "name": "Cell Press"},
        "plos.org": {"role": "science_primary_source", "tier": "tier1", "name": "PLOS"},
        "elifesciences.org": {"role": "science_primary_source", "tier": "tier1", "name": "eLife"},
        "quantamagazine.org": {"role": "context_layer", "tier": "tier1", "name": "Quanta Magazine"},
        "sciencenews.org": {"role": "context_layer", "tier": "tier1", "name": "Science News"},
        "scientificamerican.com": {"role": "context_layer", "tier": "tier1", "name": "Scientific American"},
        "theconversation.com": {"role": "context_layer", "tier": "tier2", "name": "The Conversation"},
        "arstechnica.com": {"role": "context_layer", "tier": "tier2", "name": "Ars Technica"},
        "knowablemagazine.org": {"role": "context_layer", "tier": "tier1", "name": "Knowable Magazine"},
        "newscientist.com": {"role": "context_layer", "tier": "tier2", "name": "New Scientist"},
        "arxiv.org": {"role": "science_preprint", "tier": "tier2", "name": "arXiv"},
        "biorxiv.org": {"role": "science_preprint", "tier": "tier2", "name": "bioRxiv"},
        "connect.biorxiv.org": {"role": "science_preprint", "tier": "tier2", "name": "bioRxiv"},
        "medrxiv.org": {"role": "science_preprint", "tier": "tier2", "name": "medRxiv"},
        "connect.medrxiv.org": {"role": "science_preprint", "tier": "tier2", "name": "medRxiv"},
        "chemrxiv.org": {"role": "science_preprint", "tier": "tier2", "name": "ChemRxiv"},
        "cambridge.org": {"role": "science_preprint", "tier": "tier2", "name": "ChemRxiv / Cambridge Open Engage"},
        "eurekalert.org": {"role": "science_source_finder", "tier": "tier2", "name": "EurekAlert"},
        "x.com": {"role": "social_primary_statement_only", "tier": "unknown", "name": "Twitter/X"},
        "twitter.com": {"role": "social_primary_statement_only", "tier": "unknown", "name": "Twitter/X"},
        "t.me": {"role": "blocked_as_fact_source", "tier": "unknown", "name": "Telegram"},
        "telegram.org": {"role": "blocked_as_fact_source", "tier": "unknown", "name": "Telegram"},
        "reddit.com": {"role": "blocked_as_fact_source", "tier": "unknown", "name": "Reddit"},
        "news.ycombinator.com": {"role": "blocked_as_fact_source", "tier": "unknown", "name": "Hacker News"},
        "github.com": {"role": "tech_primary_source", "tier": "tier2", "name": "GitHub"},
    }
    return SourcePolicy(source_domain_rules=mapping)


def load_source_policy(path: str) -> Tuple[SourcePolicy, str]:
    p = Path(path)
    if not p.exists():
        return conservative_default_policy(), f"Policy file missing: {path}. Using conservative defaults."
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        policy = conservative_default_policy()
        merged_rules = dict(policy.source_domain_rules)
        merged_rules.update(raw.get("source_domain_rules", {}))
        return SourcePolicy(
            default_language=raw.get("default_language", "ru"),
            timezone=raw.get("timezone", "Europe/Paris"),
            source_domain_rules=merged_rules,
            allowed_social_primary_actors=raw.get("allowed_social_primary_actors", []),
        ), ""
    except Exception as exc:
        return conservative_default_policy(), f"Invalid policy file {path}: {exc}. Using conservative defaults."


class SourcePolicyClassifier:
    def __init__(self, policy: SourcePolicy):
        self.policy = policy

    def classify(self, item: ContentItem) -> Dict[str, str]:
        domain = urlparse(str(item.url)).netloc.lower().replace("www.", "")
        source_type = item.source_type.value
        rule = None
        matches = [
            (d, r)
            for d, r in self.policy.source_domain_rules.items()
            if domain == d or domain.endswith("." + d)
        ]
        if matches:
            _, rule = max(matches, key=lambda match: len(match[0]))

        role = "unclassified"
        tier = "unknown"
        notes = ""
        if rule:
            role = rule.get("role", role)
            tier = rule.get("tier", tier)
        if source_type == "reddit":
            role, tier = "blocked_as_fact_source", "unknown"
            out_name = "Reddit"
        elif source_type == "hackernews":
            role, tier = "blocked_as_fact_source", "unknown"
            out_name = "Hacker News"
        elif source_type == "telegram":
            role, tier = "blocked_as_fact_source", "unknown"
            out_name = "Telegram"
        elif source_type == "twitter":
            role, tier = "social_primary_statement_only", "unknown"
            out_name = "Twitter/X"
        elif "github.com" in domain and "/releases" in str(item.url):
            role, tier = "tech_primary_source", "tier2"
            out_name = "GitHub"
        else:
            out_name = (rule or {}).get("name")
        if role == "science_preprint":
            notes = "not peer-reviewed"
        elif role == "science_source_finder":
            notes = "requires verification with paper, journal, or institution"

        out = {
            "source_role": role,
            "source_reliability_tier": tier,
            "source_policy_notes": notes,
            "discovery_source_type": source_type,
            "content_source_domain": domain,
            "source_role_reason": f"classified by source_type/domain: {source_type}/{domain}",
        }
        if out_name:
            out["source_name"] = out_name
        return out


class EvidenceChecker:
    def __init__(self, time_window_hours: int = 24):
        self.time_window_hours = time_window_hours

    def audit_item(self, item: ContentItem) -> Dict[str, Any]:
        meta = item.metadata
        claim_type = meta.get("claim_type", ClaimType.ANALYSIS.value)
        confidence = meta.get("confidence", "medium")
        conflicts: List[str] = []
        unsupported: List[str] = []
        missing: List[str] = []

        role = meta.get("source_role", "unclassified")
        if role in {"blocked_as_fact_source", "unclassified"} and claim_type == ClaimType.CONFIRMED_FACT.value:
            claim_type = ClaimType.UNVERIFIED_REPORT.value
            confidence = "low"
            unsupported.append("source policy does not allow confirmed_fact")

        if role == "science_preprint":
            note = "not peer-reviewed"
            current_notes = str(meta.get("source_policy_notes") or "")
            if note not in current_notes:
                meta["source_policy_notes"] = f"{current_notes}; {note}".strip("; ")
            if meta.get("evidence_strength") == "high":
                meta["evidence_strength"] = "medium"
            if claim_type == ClaimType.CONFIRMED_FACT.value:
                claim_type = ClaimType.UNVERIFIED_REPORT.value
                confidence = "low"
                meta["evidence_strength"] = "low"
                unsupported.append("preprint is not peer-reviewed")
            elif confidence == "high":
                confidence = "medium"

        if role == "science_source_finder":
            note = "requires verification with paper, journal, or institution"
            current_notes = str(meta.get("source_policy_notes") or "")
            if note not in current_notes:
                meta["source_policy_notes"] = f"{current_notes}; {note}".strip("; ")
            if meta.get("evidence_strength") == "high":
                meta["evidence_strength"] = "medium"
            if claim_type == ClaimType.CONFIRMED_FACT.value:
                claim_type = ClaimType.UNVERIFIED_REPORT.value
                confidence = "low"
                meta["evidence_strength"] = "low"
                unsupported.append("source finder is not independent confirmation")
            elif confidence == "high":
                confidence = "medium"

        if role == "social_primary_statement_only":
            original_claim_type = claim_type
            if claim_type in {
                ClaimType.CONFIRMED_FACT.value,
                ClaimType.OFFICIAL_STATEMENT.value,
                ClaimType.PARTY_CLAIM.value,
                ClaimType.MARKET_REACTION.value,
                ClaimType.CORRECTION_OR_UPDATE.value,
            }:
                claim_type = ClaimType.PRIMARY_STATEMENT.value
                confidence = "low" if original_claim_type == ClaimType.CONFIRMED_FACT.value else ("medium" if confidence == "high" else confidence)
                unsupported.append("social source downgraded to primary_statement")

        sensitive = bool(meta.get("sensitive_topic", False))
        if sensitive and role == "russian_institutional_frame" and claim_type == ClaimType.CONFIRMED_FACT.value:
            claim_type = ClaimType.PARTY_CLAIM.value
            confidence = "medium" if confidence == "high" else confidence
            unsupported.append("sensitive claim downgraded for russian institutional source")
        if sensitive and role == "official_primary_source" and claim_type == ClaimType.CONFIRMED_FACT.value:
            claim_type = ClaimType.OFFICIAL_STATEMENT.value
            confidence = "medium" if confidence == "high" else confidence

        if item.published_at is None:
            missing.append("missing publication date")
            confidence = "low"
        else:
            published_at = item.published_at
            if published_at.tzinfo is not None:
                published_at = published_at.astimezone(timezone.utc).replace(tzinfo=None)
            if published_at < datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=self.time_window_hours):
                conflicts.append("outside time window")

        if claim_type in {"official_statement", "party_claim"} and confidence == "high":
            confidence = "medium"

        meta.update({
            "claim_type": claim_type,
            "confidence": confidence,
            "source_conflicts": conflicts,
            "unsupported_claims": unsupported,
            "missing_evidence": missing,
            "requires_deep_review": bool(conflicts or unsupported or missing),
        })
        return meta


def prefilter_personal_candidates(
    items: List[ContentItem],
    policy: Optional[SourcePolicy],
) -> tuple[List[ContentItem], List[Dict[str, str]]]:
    """Drop sources that personal mode must not send to LLM scoring."""
    allowed = {a.lower() for a in (policy.allowed_social_primary_actors if policy else [])}
    candidates: List[ContentItem] = []
    excluded: List[Dict[str, str]] = []
    for item in items:
        role = item.metadata.get("source_role", "unclassified")
        parsed = urlparse(str(item.url))
        path_parts = [p for p in parsed.path.split("/") if p]
        url_handle = path_parts[0].lower().lstrip("@") if path_parts else ""
        meta = item.metadata
        handles = {
            (item.author or "").lower().lstrip("@"),
            str(meta.get("username", "")).lower().lstrip("@"),
            str(meta.get("handle", "")).lower().lstrip("@"),
            url_handle,
        }
        reason = None
        if role == "blocked_as_fact_source":
            reason = "blocked source"
        elif role == "unclassified":
            reason = "unclassified"
        elif role == "social_primary_statement_only" and handles.isdisjoint(allowed):
            reason = "social-only"
        elif role == "social_primary_statement_only":
            item.metadata["claim_type"] = "primary_statement"

        if reason:
            excluded.append({"item": item.title, "reason": reason})
        else:
            candidates.append(item)
    return candidates, excluded


def select_personal_important_items(
    items: List[ContentItem],
    *,
    checker: EvidenceChecker,
    min_importance: float,
    min_importance_priority_topics: float,
    require_dates: bool,
    priority_topics: set[str],
) -> tuple[List[ContentItem], List[Dict[str, str]]]:
    """Apply deterministic post-LLM evidence and threshold rules."""
    important: List[ContentItem] = []
    excluded: List[Dict[str, str]] = []
    for item in items:
        checker.audit_item(item)
        topic = item.metadata.get("topic", "other")
        threshold = min_importance_priority_topics if topic in priority_topics else min_importance
        reason = None
        if item.metadata.get("include") is False:
            reason = "weak evidence"
        elif item.metadata.get("source_role") in {"blocked_as_fact_source", "unclassified"}:
            reason = "blocked source" if item.metadata.get("source_role") == "blocked_as_fact_source" else "unclassified"
        elif require_dates and item.published_at is None:
            reason = "missing publication date"
        elif "outside time window" in item.metadata.get("source_conflicts", []):
            reason = "outside time window"
        elif (item.ai_score or 0) < threshold:
            reason = "low significance"

        if reason:
            excluded.append({"item": item.title, "reason": reason})
        else:
            important.append(item)
    return important, excluded


def drop_items_flagged_by_critic(items: List[ContentItem], critic: CriticResult) -> List[ContentItem]:
    """Remove items whose ids are explicitly named in critic issues."""
    bad_ids: set[str] = set()
    for issue in critic.critical_issues:
        match = re.search(r"\bfor\s+(.+)$", issue)
        if match:
            bad_ids.add(match.group(1).strip())
    if not bad_ids:
        return items
    return [item for item in items if item.id not in bad_ids]


class PersonalBriefingRenderer:
    PRIORITY_TOPICS = {"russia", "moscow", "world_economy", "tech_ai", "open_source", "big_tech", "science"}
    REASON_LABELS = {
        "blocked source": "заблокированный источник",
        "unclassified": "источник не классифицирован",
        "social-only": "социальный источник не разрешен",
        "weak evidence": "слабая доказательная база",
        "outside time window": "вне временного окна",
        "missing publication date": "нет даты публикации",
        "low significance": "низкая значимость",
    }

    def render(self, date: str, items: List[ContentItem], tracked: Optional[List[Dict[str, str]]] = None) -> str:
        if not items:
            lines = [f"# Сводка — {date}", "", "Сегодня нет событий, которые проходят заданный порог значимости и доказательности."]
            if tracked:
                lines += ["", "## Отслеживалось, но не включено"] + [
                    f"- {t['item']} — причина: {self.REASON_LABELS.get(t['reason'], t['reason'])}"
                    for t in tracked
                ]
            return "\n".join(lines)

        highs = [i for i in items if i.metadata.get("confidence") != "low"]
        disputed = [i for i in items if i.metadata.get("confidence") == "low"]
        out = [f"# Сводка — {date}", ""]

        sections = [
            ("Россия", "russia"), ("Москва", "moscow"), ("Мировая экономика", "world_economy"),
            ("Технологии: AI, Open Source, Big Tech", {"tech_ai", "open_source", "big_tech"}),
            ("Наука", "science"), ("Мир", "world"), ("Другое", "other")
        ]
        rendered_ids: set[str] = set()
        for title, topic in sections:
            if topic == "other":
                sec = [i for i in highs if i.id not in rendered_ids]
            else:
                sec = [i for i in highs if (i.metadata.get("topic") in topic if isinstance(topic, set) else i.metadata.get("topic") == topic)]
            if not sec:
                continue
            out += ["", f"## {title}"]
            for it in sec:
                out += self._card(it)
                rendered_ids.add(it.id)

        if disputed:
            out += ["", "## Спорные / слабоподтверждённые сообщения"]
            for it in disputed:
                out += self._card(it)
        return "\n".join(out)

    @staticmethod
    def _role_label(role: str) -> str:
        mapping = {
            "russian_institutional_frame": "российская институциональная рамка",
            "official_primary_source": "официальный источник",
        }
        return mapping.get(role, role)

    @staticmethod
    def _supporting_sources_text(meta: Dict[str, Any]) -> str:
        parts = []
        for src in meta.get("supporting_sources", []):
            parts.append(
                f"{src.get('source_name', 'unknown')} ({src.get('source_role', 'unclassified')}), "
                f"{src.get('publication_date', 'date unknown')}, {src.get('source_url', '')}"
            )
        return " | ".join(parts)

    def _card(self, item: ContentItem) -> List[str]:
        m = item.metadata
        date = item.published_at.date().isoformat() if item.published_at else "date unknown"
        conf = m.get("confidence", "low" if not item.published_at else "medium")
        confirmed = "; ".join(m.get("confirmed_details", [])) if isinstance(m.get("confirmed_details"), list) else str(m.get("confirmed_details", ""))
        claims = "; ".join(m.get("who_claims", [])) if isinstance(m.get("who_claims"), list) else str(m.get("who_claims", ""))
        return [
            f"### {item.title}",
            f"- Что произошло: {m.get('summary', item.ai_summary or item.title)}",
            f"- Что подтверждено: {confirmed or 'Недостаточно независимого подтверждения'}",
            f"- Кто что утверждает: {claims or m.get('claim_type', 'analysis')}",
            f"- Оценка доказательств: {m.get('evidence_strength', 'low')}",
            f"- Почему важно: {m.get('why_it_matters', item.ai_reason or '')}",
            f"- Уверенность: {conf}",
            f"- Источники: {m.get('source_name', item.source_type.value)} ({self._role_label(m.get('source_role', 'unclassified'))}), {date}, {item.url}",
        ] + ([f"- Поддерживающие источники: {self._supporting_sources_text(m)}"] if m.get("supporting_sources") else []) + [""]


def run_briefing_critic(markdown: str, items: List[ContentItem]) -> CriticResult:
    critical, minor, edits = [], [], []
    if "date unknown" in markdown:
        minor.append("some items have unknown publication dates")
    card_count = markdown.count("### ")
    expected_cards = len(set([i.id for i in items]))
    if card_count < expected_cards:
        critical.append("missing cards detected")
    elif card_count > expected_cards:
        critical.append("duplicate cards detected")
    for it in items:
        if it.metadata.get("source_role") in {"blocked_as_fact_source", "unclassified"} and it.metadata.get("claim_type") == "confirmed_fact":
            critical.append(f"source_role violation for {it.id}")
        if it.metadata.get("source_role") in {"blocked_as_fact_source", "unclassified"} and it.metadata.get("confidence") != "low":
            critical.append(f"invalid main-section source role for {it.id}")
        if it.metadata.get("source_role") == "social_primary_statement_only" and it.metadata.get("claim_type") == "confirmed_fact":
            critical.append(f"social confirmed_fact violation for {it.id}")
        if it.metadata.get("source_role") == "russian_institutional_frame" and it.metadata.get("sensitive_topic") and it.metadata.get("claim_type") == "confirmed_fact":
            critical.append(f"sensitive russian institutional claim not downgraded for {it.id}")
        if not it.metadata.get("source_url"):
            critical.append(f"missing source url for {it.id}")
        if not it.published_at and it.metadata.get("confidence") != "low":
            critical.append(f"missing date must imply low confidence for {it.id}")
        if it.metadata.get("confidence") == "low" and "## Спорные / слабоподтверждённые сообщения" not in markdown:
            critical.append("low confidence item shown outside disputed section")
    if critical:
        edits.append("revise sections and evidence labels")
    return CriticResult.model_validate({"pass": not critical, "critical_issues": critical, "minor_issues": minor, "required_edits": edits})
