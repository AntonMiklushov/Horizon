from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..models import ContentItem


DISCOVERY_ONLY_SOURCE_TYPES = {"telegram", "reddit", "hackernews"}
CONFIRMING_SOURCE_ROLES = {
    "fact_layer",
    "context_layer",
    "science_primary_source",
    "tech_primary_source",
}
TRUSTED_EVIDENCE_ROLES = CONFIRMING_SOURCE_ROLES | {
    "official_primary_source",
    "russian_institutional_frame",
    "science_preprint",
    "science_source_finder",
}
ALLOWED_TOPICS = {
    "russia",
    "moscow",
    "world_economy",
    "tech_ai",
    "open_source",
    "big_tech",
    "science",
    "world",
    "other",
}

SENSITIVE_TOPIC_RE = re.compile(
    r"\b("
    r"war|invasion|combat|missile|drone|attack|sanction|sanctions|election|protest|riot|"
    r"arrest|detention|terror|terrorism|mobilization|security|coup|hostage|explosion|"
    r"войн[а-я]*|вторжен[а-я]*|боев[а-я]*|ракет[а-я]*|дрон[а-я]*|атак[а-я]*|"
    r"санкци[а-я]*|выбор[а-я]*|протест[а-я]*|митинг[а-я]*|арест[а-я]*|задержан[а-я]*|"
    r"террор[а-я]*|теракт[а-я]*|мобилизаци[а-я]*|безопасност[а-я]*|переворот[а-я]*|заложник[а-я]*|взрыв[а-я]*"
    r")\b",
    re.IGNORECASE,
)


class ClaimType(str, Enum):
    CONFIRMED_FACT = "confirmed_fact"
    OFFICIAL_STATEMENT = "official_statement"
    PARTY_CLAIM = "party_claim"
    ANALYSIS = "analysis"
    MARKET_REACTION = "market_reaction"
    UNVERIFIED_REPORT = "unverified_report"
    CORRECTION_OR_UPDATE = "correction_or_update"
    PRIMARY_STATEMENT = "primary_statement"


STATEMENT_CLAIM_TYPES = {
    ClaimType.OFFICIAL_STATEMENT.value,
    ClaimType.PARTY_CLAIM.value,
    ClaimType.PRIMARY_STATEMENT.value,
    ClaimType.MARKET_REACTION.value,
    ClaimType.CORRECTION_OR_UPDATE.value,
}


class CriticResult(BaseModel):
    passed: bool = Field(alias="pass")
    critical_issues: List[str] = Field(default_factory=list)
    minor_issues: List[str] = Field(default_factory=list)
    required_edits: List[str] = Field(default_factory=list)


class SourceRule(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str = "unclassified"
    tier: str = "unknown"
    name: str = ""
    can_confirm_fact: Optional[bool] = None
    can_confirm_sensitive: Optional[bool] = None
    counts_as_independent_confirmation: Optional[bool] = None
    notes: str = ""

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    @field_validator("role", "tier", "name", "notes", mode="before")
    @classmethod
    def _stringify(cls, value: Any) -> str:
        return "" if value is None else str(value)


class DiscoveryPolicy(BaseModel):
    model_config = ConfigDict(extra="ignore")

    discovery_source_types: List[str] = Field(default_factory=lambda: sorted(DISCOVERY_ONLY_SOURCE_TYPES))
    allow_trusted_external_links: bool = True
    exclude_discovery_only_without_trusted_link: bool = True
    allowlisted_telegram_channels: List[str] = Field(default_factory=list)


class CorroborationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    sensitive_requires_independent_confirmation: bool = True
    min_independent_confirmations: int = Field(default=1, ge=0, le=5)
    confirming_roles: List[str] = Field(default_factory=lambda: sorted(CONFIRMING_SOURCE_ROLES))


class PersonalEnrichmentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    disable_for_sensitive_topics: bool = True
    filter_search_results_by_source_policy: bool = True


class SelectionCapsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    max_items_per_source_role: Dict[str, int] = Field(default_factory=dict)
    max_sensitive_statement_items: int = 2


class SourcePolicy(BaseModel):
    model_config = ConfigDict(extra="ignore")

    default_language: str = "ru"
    timezone: str = "Europe/Paris"
    source_domain_rules: Dict[str, SourceRule] = Field(default_factory=dict)
    allowed_social_primary_actors: List[str] = Field(default_factory=list)
    discovery: DiscoveryPolicy = Field(default_factory=DiscoveryPolicy)

    @field_validator("source_domain_rules", mode="before")
    @classmethod
    def _coerce_rules(cls, value: Any) -> Dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        return {str(domain).lower().replace("www.", ""): rule for domain, rule in value.items()}


def conservative_default_policy() -> SourcePolicy:
    mapping = {
        "reuters.com": {"role": "fact_layer", "tier": "tier1", "name": "Reuters"},
        "reutersagency.com": {"role": "fact_layer", "tier": "tier1", "name": "Reuters"},
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
            discovery=raw.get("discovery", {}),
        ), ""
    except Exception as exc:
        return conservative_default_policy(), f"Invalid policy file {path}: {exc}. Using conservative defaults."


def _domain(url: str) -> str:
    return urlparse(str(url)).netloc.lower().replace("www.", "")


def _discovery_url(item: ContentItem) -> str:
    meta = item.metadata
    return str(
        meta.get("discovery_url")
        or meta.get("msg_url")
        or meta.get("discussion_url")
        or item.url
    )


def _discovery_name(item: ContentItem) -> str:
    meta = item.metadata
    if meta.get("channel"):
        return f"@{meta['channel']}"
    if meta.get("subreddit"):
        return f"r/{meta['subreddit']}"
    return str(meta.get("source_name") or item.author or item.source_type.value)


def _append_note(existing: str, note: str) -> str:
    if not existing:
        return note
    if note in existing:
        return existing
    return f"{existing}; {note}"


def _tracked_exclusion(item: ContentItem, reason: str) -> Dict[str, Any]:
    meta = item.metadata
    return {
        "id": item.id,
        "item": item.title,
        "reason": reason,
        "source_role": meta.get("source_role", "unclassified"),
        "discovery_source_type": meta.get("discovery_source_type", item.source_type.value),
        "discovery_source_name": meta.get("discovery_source_name") or _discovery_name(item),
        "content_source_domain": meta.get("content_source_domain") or _domain(str(item.url)),
        "canonical_url": str(item.url),
        "discovery_url": meta.get("discovery_url") or _discovery_url(item),
        "policy_decision": meta.get("policy_decision", ""),
    }


class SourcePolicyClassifier:
    def __init__(self, policy: SourcePolicy):
        self.policy = policy

    def classify(self, item: ContentItem) -> Dict[str, str]:
        canonical_url = str(item.url)
        domain = _domain(canonical_url)
        source_type = item.source_type.value
        discovery_url = _discovery_url(item)
        discovery_domain = _domain(discovery_url)
        is_discovery_source = source_type in {s.lower() for s in self.policy.discovery.discovery_source_types}
        is_external_link = bool(discovery_domain and domain and discovery_domain != domain)
        if is_discovery_source and domain and not _is_platform_domain(source_type, domain):
            is_external_link = True
        rule = self._rule_for_domain(domain)

        role = rule.role if rule else "unclassified"
        tier = rule.tier if rule else "unknown"
        notes = rule.notes if rule and rule.notes else ""
        out_name = rule.name if rule and rule.name else None

        discovery_role = "origin"
        policy_decision = "allow"
        if is_discovery_source:
            discovery_role = "discovery_signal"
            if is_external_link and self.policy.discovery.allow_trusted_external_links and role in TRUSTED_EVIDENCE_ROLES:
                policy_decision = "allow_linked_evidence"
            else:
                role, tier = self._discovery_fallback_role(item, source_type)
                out_name = self._discovery_source_name(source_type)
                policy_decision = (
                    "allowlisted_discovery_statement"
                    if role in {"primary_statement", "unverified_report"}
                    else "exclude_discovery_only"
                )
        elif source_type == "twitter":
            role, tier = "social_primary_statement_only", "unknown"
            out_name = "Twitter/X"
            policy_decision = "social_primary_statement_only"
        elif "github.com" in domain and "/releases" in canonical_url:
            role, tier = "tech_primary_source", "tier2"
            out_name = "GitHub"

        if role == "science_preprint":
            notes = _append_note(notes, "not peer-reviewed")
        elif role == "science_source_finder":
            notes = _append_note(notes, "requires verification with paper, journal, or institution")

        can_confirm_fact, can_confirm_sensitive, independent = self._capabilities(role, is_discovery_source)
        if role in {"blocked_as_fact_source", "unclassified"}:
            policy_decision = "exclude_" + role

        out = {
            "source_role": role,
            "source_reliability_tier": tier,
            "source_policy_notes": notes,
            "discovery_source_type": source_type,
            "discovery_source_name": _discovery_name(item),
            "discovery_url": discovery_url,
            "discovery_role": discovery_role,
            "content_source_domain": domain,
            "can_confirm_fact": can_confirm_fact,
            "can_confirm_sensitive": can_confirm_sensitive,
            "counts_as_independent_confirmation": independent,
            "policy_decision": policy_decision,
            "source_role_reason": f"classified by discovery/canonical domain: {source_type}/{domain}",
        }
        if out_name:
            out["source_name"] = out_name
        return out

    def _rule_for_domain(self, domain: str) -> Optional[SourceRule]:
        matches = [
            (d, r)
            for d, r in self.policy.source_domain_rules.items()
            if domain == d or domain.endswith("." + d)
        ]
        if not matches:
            return None
        _, rule = max(matches, key=lambda match: len(match[0]))
        return rule

    def _discovery_fallback_role(self, item: ContentItem, source_type: str) -> tuple[str, str]:
        if source_type == "telegram":
            channel = str(item.metadata.get("channel") or item.author or "").lower().lstrip("@")
            allowed = {c.lower().lstrip("@") for c in self.policy.discovery.allowlisted_telegram_channels}
            if channel and channel in allowed:
                return "primary_statement", "unknown"
        return "blocked_as_fact_source", "unknown"

    @staticmethod
    def _discovery_source_name(source_type: str) -> str:
        return {
            "reddit": "Reddit",
            "hackernews": "Hacker News",
            "telegram": "Telegram",
        }.get(source_type, source_type)

    @staticmethod
    def _capabilities(role: str, is_discovery_source: bool) -> tuple[bool, bool, bool]:
        can_confirm_fact = role in CONFIRMING_SOURCE_ROLES
        can_confirm_sensitive = role == "fact_layer"
        independent = can_confirm_fact and not is_discovery_source and role != "social_primary_statement_only"
        return can_confirm_fact, can_confirm_sensitive, independent


def _normalize_claim_type(value: Any) -> str:
    allowed = {member.value for member in ClaimType}
    normalized = str(value or ClaimType.ANALYSIS.value).strip().lower()
    return normalized if normalized in allowed else ClaimType.ANALYSIS.value


def _normalize_topic(value: Any) -> str:
    normalized = str(value or "other").strip().lower()
    return normalized if normalized in ALLOWED_TOPICS else "other"


def _normalize_confidence(value: Any) -> str:
    normalized = str(value or "medium").strip().lower()
    return normalized if normalized in {"low", "medium", "high"} else "medium"


def _metadata_bool(meta: Dict[str, Any], key: str, default: bool) -> bool:
    value = meta.get(key)
    if value is None:
        meta[key] = default
        return default
    return bool(value)


def _append_unique(values: List[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _append_source_policy_note(meta: Dict[str, Any], note: str) -> None:
    meta["source_policy_notes"] = _append_note(str(meta.get("source_policy_notes") or ""), note)


def _downgraded_claim_type_for_role(role: str) -> str:
    if role == "official_primary_source":
        return ClaimType.OFFICIAL_STATEMENT.value
    if role == "russian_institutional_frame":
        return ClaimType.PARTY_CLAIM.value
    if role in {"social_primary_statement_only", "primary_statement"}:
        return ClaimType.PRIMARY_STATEMENT.value
    return ClaimType.UNVERIFIED_REPORT.value


def detect_sensitive_topic(item: ContentItem) -> bool:
    meta = item.metadata
    parts = [
        item.title,
        item.content or "",
        item.ai_summary or "",
        str(meta.get("summary") or ""),
        str(meta.get("why_it_matters") or ""),
        " ".join(str(x) for x in meta.get("confirmed_details", []) if isinstance(meta.get("confirmed_details"), list)),
    ]
    return bool(SENSITIVE_TOPIC_RE.search("\n".join(parts)))


def _is_platform_domain(source_type: str, domain: str) -> bool:
    platform_domains = {
        "telegram": ("t.me", "telegram.org"),
        "reddit": ("reddit.com",),
        "hackernews": ("news.ycombinator.com",),
    }
    return any(domain == candidate or domain.endswith("." + candidate) for candidate in platform_domains.get(source_type, ()))


class EvidenceChecker:
    def __init__(self, time_window_hours: int = 24):
        self.time_window_hours = time_window_hours

    def audit_item(self, item: ContentItem) -> Dict[str, Any]:
        meta = item.metadata
        claim_type = _normalize_claim_type(meta.get("claim_type", ClaimType.ANALYSIS.value))
        confidence = _normalize_confidence(meta.get("confidence", "medium"))
        meta["topic"] = _normalize_topic(meta.get("topic", "other"))
        conflicts: List[str] = []
        unsupported: List[str] = list(meta.get("unsupported_claims") or []) if isinstance(meta.get("unsupported_claims"), list) else []
        missing: List[str] = []

        role = meta.get("source_role", "unclassified")
        if role in {"blocked_as_fact_source", "unclassified"} and claim_type == ClaimType.CONFIRMED_FACT.value:
            claim_type = ClaimType.UNVERIFIED_REPORT.value
            confidence = "low"
            _append_unique(unsupported, "source policy does not allow confirmed_fact")

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
                _append_unique(unsupported, "preprint is not peer-reviewed")
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
                _append_unique(unsupported, "source finder is not independent confirmation")
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
                _append_unique(unsupported, "social source downgraded to primary_statement")

        sensitive_auto = detect_sensitive_topic(item)
        meta["sensitive_topic_auto"] = sensitive_auto
        sensitive = bool(meta.get("sensitive_topic", False)) or sensitive_auto
        meta["sensitive_topic"] = sensitive

        can_confirm_fact, can_confirm_sensitive, independent = SourcePolicyClassifier._capabilities(
            role,
            str(meta.get("discovery_role", "")) == "discovery_signal",
        )
        can_confirm_fact = _metadata_bool(meta, "can_confirm_fact", can_confirm_fact)
        can_confirm_sensitive = _metadata_bool(meta, "can_confirm_sensitive", can_confirm_sensitive)
        independent = _metadata_bool(meta, "counts_as_independent_confirmation", independent)
        has_supporting_confirmation = bool(_supporting_confirmation_sources(item, sensitive=sensitive))

        if claim_type == ClaimType.CONFIRMED_FACT.value and (
            role in {"blocked_as_fact_source", "unclassified"}
            or not can_confirm_fact
            or not independent
            or str(meta.get("discovery_role", "")) == "discovery_signal"
        ):
            claim_type = _downgraded_claim_type_for_role(role)
            confidence = "low" if role in {"blocked_as_fact_source", "unclassified"} else ("medium" if confidence == "high" else confidence)
            _append_unique(unsupported, "source policy does not allow confirmed_fact")
            _append_source_policy_note(meta, "downgraded because source cannot independently confirm facts")

        if sensitive and (
            not can_confirm_sensitive
            or not independent
            or not has_supporting_confirmation
        ):
            if claim_type == ClaimType.CONFIRMED_FACT.value:
                claim_type = _downgraded_claim_type_for_role(role)
            if confidence == "high":
                confidence = "medium"
            _append_unique(unsupported, "sensitive topic lacks independent sensitive confirmation")
            _append_source_policy_note(meta, "downgraded because sensitive topic lacks independent confirmation")

        if sensitive and role == "russian_institutional_frame" and claim_type == ClaimType.CONFIRMED_FACT.value:
            claim_type = ClaimType.PARTY_CLAIM.value
            confidence = "medium" if confidence == "high" else confidence
            _append_unique(unsupported, "sensitive claim downgraded for russian institutional source")
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

        if confidence == "high" and not has_supporting_confirmation:
            confidence = "medium"
            if meta.get("evidence_strength") == "high":
                meta["evidence_strength"] = "medium"
            _append_unique(unsupported, "high confidence requires independent supporting source")
            _append_source_policy_note(meta, "confidence capped until corroborated by an independent supporting source")

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
) -> tuple[List[ContentItem], List[Dict[str, Any]]]:
    """Drop sources that personal mode must not send to LLM scoring."""
    allowed = {a.lower() for a in (policy.allowed_social_primary_actors if policy else [])}
    candidates: List[ContentItem] = []
    excluded: List[Dict[str, Any]] = []
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
        discovery_role = str(meta.get("discovery_role", ""))
        has_trusted_link = role in TRUSTED_EVIDENCE_ROLES and role not in {"blocked_as_fact_source", "unclassified"}
        reason = None
        if role == "blocked_as_fact_source":
            reason = "discovery-only" if discovery_role == "discovery_signal" else "blocked source"
        elif role == "unclassified":
            reason = "unclassified"
        elif discovery_role == "discovery_signal" and not has_trusted_link:
            reason = "discovery-only"
        elif role == "social_primary_statement_only" and handles.isdisjoint(allowed):
            reason = "social-only"
        elif role == "social_primary_statement_only":
            item.metadata["claim_type"] = "primary_statement"
        elif role == "primary_statement":
            item.metadata["claim_type"] = "primary_statement"
        elif role == "unverified_report":
            item.metadata["claim_type"] = "unverified_report"

        if reason:
            excluded.append(_tracked_exclusion(item, reason))
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
) -> tuple[List[ContentItem], List[Dict[str, Any]]]:
    """Apply deterministic post-LLM evidence and threshold rules."""
    important: List[ContentItem] = []
    excluded: List[Dict[str, Any]] = []
    for item in items:
        checker.audit_item(item)
        topic = _normalize_topic(item.metadata.get("topic", "other"))
        item.metadata["topic"] = topic
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
            excluded.append(_tracked_exclusion(item, reason))
        else:
            important.append(item)
    return important, excluded


class CorroborationGate:
    def __init__(self, config: Any | None = None):
        raw = config
        if raw is None:
            self.config = CorroborationConfig()
        elif isinstance(raw, CorroborationConfig):
            self.config = raw
        elif hasattr(raw, "model_dump"):
            self.config = CorroborationConfig.model_validate(raw.model_dump())
        else:
            self.config = CorroborationConfig.model_validate(raw)

    def apply(self, items: List[ContentItem]) -> List[ContentItem]:
        if not self.config.enabled:
            return items
        for item in items:
            self.audit(item)
        return items

    def audit(self, item: ContentItem) -> Dict[str, Any]:
        meta = item.metadata
        role = str(meta.get("source_role", "unclassified"))
        claim_type = _normalize_claim_type(meta.get("claim_type", ClaimType.ANALYSIS.value))
        confidence = _normalize_confidence(meta.get("confidence", "medium"))
        sensitive = bool(meta.get("sensitive_topic")) or bool(meta.get("sensitive_topic_auto"))
        confirming = self._confirming_sources(item, sensitive=sensitive)
        supporting_confirming = _supporting_confirmation_sources(
            item,
            sensitive=sensitive,
            confirming_roles=set(self.config.confirming_roles),
        )
        required = (
            self.config.min_independent_confirmations
            if sensitive and self.config.sensitive_requires_independent_confirmation
            else 0
        )
        passed = len(confirming) >= required
        notes: List[str] = []

        can_confirm_fact = bool(meta.get("can_confirm_fact"))
        can_confirm_sensitive = bool(meta.get("can_confirm_sensitive"))
        independent = bool(meta.get("counts_as_independent_confirmation"))
        if claim_type == ClaimType.CONFIRMED_FACT.value and (not can_confirm_fact or not independent):
            claim_type = _downgraded_claim_type_for_role(role)
            confidence = "low" if role in {"blocked_as_fact_source", "unclassified"} else ("medium" if confidence == "high" else confidence)
            notes.append("source policy does not allow confirmed_fact")

        if sensitive and role == "russian_institutional_frame" and claim_type == ClaimType.CONFIRMED_FACT.value and not passed:
            claim_type = ClaimType.PARTY_CLAIM.value
            confidence = "medium" if confidence == "high" else confidence
            notes.append("sensitive russian institutional item lacks independent corroboration")
        if sensitive and role == "official_primary_source" and claim_type == ClaimType.CONFIRMED_FACT.value and not passed:
            claim_type = ClaimType.OFFICIAL_STATEMENT.value
            confidence = "medium" if confidence == "high" else confidence
            notes.append("sensitive official source lacks external confirmation")
        if sensitive and claim_type == ClaimType.CONFIRMED_FACT.value and (not passed or not can_confirm_sensitive or not supporting_confirming):
            claim_type = _downgraded_claim_type_for_role(role)
            confidence = "medium" if confidence == "high" else confidence
            notes.append("sensitive item lacks independent sensitive confirmation")
        if confidence == "high" and not supporting_confirming:
            confidence = "medium"
            notes.append("high confidence downgraded pending independent corroboration")

        meta["claim_type"] = claim_type
        meta["confidence"] = confidence
        meta["corroboration"] = {
            "passed": passed,
            "required_independent_confirmations": required,
            "independent_confirmations": len(confirming),
            "confirming_sources": confirming,
            "notes": notes,
        }
        if notes:
            unsupported = meta.setdefault("unsupported_claims", [])
            if isinstance(unsupported, list):
                unsupported.extend(note for note in notes if note not in unsupported)
            meta["requires_deep_review"] = True
        return meta

    def _confirming_sources(self, item: ContentItem, *, sensitive: bool = False) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []
        primary = _source_confirmation_payload(item)
        if self._counts(primary, sensitive=sensitive):
            sources.append(primary)
        for raw in item.metadata.get("supporting_sources", []):
            if not isinstance(raw, dict):
                continue
            payload = _supporting_confirmation_payload(raw)
            if self._counts(payload, sensitive=sensitive) and not _same_confirmation_source(payload, sources):
                sources.append(payload)
        return sources

    def _counts(self, payload: Dict[str, Any], *, sensitive: bool) -> bool:
        return _confirmation_payload_counts(
            payload,
            sensitive=sensitive,
            confirming_roles=set(self.config.confirming_roles),
        )


def _source_confirmation_payload(item: ContentItem) -> Dict[str, Any]:
    meta = item.metadata
    return {
        "source_name": meta.get("source_name", item.source_type.value),
        "source_url": str(meta.get("source_url") or item.url),
        "source_role": meta.get("source_role", "unclassified"),
        "content_source_domain": meta.get("content_source_domain") or _domain(str(item.url)),
        "can_confirm_fact": bool(meta.get("can_confirm_fact")),
        "can_confirm_sensitive": bool(meta.get("can_confirm_sensitive")),
        "counts_as_independent_confirmation": bool(meta.get("counts_as_independent_confirmation")),
    }


def _supporting_confirmation_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    role = str(raw.get("source_role", "unclassified"))
    domain = str(raw.get("content_source_domain") or _domain(str(raw.get("source_url", ""))))
    can_confirm_fact = raw.get("can_confirm_fact")
    can_confirm_sensitive = raw.get("can_confirm_sensitive")
    independent = raw.get("counts_as_independent_confirmation")
    if can_confirm_fact is None:
        can_confirm_fact = role in CONFIRMING_SOURCE_ROLES
    if can_confirm_sensitive is None:
        can_confirm_sensitive = role == "fact_layer"
    if independent is None:
        independent = bool(can_confirm_fact) and str(raw.get("discovery_role", "")) != "discovery_signal"
    return {
        "source_name": raw.get("source_name", "unknown"),
        "source_url": raw.get("source_url", ""),
        "source_role": role,
        "content_source_domain": domain,
        "can_confirm_fact": bool(can_confirm_fact),
        "can_confirm_sensitive": bool(can_confirm_sensitive),
        "counts_as_independent_confirmation": bool(independent),
    }


def _same_confirmation_source(payload: Dict[str, Any], existing: List[Dict[str, Any]]) -> bool:
    domain = payload.get("content_source_domain")
    url = payload.get("source_url")
    return any(src.get("content_source_domain") == domain or src.get("source_url") == url for src in existing)


def _confirmation_payload_counts(
    payload: Dict[str, Any],
    *,
    sensitive: bool,
    confirming_roles: set[str] | None = None,
) -> bool:
    roles = confirming_roles or CONFIRMING_SOURCE_ROLES
    return bool(
        payload.get("counts_as_independent_confirmation")
        and payload.get("can_confirm_fact")
        and (not sensitive or payload.get("can_confirm_sensitive"))
        and payload.get("source_role") in roles
    )


def _supporting_confirmation_sources(
    item: ContentItem,
    *,
    sensitive: bool,
    confirming_roles: set[str] | None = None,
) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for raw in item.metadata.get("supporting_sources", []):
        if not isinstance(raw, dict):
            continue
        payload = _supporting_confirmation_payload(raw)
        if _confirmation_payload_counts(payload, sensitive=sensitive, confirming_roles=confirming_roles):
            sources.append(payload)
    return sources


def apply_personal_selection_caps(
    items: List[ContentItem],
    config: Any | None = None,
) -> tuple[List[ContentItem], List[Dict[str, Any]]]:
    caps = _coerce_selection_caps(config)
    if not caps.enabled:
        return items, []

    kept: List[ContentItem] = []
    excluded: List[Dict[str, Any]] = []
    role_counts: Dict[str, int] = {}
    sensitive_statement_count = 0
    for item in items:
        role = str(item.metadata.get("source_role", "unclassified"))
        role_limit = caps.max_items_per_source_role.get(role)
        if role_limit is not None and role_counts.get(role, 0) >= role_limit:
            excluded.append(_tracked_exclusion(item, "source role cap"))
            continue

        is_sensitive_statement = (
            bool(item.metadata.get("sensitive_topic"))
            and item.metadata.get("claim_type") in {"party_claim", "official_statement"}
        )
        if is_sensitive_statement and sensitive_statement_count >= caps.max_sensitive_statement_items:
            excluded.append(_tracked_exclusion(item, "sensitive statement cap"))
            continue

        role_counts[role] = role_counts.get(role, 0) + 1
        if is_sensitive_statement:
            sensitive_statement_count += 1
        kept.append(item)
    return kept, excluded


def _coerce_selection_caps(config: Any | None) -> SelectionCapsConfig:
    if config is None:
        return SelectionCapsConfig()
    if isinstance(config, SelectionCapsConfig):
        return config
    if hasattr(config, "model_dump"):
        return SelectionCapsConfig.model_validate(config.model_dump())
    return SelectionCapsConfig.model_validate(config)


def source_policy_decisions(items: List[ContentItem], excluded: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    return {
        "items": [
            {
                "id": item.id,
                "title": item.title,
                "canonical_url": str(item.url),
                "source_role": item.metadata.get("source_role"),
                "discovery_source_type": item.metadata.get("discovery_source_type"),
                "discovery_source_name": item.metadata.get("discovery_source_name"),
                "discovery_url": item.metadata.get("discovery_url"),
                "content_source_domain": item.metadata.get("content_source_domain"),
                "can_confirm_fact": item.metadata.get("can_confirm_fact"),
                "can_confirm_sensitive": item.metadata.get("can_confirm_sensitive"),
                "counts_as_independent_confirmation": item.metadata.get("counts_as_independent_confirmation"),
                "policy_decision": item.metadata.get("policy_decision"),
                "claim_type": item.metadata.get("claim_type"),
                "confidence": item.metadata.get("confidence"),
                "source_policy_notes": item.metadata.get("source_policy_notes"),
                "unsupported_claims": item.metadata.get("unsupported_claims"),
                "corroboration": item.metadata.get("corroboration"),
                "sensitive_topic_auto": item.metadata.get("sensitive_topic_auto"),
                "enrichment_skipped": item.metadata.get("enrichment_skipped"),
            }
            for item in items
        ],
        "excluded": list(excluded or []),
    }


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

    def render(
        self,
        date: str,
        items: List[ContentItem],
        tracked: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        context = context or {}
        if not items:
            lines = [f"# Сводка — {date}", "", "Сегодня нет событий, которые проходят заданный порог значимости и доказательности."]
            context_lines = []
            if context.get("total_fetched") is not None:
                context_lines.append(f"- Получено: {context['total_fetched']}")
            if context.get("source_items") is not None:
                context_lines.append(f"- Проверено на этапе отбора: {context['source_items']}")
            if context.get("selected_count") is not None:
                context_lines.append(f"- Прошло в итоговую сводку: {context['selected_count']}")
            if context.get("threshold") is not None:
                context_lines.append(f"- Порог значимости: {context['threshold']}")
            if context_lines:
                lines += ["", "## Контекст отбора"] + context_lines
            if tracked:
                lines += ["", "## Отслеживалось, но не включено"] + [
                    f"- {t['item']} — причина: {self.REASON_LABELS.get(t['reason'], t['reason'])}"
                    for t in tracked
                ]
            return "\n".join(lines)

        non_low = [i for i in items if i.metadata.get("confidence") != "low"]
        statements = [
            i
            for i in non_low
            if _normalize_claim_type(i.metadata.get("claim_type")) in STATEMENT_CLAIM_TYPES
        ]
        statement_ids = {i.id for i in statements}
        highs = [i for i in non_low if i.id not in statement_ids]
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

        if statements:
            out += ["", "## Заявления и сообщения, требующие контекста"]
            for it in statements:
                out += self._card(it)
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
        independent = "да" if m.get("counts_as_independent_confirmation") else "нет"
        sensitive_ok = "да" if m.get("can_confirm_sensitive") else "нет"
        policy_decision = m.get("policy_decision") or "unknown"
        return [
            f"### {item.title}",
            f"- Что произошло: {m.get('summary', item.ai_summary or item.title)}",
            f"- Что подтверждено: {confirmed or 'Недостаточно независимого подтверждения'}",
            f"- Кто что утверждает: {claims or m.get('claim_type', 'analysis')}",
            f"- Оценка доказательств: {m.get('evidence_strength', 'low')}",
            f"- Почему важно: {m.get('why_it_matters', item.ai_reason or '')}",
            f"- Уверенность: {conf}",
            f"- Политика источника: роль {self._role_label(m.get('source_role', 'unclassified'))}; решение {policy_decision}; независимое подтверждение: {independent}; чувствительные факты: {sensitive_ok}",
            f"- Источники: {m.get('source_name', item.source_type.value)} ({self._role_label(m.get('source_role', 'unclassified'))}), {date}, {item.url}",
        ] + ([f"- Поддерживающие источники: {self._supporting_sources_text(m)}"] if m.get("supporting_sources") else []) + [""]


def run_briefing_critic(markdown: str, items: List[ContentItem]) -> CriticResult:
    critical, minor, edits = [], [], []
    if "date unknown" in markdown:
        minor.append("some items have unknown publication dates")
    card_count = markdown.count("### ")
    expected_cards = len(set([i.id for i in items]))
    if not items:
        empty_markers = ("нет событий", "Прошло в итоговую сводку: 0", "no items passed")
        if not any(marker in markdown for marker in empty_markers):
            critical.append("empty summary does not explain that no items passed selection")
        if critical:
            edits.append("revise empty summary context")
        return CriticResult.model_validate({"pass": not critical, "critical_issues": critical, "minor_issues": minor, "required_edits": edits})
    if card_count < expected_cards:
        critical.append("missing cards detected")
    elif card_count > expected_cards:
        critical.append("duplicate cards detected")
    for it in items:
        role = str(it.metadata.get("source_role", "unclassified"))
        claim_type = _normalize_claim_type(it.metadata.get("claim_type"))
        confidence = _normalize_confidence(it.metadata.get("confidence"))
        sensitive = bool(it.metadata.get("sensitive_topic")) or bool(it.metadata.get("sensitive_topic_auto"))
        can_confirm_fact = bool(it.metadata.get("can_confirm_fact"))
        can_confirm_sensitive = bool(it.metadata.get("can_confirm_sensitive"))
        independent = bool(it.metadata.get("counts_as_independent_confirmation"))
        supporting_confirmations = _supporting_confirmation_sources(it, sensitive=sensitive)
        if role in {"blocked_as_fact_source", "unclassified"} and claim_type == "confirmed_fact":
            critical.append(f"source_role violation for {it.id}")
        if role in {"blocked_as_fact_source", "unclassified"} and confidence != "low":
            critical.append(f"invalid main-section source role for {it.id}")
        if role == "social_primary_statement_only" and claim_type == "confirmed_fact":
            critical.append(f"social confirmed_fact violation for {it.id}")
        if role == "russian_institutional_frame" and sensitive and claim_type == "confirmed_fact":
            critical.append(f"sensitive russian institutional claim not downgraded for {it.id}")
        if claim_type == "confirmed_fact" and (not can_confirm_fact or not independent):
            critical.append(f"confirmed_fact lacks source-policy authority for {it.id}")
        if sensitive and claim_type == "confirmed_fact" and (not can_confirm_sensitive or not independent or not supporting_confirmations):
            critical.append(f"sensitive confirmed_fact lacks independent sensitive confirmation for {it.id}")
        if confidence == "high" and not supporting_confirmations:
            critical.append(f"high confidence lacks independent supporting source for {it.id}")
        if not (it.metadata.get("source_url") or it.url):
            critical.append(f"missing source url for {it.id}")
        if not it.published_at and confidence != "low":
            critical.append(f"missing date must imply low confidence for {it.id}")
        if confidence == "low" and "## Спорные / слабоподтверждённые сообщения" not in markdown:
            critical.append("low confidence item shown outside disputed section")
    if critical:
        edits.append("revise sections and evidence labels")
    return CriticResult.model_validate({"pass": not critical, "critical_issues": critical, "minor_issues": minor, "required_edits": edits})
