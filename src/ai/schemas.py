"""Typed schemas for model-produced JSON payloads."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ALLOWED_PERSONAL_TOPICS = {
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
ALLOWED_CLAIM_TYPES = {
    "confirmed_fact",
    "official_statement",
    "party_claim",
    "analysis",
    "market_reaction",
    "unverified_report",
    "correction_or_update",
    "primary_statement",
}


class _ModelPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")


class AnalysisResult(_ModelPayload):
    score: float = Field(ge=0, le=10)
    reason: str = ""
    summary: str = ""
    tags: list[str] = Field(default_factory=list)


class AnalysisResultWithId(AnalysisResult):
    id: str


class AnalysisBatchResult(_ModelPayload):
    items: list[AnalysisResultWithId]


class PersonalAnalysisResult(_ModelPayload):
    importance: float = Field(ge=0, le=10)
    include: bool = True
    topic: str = "other"
    claim_type: str = "analysis"
    sensitive_topic: bool = False
    evidence_strength: Literal["low", "medium", "high"] = "low"
    confidence: Literal["low", "medium", "high"] = "low"
    summary: str = ""
    confirmed_details: list[str] = Field(default_factory=list)
    who_claims: list[str] = Field(default_factory=list)
    why_it_matters: str = ""
    source_policy_notes: str = ""
    requires_deep_review: bool = False
    noise_penalty: float = Field(default=0, ge=0)
    weak_evidence_penalty: float = Field(default=0, ge=0)
    reason: str = ""

    @field_validator("topic", mode="before")
    @classmethod
    def _normalize_topic(cls, value: Any) -> str:
        normalized = str(value or "other").strip().lower()
        return normalized if normalized in ALLOWED_PERSONAL_TOPICS else "other"

    @field_validator("claim_type", mode="before")
    @classmethod
    def _normalize_claim_type(cls, value: Any) -> str:
        normalized = str(value or "analysis").strip().lower()
        return normalized if normalized in ALLOWED_CLAIM_TYPES else "analysis"


class PersonalAnalysisResultWithId(PersonalAnalysisResult):
    id: str


class PersonalAnalysisBatchResult(_ModelPayload):
    items: list[PersonalAnalysisResultWithId]


class TopicDedupResult(_ModelPayload):
    duplicates: list[list[int]] = Field(default_factory=list)

    @field_validator("duplicates")
    @classmethod
    def _groups_must_have_ints(cls, groups: list[list[int]]) -> list[list[int]]:
        return [[int(i) for i in group] for group in groups if len(group) >= 2]


class EnrichmentResult(_ModelPayload):
    title_en: str = ""
    title_zh: str = ""
    whats_new_en: str = ""
    whats_new_zh: str = ""
    why_it_matters_en: str = ""
    why_it_matters_zh: str = ""
    key_details_en: str = ""
    key_details_zh: str = ""
    background_en: str = ""
    background_zh: str = ""
    community_discussion_en: str = ""
    community_discussion_zh: str = ""
    sources: list[str] = Field(default_factory=list)

    @field_validator("*", mode="before")
    @classmethod
    def _unwrap_text_objects(cls, value: Any) -> Any:
        if isinstance(value, dict) and "text" in value:
            return value["text"]
        return value
