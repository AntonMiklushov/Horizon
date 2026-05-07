from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.horizon_ext.personal import (
    EvidenceChecker,
    SourcePolicyClassifier,
    load_source_policy,
    prefilter_personal_candidates,
)
from src.models import ContentItem, RSSSourceConfig, SourceType, TelegramChannelConfig


ROOT = Path(__file__).resolve().parents[1]


def _item(url: str, source: SourceType = SourceType.RSS, author: str = "") -> ContentItem:
    return ContentItem(
        id=url,
        source_type=source,
        title=url,
        url=url,
        author=author,
        content="content",
        published_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )


def _science_sources() -> list[dict]:
    presets = json.loads((ROOT / "data" / "presets.json").read_text(encoding="utf-8"))
    [science] = [domain for domain in presets["domains"] if domain["id"] == "science"]
    return science["sources"]


def test_science_preset_expands_rss_preprint_and_social_discovery_layers() -> None:
    sources = _science_sources()
    rss_sources = [source for source in sources if source["type"] == "rss"]
    telegram_sources = [source for source in sources if source["type"] == "telegram"]
    rss_names = {source["config"]["name"] for source in rss_sources}

    assert {
        "Nature",
        "Science/AAAS",
        "PNAS",
        "Cell Press",
        "PLOS Biology",
        "eLife",
        "Quanta Magazine",
        "Science News",
        "Scientific American",
        "The Conversation Science",
        "Ars Technica Science",
        "Knowable Magazine",
        "NASA News Releases",
        "NASA Science",
        "ESA Space News",
        "CERN News",
        "NSF News",
        "NIH News Releases",
        "NOAA Ocean Service News",
        "CDC Media Releases",
        "WHO Disease Outbreak News",
        "arXiv Physics",
        "bioRxiv",
        "medRxiv",
        "ChemRxiv",
        "EurekAlert",
    } <= rss_names

    for source in rss_sources:
        RSSSourceConfig(**source["config"])

    preprints = [source for source in rss_sources if source["config"].get("category") == "science_preprint"]
    assert {"arXiv Physics", "arXiv Astrophysics", "arXiv Mathematics", "bioRxiv", "medRxiv", "ChemRxiv"} <= {
        source["config"]["name"] for source in preprints
    }
    assert all("not peer-reviewed" in source["description"] for source in preprints)

    [eurekalert] = [source for source in rss_sources if source["config"]["name"] == "EurekAlert"]
    assert eurekalert["config"]["category"] == "science_discovery"
    assert "source-finder" in eurekalert["tags"]

    assert {source["config"]["channel"] for source in telegram_sources} == {
        "nplusone",
        "postnauka",
        "biomolecula",
        "trvscience",
        "scienceandlife",
        "nsmag",
        "naukatv_ru",
        "indicator_news",
    }
    for source in telegram_sources:
        TelegramChannelConfig(**source["config"])
        assert "discovery-only" in source["tags"]


def test_science_source_policy_classifies_new_layers() -> None:
    policy, warning = load_source_policy(str(ROOT / "data" / "config.personal-news.example.json"))
    assert warning == ""
    classifier = SourcePolicyClassifier(policy)

    cases = [
        ("https://www.pnas.org/doi/10.1073/pnas.1", "science_primary_source", "PNAS", ""),
        ("https://www.cell.com/cell/fulltext/example", "science_primary_source", "Cell Press", ""),
        ("https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1", "science_primary_source", "PLOS", ""),
        ("https://elifesciences.org/articles/12345", "science_primary_source", "eLife", ""),
        ("https://science.nasa.gov/universe/example", "science_primary_source", "NASA", ""),
        ("https://www.esa.int/Science_Exploration/example", "science_primary_source", "ESA", ""),
        ("https://home.cern/news/news/physics/example", "science_primary_source", "CERN", ""),
        ("https://www.nsf.gov/news/example", "science_primary_source", "NSF", ""),
        ("https://www.nih.gov/news-events/news-releases/example", "science_primary_source", "NIH", ""),
        ("https://oceanservice.noaa.gov/news/example", "science_primary_source", "NOAA", ""),
        ("https://www.cdc.gov/media/releases/2026/example.html", "science_primary_source", "CDC", ""),
        ("https://www.who.int/emergencies/disease-outbreak-news/item/example", "science_primary_source", "WHO", ""),
        ("https://www.quantamagazine.org/example", "context_layer", "Quanta Magazine", ""),
        ("https://www.biorxiv.org/content/10.1101/2026.01.01.1v1", "science_preprint", "bioRxiv", "not peer-reviewed"),
        ("https://www.medrxiv.org/content/10.1101/2026.01.01.1v1", "science_preprint", "medRxiv", "not peer-reviewed"),
        ("https://chemrxiv.org/engage/chemrxiv/article-details/example", "science_preprint", "ChemRxiv", "not peer-reviewed"),
        (
            "https://www.eurekalert.org/news-releases/123456",
            "science_source_finder",
            "EurekAlert",
            "requires verification with paper, journal, or institution",
        ),
    ]

    for url, role, source_name, notes in cases:
        out = classifier.classify(_item(url))
        assert out["source_role"] == role
        assert out["source_name"] == source_name
        assert out["source_policy_notes"] == notes


def test_science_preprints_and_source_finders_are_downgraded_after_analysis() -> None:
    checker = EvidenceChecker(time_window_hours=24)

    preprint = _item("https://www.biorxiv.org/content/10.1101/2026.01.01.1v1")
    preprint.metadata.update({
        "source_role": "science_preprint",
        "claim_type": "confirmed_fact",
        "confidence": "high",
        "evidence_strength": "high",
        "source_policy_notes": "",
    })
    checker.audit_item(preprint)
    assert preprint.metadata["claim_type"] == "unverified_report"
    assert preprint.metadata["confidence"] == "low"
    assert preprint.metadata["evidence_strength"] == "low"
    assert preprint.metadata["source_policy_notes"] == "not peer-reviewed"

    source_finder = _item("https://www.eurekalert.org/news-releases/123456")
    source_finder.metadata.update({
        "source_role": "science_source_finder",
        "claim_type": "confirmed_fact",
        "confidence": "high",
        "evidence_strength": "high",
        "source_policy_notes": "",
    })
    checker.audit_item(source_finder)
    assert source_finder.metadata["claim_type"] == "unverified_report"
    assert source_finder.metadata["confidence"] == "low"
    assert source_finder.metadata["evidence_strength"] == "low"
    assert "requires verification" in source_finder.metadata["source_policy_notes"]


def test_social_candidates_remain_primary_statement_or_blocked() -> None:
    policy, _ = load_source_policy(str(ROOT / "data" / "config.personal-news.example.json"))
    classifier = SourcePolicyClassifier(policy)
    nasa_x = _item("https://x.com/NASAScience_/status/1", source=SourceType.TWITTER)
    random_x = _item("https://x.com/random_science/status/1", source=SourceType.TWITTER)
    telegram = _item("https://t.me/nplusone/1", source=SourceType.TELEGRAM)
    for item in [nasa_x, random_x, telegram]:
        item.metadata.update(classifier.classify(item))

    candidates, excluded = prefilter_personal_candidates([nasa_x, random_x, telegram], policy)

    assert candidates == [nasa_x]
    assert nasa_x.metadata["claim_type"] == "primary_statement"
    assert {entry["item"] for entry in excluded} == {random_x.title, telegram.title}
    assert telegram.metadata["source_role"] == "blocked_as_fact_source"
