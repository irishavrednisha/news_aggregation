"""Read-only end-to-end content audit on REAL articles, no DB writes or MAX.

Collect -> primary filter -> TextRank -> GroundedTagger -> quality signals
and duplicate-title hints. Findings are diagnostic, NOT automatic truth.

Examples:
    python scripts/audit_news_pipeline.py --source "ТАСС" --source "Интерфакс" --limit 3
    python scripts/audit_news_pipeline.py --source "MASHNEWS" --limit 5

Saves collector_test_results/pipeline_audit.json and displays key issues.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collectors.data_collector import collect_news
from processing.grounded_tagger import select_tags
from processing.summarizer import build_textrank_summary
from processing.topic_filter import classify_article
from scripts.check_sources import get_configured_sources


DEFAULT_SOURCES = ("Военное.рф", "Интерфакс", "РИА Новости", "ТАСС", "MASHNEWS")
DEFAULT_OUTPUT = ROOT / "collector_test_results" / "pipeline_audit.json"


def _title_key(title: str) -> str:
    text = (title or "").lower().replace("ё", "е")
    return re.sub(r"[^а-яa-z0-9]+", " ", text).strip()


def _time(value) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else str(value) if value else None


def _summary_warnings(text: str, summary: str) -> list[str]:
    warnings = []
    if not summary:
        warnings.append("empty_summary")
    elif len(summary) > 800:
        warnings.append("summary_longer_than_limit")
    if summary:
        # Flag-only. Numbers in different formats, paraphrases and dates
        # require a human review and must not be auto-rejected.
        original_numbers = set(re.findall(r"\d[\d,.]*", text))
        summary_numbers = set(re.findall(r"\d[\d,.]*", summary))
        if summary_numbers - original_numbers:
            warnings.append("check_summary_numbers")
    return warnings


def audit_article(article) -> dict:
    """Inspect without mutating source article or any database object."""
    decision = classify_article(article)
    errors: list[str] = []
    try:
        summary = build_textrank_summary(
            article.text or "", title=article.title,
            sentence_count=4, max_chars=800,
        )
    except Exception as exc:
        errors.append("summary_error:" + type(exc).__name__)
        summary = ""
    try:
        tagging = select_tags(article.title, summary)
        tags = list(tagging.tags)
        tag_outcome = tagging.outcome
        tag_reason = tagging.reason
        evidence = list(tagging.evidence)
    except Exception as exc:
        errors.append("tagger_error:" + type(exc).__name__)
        tags, tag_outcome, tag_reason, evidence = [], "error", "", []

    warnings = _summary_warnings(article.text or "", summary)
    if decision.accepted and tag_outcome != "tagged":
        warnings.append("primary_filter_accepted_but_tagger_not_tagged")
    elif not decision.accepted and tag_outcome == "tagged":
        warnings.append("primary_filter_rejected_but_tagger_tagged")
    if getattr(article, "text_source", "") != "html":
        warnings.append("full_html_not_obtained")
    if not getattr(article, "published_at", None):
        warnings.append("missing_publication_date")
    if len(article.text or "") < 250:
        warnings.append("short_body")
    warnings.extend(errors)

    return {
        "source": article.source,
        "title": article.title,
        "url": article.url,
        "published_at": _time(getattr(article, "published_at", None)),
        "text_source": getattr(article, "text_source", None),
        "body_chars": len(article.text or ""),
        "body_preview": (article.text or "")[:500],
        "filter": {"accepted": decision.accepted, "reason": decision.reason},
        "summary": summary,
        "tags": tags,
        "tagger": {"outcome": tag_outcome, "reason": tag_reason, "evidence": evidence},
        "image_candidates": [
            {"url": candidate.url, "origin": candidate.origin}
            for candidate in (getattr(article, "image_candidates", None) or [])[:5]
        ],
        "warnings": warnings,
    }


def build_report(articles, diagnostics: dict, sources: list[str]) -> dict:
    rows = [audit_article(article) for article in articles]
    normalized: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        normalized[_title_key(row["title"])].append(idx)
    # Exact-title duplicates across different URLs (e.g. Mil.Press mirrors).
    duplicates = []
    for indices in normalized.values():
        urls = {rows[i]["url"] for i in indices}
        if len(urls) <= 1:
            continue
        duplicates.append([
            {"source": rows[i]["source"], "title": rows[i]["title"],
             "url": rows[i]["url"]}
            for i in indices
        ])
        for i in indices:
            rows[i]["warnings"].append("duplicate_title_different_url")

    return {
        "read_only": True,
        "db_modified": False,
        "max_publishing_performed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources_requested": sources,
        "collected_total": len(rows),
        "filter_accepted": sum(bool(r["filter"]["accepted"]) for r in rows),
        "tagger_tagged": sum(r["tagger"]["outcome"] == "tagged" for r in rows),
        "needs_review": sum(bool(r["warnings"]) for r in rows),
        "summary_empty": sum(not r["summary"] for r in rows),
        "repeated_titles": duplicates,
        "source_diagnostics": diagnostics,
        "articles": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit real news: text/filter/TextRank/Grounded tags, read-only."
    )
    parser.add_argument("--source", action="append", default=[],
                        help="Repeat to select sources; defaults to five representative sites.")
    parser.add_argument("--limit", type=int, default=3, help="Articles per source (1-10).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit not in range(1, 11):
        parser.error("--limit must be between 1 and 10")
    if args.output.suffix.lower() != ".json":
        parser.error("--output needs a .json filename")

    configured = get_configured_sources()
    names = list(dict.fromkeys(args.source or DEFAULT_SOURCES))
    unknown = [name for name in names if name not in configured]
    if unknown:
        parser.error("Unknown source(s): " + ", ".join(unknown))

    diagnostics = {}
    all_articles = []
    for name in names:
        source = configured[name]
        try:
            sample = collect_news(
                [source], limit_per_source=args.limit,
                session=None, diagnostics=diagnostics,
                apply_filters=False,
            )
            all_articles.extend(sample)
        except Exception as exc:
            diagnostics.setdefault(name, {})["audit_error"] = (
                type(exc).__name__ + ": " + str(exc)[:300]
            )

    report = build_report(all_articles, diagnostics, names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print("\n=== АУДИТ МОДУЛЕЙ (БД и MAX не изменялись) ===")
    print(f"Источников: {len(names)}; статей: {report['collected_total']}")
    print(f"Первичный фильтр оставил: {report['filter_accepted']}")
    print(f"GroundedTagger определил отрасль: {report['tagger_tagged']}")
    print(f"Пустых сводок: {report['summary_empty']}")
    print(f"Одинаковых заголовков по разным URL: {len(report['repeated_titles'])} групп")
    for row in report["articles"]:
        status = "PASS" if row["filter"]["accepted"] else "SKIP"
        labels = ", ".join(row["tags"]) or "не определены"
        flags = ", ".join(row["warnings"]) or "—"
        print(f"[{status}] {row['source']}: {row['title']}")
        print(f"  Фильтр: {row['filter']['reason']}; Теги: {labels} ({row['tagger']['outcome']})")
        print(f"  Замечания: {flags}")
    print(f"Полный JSON с текстовыми примерами и сводками: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
