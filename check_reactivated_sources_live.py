"""Read-only live smoke test of the eight reactivated news sources.

Usage:
    python check_reactivated_sources_live.py --limit 2

Does not connect to PostgreSQL, change source flags or publish to MAX.
Only loads public web pages and writes a local JSON diagnostic report.
No thematic, foreign-story or editorial filters are applied in this mode.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from collectors.data_collector import collect_news
from database.repository import seed_sources


NAMES = (
    "Газпром", "Военное.рф", "ФлотПром", "Отраслевое.рф",
    "ФлотКом", "РИА Новости", "ТАСС", "Интерфакс",
)


def configured_sources():
    """Read seed_sources configuration without touching the database."""
    definitions = []
    with patch(
        "database.repository.get_or_create_source",
        side_effect=lambda **kwargs: definitions.append(kwargs),
    ):
        seed_sources(None)
    source_by_name = {source["name"]: source for source in definitions}
    return [SimpleNamespace(**source_by_name[name]) for name in NAMES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument(
        "--source", action="append", choices=NAMES,
        help="Проверить только указанный источник (можно повторить)",
    )
    parser.add_argument(
        "--output", default="collector_test_results/reactivated_sources_live.json"
    )
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 5:
        parser.error("--limit должен быть от 1 до 5")

    sources = configured_sources()
    if args.source:
        requested = set(args.source)
        sources = [source for source in sources if source.name in requested]
    diagnostics = {}
    articles = collect_news(
        sources=sources,
        limit_per_source=args.limit,
        session=None,
        diagnostics=diagnostics,
        allow_unverified_tls=False,
        apply_filters=False,
    )
    summary = {}
    for source in sources:
        examples = []
        for article in articles:
            if article.source != source.name:
                continue
            examples.append({
                "title": article.title,
                "url": article.url,
                "published_at": (
                    article.published_at.isoformat()
                    if article.published_at else None
                ),
                "text_length": len(article.text or ""),
                "text_source": article.text_source,
                "has_full_html_text": article.text_source == "html",
                "text": article.text or "",
                "text_sample": (article.text or "")[:350],
            })
        summary[source.name] = {
            "configured_active": source.is_active,
            "candidates": diagnostics.get(source.name, {}).get("candidates", 0),
            "collected": len(examples),
            "full_html_texts": sum(e["has_full_html_text"] for e in examples),
            "dated": sum(e["published_at"] is not None for e in examples),
            "status": diagnostics.get(source.name, {}).get("status", "unknown"),
            "short_or_empty": diagnostics.get(source.name, {}).get("short_or_empty", 0),
            "article_probes": diagnostics.get(source.name, {}).get("article_probes", []),
            "empty_article_examples": diagnostics.get(source.name, {}).get("empty_article_examples", []),
            "errors": diagnostics.get(source.name, {}).get("errors", [])[:5],
            "fetch_failures": diagnostics.get(source.name, {}).get(
                "fetch_failures", []
            )[:5],
            "examples": examples,
        }

    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "filtering_applied": False,
        "full_text_included": True,
        "limit_per_source": args.limit,
        "results": summary,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 85)
    print("READ-ONLY LIVE SOURCE CHECK (no database writes, no MAX publishing)")
    for name, stat in summary.items():
        print(
            f"{name}: status={stat['status']}, candidates={stat['candidates']}, "
            f"collected={stat['collected']}, full_html_texts={stat['full_html_texts']}, "
            f"dated={stat['dated']}"
        )
        for example in stat["examples"]:
            print(f"  - {example['title']} | {example['url']}")
            print(
                f"    chars={example['text_length']}, "
                f"published_at={example['published_at']}, "
                f"has_full_html_text={example['has_full_html_text']}"
            )
        for probe in stat["article_probes"][:2]:
            print(f"    PAGE PROBE: {probe}")
        for empty in stat["empty_article_examples"][:2]:
            print(f"    NO TEXT: {empty}")
        for error in stat["errors"][:2]:
            print(f"    ERROR: {error}")
    print(f"Report: {destination}")
    print("=" * 85)

    # External sites may reject GitHub-hosted IPs. Keep this report
    # diagnostic; availability must not make the offline CI suite flaky.


if __name__ == "__main__":
    main()
