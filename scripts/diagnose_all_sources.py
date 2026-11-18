"""One-command, read-only diagnostics of ALL sources currently in PostgreSQL.

Example (Windows, no ML): .\\.venv\\Scripts\\python.exe scripts\\diagnose_all_sources.py --limit 3
Reads sources, never seeds/updates tables, never publishes or saves news.
HTML extraction probes are bounded by the collector's per-source limits.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_sources():
    from database.db import get_session
    from database.models import Source

    session = get_session()
    try:
        # Keep inactive rows too: the database, not seed_sources, is authoritative.
        rows = session.query(Source).order_by(Source.id.asc()).all()
        return [
            SimpleNamespace(id=s.id, name=s.name, url=s.url, rss_url=s.rss_url,
                            source_type=s.source_type, is_active=s.is_active)
            for s in rows
        ]
    finally:
        session.close()


def load_configured_sources():
    """CI fallback only: seed definitions, NOT current local database records."""
    from scripts.check_sources import get_configured_sources

    return [SimpleNamespace(id=None, **vars(row))
            for row in get_configured_sources().values()]


def article_metrics(article, include_text=False):
    title = (article.title or "").strip()
    text = (article.text or "").strip()
    published = article.published_at
    if published is not None:
        try:
            dated = published.date() if hasattr(published, "date") else published
            from datetime import date
            date_ok = date(1995, 1, 1) <= dated <= (datetime.now(timezone.utc) + timedelta(days=2)).date()
        except (TypeError, ValueError, AttributeError):
            date_ok = False
    else:
        date_ok = False
    title_ok = len(title) >= 15 and title.lower() not in (
        "access denied", "403 forbidden", "just a moment", "service unavailable",
    )
    from collectors.article_quality import is_error_page_text
    placeholder = is_error_page_text(text)
    # "Complete" is a heuristic; true completeness needs comparison with original page.
    full_text = article.text_source == "html" and len(text) >= 250 and not placeholder
    result = {
        "url": article.url,
        "title": title,
        "title_valid": title_ok,
        "published_at": published.isoformat() if hasattr(published, "isoformat") else None,
        "date_valid": date_ok,
        "text_chars": len(text),
        "text_method": article.text_source,
        "full_text_heuristic": full_text,
        "image_candidates": len(article.image_candidates or []),
        "image_urls": [x.url for x in (article.image_candidates or [])[:4]],
        "issues": (
            ([] if title_ok else ["invalid_title"]) +
            ([] if date_ok else ["missing_or_implausible_date"]) +
            ([] if full_text else ["missing_or_short_full_html_text"]) +
            (["error_or_access_page"] if placeholder else [])
        ),
    }
    if include_text:
        result["text"] = text
    return result


def inspect_source(source, limit, allow_unverified_tls=False, include_text=False):
    from collectors.data_collector import collect_news

    diagnostics = {}
    try:
        articles = collect_news(
            [source], limit_per_source=limit, session=None, diagnostics=diagnostics,
            apply_filters=False, allow_unverified_tls=allow_unverified_tls,
        )
        failures = []
    except Exception as exc:
        articles = []
        failures = [f"{type(exc).__name__}: {exc}"]
    stat = diagnostics.get(source.name, {})
    metrics = [article_metrics(a, include_text=include_text) for a in articles]
    result = {
        "source_id": source.id,
        "name": source.name,
        "url": source.url,
        "rss_url": source.rss_url,
        "configured_active": bool(source.is_active),
        "status": stat.get("status", "exception"),
        "available": (
            bool(stat.get("candidates"))
            and stat.get("status") not in ("fetch_failed", "collector_error")
        ) or any(x["text_method"] == "html" for x in metrics),
        "links_found": stat.get("candidates", 0),
        "articles_extracted": sum(
            x["text_method"] == "html" and x["text_chars"] >= 80
            for x in metrics
        ),
        "rss_previews_only": sum(x["text_method"] == "rss" for x in metrics),
        "full_html_texts_250_plus": sum(bool(x["full_text_heuristic"]) for x in metrics),
        "valid_titles": sum(bool(x["title_valid"]) for x in metrics),
        "valid_dates": sum(bool(x["date_valid"]) for x in metrics),
        "articles_with_image_candidates": sum(bool(x["image_candidates"]) for x in metrics),
        "short_or_empty": stat.get("short_or_empty", 0),
        "error_page_rejections": stat.get("error_page_rejections", 0),
        "rss_preview_only_events": stat.get("rss_preview_only", 0),
        "errors": failures + stat.get("errors", []),
        "fetch_failures": stat.get("fetch_failures", []),
        "article_probes": stat.get("quality_probes", []),
        "articles": metrics,
        "tls_unverified_urls": stat.get("unverified_tls_urls", []),
    }
    if not metrics and not result["errors"]:
        result["errors"].append(
            "no_links" if not result["links_found"] else "no_usable_article_text"
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only audit of every PostgreSQL source")
    parser.add_argument("--limit", type=int, default=3, help="Up to N articles per source")
    parser.add_argument("--source", action="append", help="Filter by exact source name")
    parser.add_argument("--output", default="collector_test_results/all_sources_diagnostics.json")
    parser.add_argument("--include-text", action="store_true", help="Include extracted text in JSON")
    parser.add_argument("--allow-unverified-tls", action="store_true",
                        help="Explicit diagnostic ONLY; never records news or publishes")
    parser.add_argument("--strict", action="store_true", help="Exit 1 if any source has no valid article")
    parser.add_argument("--from-config", action="store_true",
                        help="CI-only source definitions from seed_sources, NOT PostgreSQL")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel independent source probes, 1-4 (default 1)")
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= 10:
        parser.error("--limit must be from 1 to 10")
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be from 1 to 4")
    try:
        sources = load_configured_sources() if args.from_config else load_sources()
    except Exception as exc:
        print(f"Failed to load sources: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.source:
        names = set(args.source)
        unknown = names - {s.name for s in sources}
        if unknown:
            parser.error("Not found in PostgreSQL: " + ", ".join(sorted(unknown)))
        sources = [s for s in sources if s.name in names]
    if not sources:
        print("No sources in PostgreSQL. Nothing was changed.", file=sys.stderr)
        return 2

    if args.workers == 1:
        results = [inspect_source(s, args.limit, args.allow_unverified_tls, args.include_text)
                   for s in sources]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(
                lambda source: inspect_source(
                    source, args.limit, args.allow_unverified_tls, args.include_text,
                ), sources,
            ))
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_origin": "seed_configuration_not_database" if args.from_config else "postgresql_sources_table",
        "source_count_from_database": None if args.from_config else len(sources),
        "source_count_checked": len(sources),
        "read_only": True,
        "max_publishing_enabled": False,
        "limit_per_source": args.limit,
        "text_completeness": "heuristic (HTML text >=250 chars), not manual verification",
        "results": results,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = out.with_suffix(out.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temp.replace(out)
    for r in results:
        print(f"{r['name']}: {r['status']} | links={r['links_found']}, "
              f"extracted={r['articles_extracted']}, full={r['full_html_texts_250_plus']}, "
              f"dated={r['valid_dates']}, images={r['articles_with_image_candidates']}")
        if r["errors"]:
            print("    " + "; ".join(r["errors"][:2]))
    good = sum(r["articles_extracted"] > 0 for r in results)
    origin = "configured URLs (NOT PostgreSQL)" if args.from_config else "PostgreSQL rows"
    print(f"Checked {len(results)} {origin}; {good} yielded HTML articles. JSON: {out}")
    return int(args.strict and good < len(results))


if __name__ == "__main__":
    raise SystemExit(main())
