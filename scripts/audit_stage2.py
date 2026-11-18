"""Stage 2 read-only audit: real RSS/HTML articles and independently curated gold cases.

Examples (no MAX/Telegram and no database writes):
 python scripts/audit_stage2.py --live --per-source 15 --target 250
 python scripts/audit_stage2.py --input collector_test_results/raw.json --labels manual_labels.json
 python scripts/audit_stage2.py --live --from-db --per-source 18 --target 250 \
     --exclude-urls training/previously_used_urls.json --require-coverage

Labels are EXTERNAL judgments. Unlabeled real articles are never counted as
true/false positives or negatives. A 250-item sample is not 250 gold labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from processing.exact_duplicates import ExactDuplicateIndex, normalize_url
from processing.news_intake import classify_intake

DEFAULT_OUT = ROOT / "collector_test_results" / "stage2_audit.json"
FIXTURES = ROOT / "tests" / "fixtures" / "stage2_gold_cases.json"
GOLD_BODY = (
    "На российском заводе инженеры внедрили цифровую систему контроля качества "
    "и испытали новые технологии для обработки металлов. Предприятие модернизировало "
    "промышленное оборудование, увеличило скорость обработки сырья, открыло участок "
    "производства деталей для двигателей и провело испытания экспериментального "
    "оборудования. Измерения подтвердили производительность комплекса. "
    "Научный коллектив подготовил техническую документацию для дальнейшей "
    "эксплуатации и передал результаты специалистам. "
)
GOLD_BOILER = "Политика конфиденциальности. Все права защищены. " * 12


def _article(row):
    if not isinstance(row, dict):
        raise TypeError("Article must be a JSON object")
    return SimpleNamespace(
        source=row.get("source", ""),
        title=row.get("title", ""),
        url=row.get("url", ""),
        text=row.get("text") or row.get("body") or "",
        text_source=row.get("text_source") or "html",
        published_at=row.get("published_at"),
    )


def _gold_article(row):
    kind = row.get("body_kind", "full")
    text = {
        "full": GOLD_BODY, "short": "На заводе испытали станок.",
        "rss": GOLD_BODY, "empty": "", "boiler": GOLD_BOILER,
    }[kind]
    return _article({
        "source": "GoldControl", "title": row["title"], "url": row["url"],
        "text": text, "published_at": row["published_at"],
        "text_source": "rss" if kind == "rss" else "html",
    })


def _json_dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")


def _load_rows(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("articles", "news", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        # Allow deterministic replay of the stage2_real_250.json report itself
        # without fetching any sites again or touching the local database.
        if isinstance(payload.get("real_sample"), dict):
            nested = payload["real_sample"].get("articles")
            if isinstance(nested, list):
                return nested
    raise ValueError("Expected article array or dict containing article records")


def _labels(path):
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("labels", []) if isinstance(data, dict) else data
    return {normalize_url(item["url"]): item for item in items if item.get("url")}


def _gold_report(*, today):
    cases = json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]
    mismatches = []
    counts = Counter()
    examples = []
    for row in cases:
        prediction = classify_intake(_gold_article(row), now=today)
        counts["control_total"] += 1
        counts["correct_status"] += prediction.status == row["expected_status"]
        counts["correct_reason"] += prediction.reason == row["expected_reason"]
        if row["expected_status"] == "rejected" and prediction.status == "accepted":
            counts["false_junk_pass"] += 1
        if row["expected_status"] == "accepted" and prediction.status == "rejected":
            counts["false_useful_reject"] += 1
        if prediction.status != row["expected_status"] or prediction.reason != row["expected_reason"]:
            mismatches.append({
                "case": row["id"], "title": row["title"],
                "expected": row["expected_status"], "predicted": prediction.status,
                "expected_reason": row["expected_reason"], "reason": prediction.reason,
            })
    return {"metrics": dict(counts), "mismatches": mismatches}


def audit_real(articles, *, today=None, expected_labels=None, excluded=None):
    """Pure replay over independent article objects, no SQLAlchemy or writes."""
    expected_labels = expected_labels or {}
    excluded = excluded or set()
    index = ExactDuplicateIndex()
    raw_index = ExactDuplicateIndex()
    rows = []
    article_by_url = {}
    raw_by_id = {}
    counts = Counter()
    next_id = 1
    raw_next_id = 1
    for article in articles:
        canonical = normalize_url(article.url)
        if canonical in excluded:
            counts["excluded_previous_url"] += 1
            continue
        decision = classify_intake(article, now=today)
        match = index.find(article) if decision.status == "accepted" else None
        raw_match = raw_index.find(article) if canonical and len(
            article.text or ""
        ) >= 250 else None
        duplicate = match or (raw_match if decision.status != "accepted" else None)
        status = "rejected" if duplicate is not None else decision.status
        reason = duplicate.reason if duplicate is not None else decision.reason
        target = next(
            (u for u, n in article_by_url.items() if match is not None and n == match.news_id),
            None,
        )
        if target is None and duplicate is not None and raw_match is not None:
            target = raw_by_id.get(raw_match.news_id)
        row = {
            "source": article.source,
            "title": article.title,
            "url": article.url,
            "canonical_url": canonical,
            "published_at": str(article.published_at) if article.published_at else None,
            "text_source": getattr(article, "text_source", None),
            "text": article.text,
            "body_chars": len(article.text or ""),
            "status": status, "reason": reason,
            "stage": "exact_duplicate" if duplicate else decision.stage,
            "duplicate_of_url": target,
        }
        rows.append(row)
        counts["evaluated_total"] += 1
        counts[status] += 1
        counts["reason:" + reason] += 1
        if duplicate is not None:
            counts["exact_duplicates"] += 1
        if status == "accepted":
            index.add(SimpleNamespace(id=next_id, **vars(article)))
            article_by_url[canonical] = next_id
            next_id += 1
        if duplicate is None and canonical and len(article.text or "") >= 250:
            raw_index.add(SimpleNamespace(id=raw_next_id, **vars(article)))
            raw_by_id[raw_next_id] = canonical
            raw_next_id += 1

    # Replaying the same input against the accepted pool must create zero news.
    index2 = ExactDuplicateIndex()
    for row in rows:
        if row["status"] == "accepted":
            article = _article(row)
            article.id = len(index2.urls) + 1
            index2.add(article)
    replay_new_news = 0
    for row in rows:
        article = _article(row)
        decision = classify_intake(article, now=today)
        if decision.status == "accepted" and index2.find(article) is None:
            replay_new_news += 1

    labeled = []
    label_counts = Counter()
    for row in rows:
        gold = expected_labels.get(row["canonical_url"])
        if not gold:
            continue
        expected = gold.get("expected_status")
        # Two output statuses only. Older manual gold may identify copies
        # with expected_status=duplicate; retain compatibility for evaluation.
        if expected not in ("accepted", "rejected", "duplicate"):
            continue
        expected_duplicate = (
            expected == "duplicate" or
            str(gold.get("expected_reason") or "").startswith("duplicate_") or
            gold.get("expected_duplicate") is True
        )
        expected_binary = "rejected" if expected_duplicate else expected
        label_counts["labeled_real_count"] += 1
        predicted = row["status"]
        actual_duplicate = row["stage"] == "exact_duplicate"
        if expected_binary == "rejected" and predicted == "accepted":
            label_counts["false_junk_pass"] += 1
        if expected_binary == "accepted" and predicted == "rejected":
            label_counts["false_useful_reject"] += 1
        if expected_duplicate:
            label_counts["duplicate_tp" if actual_duplicate else "duplicate_fn"] += 1
        elif actual_duplicate:
            label_counts["duplicate_fp"] += 1
        if expected_binary != predicted or expected_duplicate != actual_duplicate:
            labeled.append({
                "title": row["title"], "url": row["url"],
                "expected": expected_binary, "predicted": predicted,
                "expected_duplicate": expected_duplicate,
                "predicted_duplicate": actual_duplicate,
                "reason": row["reason"],
            })
    return {
        "counts": dict(counts),
        "repeatability": {"replay_new_news": replay_new_news},
        "duplicate_audit_scope": "all_raw_statuses_with_eligible_text",
        "status_contract": ["accepted", "rejected"],
        "gold_real_metrics": dict(label_counts) if label_counts else None,
        "labeled_mistakes": labeled,
        "articles": rows,
    }


def _live_articles(*, names, per_source, from_db):
    from collectors.data_collector import collect_news
    if from_db:
        from database.db import get_session
        from database.models import Source
        session = get_session()
        try:
            sources = [
                SimpleNamespace(name=s.name, url=s.url, rss_url=s.rss_url,
                                source_type=s.source_type, is_active=s.is_active)
                for s in session.query(Source).filter(Source.is_active.is_(True)).all()
            ]
        finally:
            session.close()
    else:
        from scripts.check_sources import get_configured_sources
        sources = list(get_configured_sources().values())
    if names:
        sources = [s for s in sources if s.name in names]
    diagnostics = {}
    articles = []
    for source in sources:
        try:
            articles.extend(collect_news(
                [source], limit_per_source=per_source, session=None,
                apply_filters=False, diagnostics=diagnostics,
            ))
        except Exception as exc:
            diagnostics[source.name] = {
                "status": "audit_error",
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
    return articles, diagnostics, [s.name for s in sources]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--live", action="store_true")
    source.add_argument("--input", type=Path)
    parser.add_argument("--from-db", action="store_true",
                        help="Read active source settings from DB; no DB writes.")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--per-source", type=int, default=15)
    parser.add_argument("--target", type=int, default=250)
    parser.add_argument("--require-coverage", action="store_true")
    parser.add_argument("--labels", type=Path,
                        help="Independent manual judgments for real articles.")
    parser.add_argument("--exclude-urls", type=Path,
                        help="JSON URL list previously used in training or evaluations.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--date", type=date.fromisoformat, default=None,
                        help="Frozen comparison date YYYY-MM-DD for reproducibility.")
    args = parser.parse_args(argv)
    if not (200 <= args.target <= 300):
        parser.error("--target must be 200..300")
    if not (1 <= args.per_source <= 40):
        parser.error("--per-source must be 1..40")
    if args.from_db and not args.live:
        parser.error("--from-db only applies to --live")
    if args.live:
        articles, diagnostics, sources = _live_articles(
            names=args.source, per_source=args.per_source, from_db=args.from_db
        )
        provenance = "live_rss_html"
    else:
        records = _load_rows(args.input)
        articles = [_article(r) for r in records]
        diagnostics, sources = {}, sorted({x.source for x in articles})
        provenance = str(args.input)

    # Stable round-robin by source prevents overfilling one publisher's feed.
    buckets = {}
    for a in articles:
        buckets.setdefault(a.source, []).append(a)
    selected = []
    while len(selected) < args.target and any(buckets.values()):
        for name in sorted(buckets):
            if buckets[name] and len(selected) < args.target:
                selected.append(buckets[name].pop(0))
    exclusions = set()
    if args.exclude_urls:
        exclusion_rows = json.loads(args.exclude_urls.read_text(encoding="utf-8"))
        if isinstance(exclusion_rows, dict):
            exclusion_rows = exclusion_rows.get("urls", [])
        exclusions = {
            normalize_url(x if isinstance(x, str) else x.get("url", ""))
            for x in exclusion_rows
        }
    today = args.date or datetime.now(timezone.utc).date()
    real = audit_real(selected, today=today, expected_labels=_labels(args.labels),
                      excluded=exclusions)
    # Differences against an earlier *model* decision are not ground-truth
    # errors; they measure how much the new triage behavior changed.
    old_by_url = {}
    if args.input:
        for old in _load_rows(args.input):
            if isinstance(old, dict) and old.get("status"):
                old_by_url.setdefault(normalize_url(old.get("url", "")),
                                      (old["status"], old.get("reason")))
    changes = []
    transitions = Counter()
    for row in real["articles"]:
        old = old_by_url.get(row["canonical_url"])
        if not old:
            continue
        transitions[old[0] + " -> " + row["status"]] += 1
        if old[0] != row["status"] or old[1] != row["reason"]:
            changes.append({
                "title": row["title"], "url": row["url"],
                "before_status": old[0], "before_reason": old[1],
                "after_status": row["status"], "after_reason": row["reason"],
            })
    report = {
        "stage": 2, "read_only": True, "db_written": False,
        "max_published": False, "telegram_started": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "audit_date": today.isoformat(),
        "provenance": provenance, "requested_target": args.target,
        "collected_before_sampling": len(articles),
        "selected_before_exclusions": len(selected),
        "sample_complete": real["counts"].get("evaluated_total", 0) >= 200,
        "independence_from_prior_samples": (
            "checked_against_supplied_exclusions" if args.exclude_urls else "not_verified"
        ),
        "sources": sources, "source_diagnostics": diagnostics,
        "controlled_gold": _gold_report(today=date(2026, 11, 18)),
        "real_sample": real,
        "comparison_with_previous_model": {
            "transitions": dict(transitions),
            "changed": changes,
            "note": "Changes are not independent accuracy labels.",
        },
        "caveat": (
            "Unlabeled real items are not error-rate denominators. "
            "Manual labels are needed to measure real false positives and negatives."
        ),
    }
    _json_dump(args.output, report)
    print(json.dumps({
        "output": str(args.output), "collected": len(articles),
        "real_evaluated": real["counts"].get("evaluated_total", 0),
        "real_sample_complete": report["sample_complete"],
        "real_labeled_count": (real["gold_real_metrics"] or {}).get("labeled_real_count", 0),
        "controlled": report["controlled_gold"]["metrics"],
        "replay_new_news": real["repeatability"]["replay_new_news"],
    }, ensure_ascii=False, indent=2))
    if args.require_coverage and not report["sample_complete"]:
        return 2
    if report["controlled_gold"]["mismatches"] or real["repeatability"]["replay_new_news"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
