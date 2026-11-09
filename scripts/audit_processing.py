"""Read-only replay of the *existing* first-stage news processing on a collector JSON.

The report records each filter decision, potential event-topic mistakes and
cross-source repeated headlines. It does not assign gold labels, merge news,
run an LLM, connect to PostgreSQL, or publish to MAX.

Example:
    python -m scripts.audit_processing collector_test_results/news_fixed_20261109_031438.json \
        --output collector_test_results/processing_audit_20261109.json \
        --candidates collector_test_results/processing_candidates_20261109.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import date
import json
from pathlib import Path
import re
from types import SimpleNamespace

from processing.topic_filter import classify_article
from processing.tagging_event_guard import nonindustrial_event_reason


def title_key(title: str) -> str:
    """A conservative repeated-headline signal, not semantic clustering."""
    return " ".join(re.findall(r"[a-zа-яё0-9]+", (title or "").casefold().replace("ё", "е")))


def audit_payload(
    payload: dict, *, as_of: date, max_age_days: int = 30,
) -> tuple[dict, dict]:
    """Return (audit, accepted news JSON) without modifying the input."""
    rows = payload.get("articles")
    if not isinstance(rows, list):
        raise ValueError("Ожидался JSON с массивом articles")

    repeated = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Все элементы articles должны быть объектами")
        key = title_key(str(row.get("title") or ""))
        if key:
            repeated[key].append(row)

    # A repeated headline is only a *candidate* for merging. Never silently
    # discard it: same headlines can describe independently updated articles.
    title_groups = {
        key: group for key, group in repeated.items()
        if len(group) > 1 and len({str(r.get("source") or "") for r in group}) > 1
    }
    repeated_urls = {
        str(r.get("url") or "")
        for group in title_groups.values() for r in group
    }

    decisions = []
    accepted = []
    reasons = Counter()
    guard_counts = Counter()
    unverified_count = 0
    for row in rows:
        decision = classify_article(
            SimpleNamespace(
                title=row.get("title"), text=row.get("text"),
                url=row.get("url"), published_at=row.get("published_at"),
            ),
            now=as_of,
            max_age_days=max_age_days,
        )
        title = str(row.get("title") or "")
        event_reason = nonindustrial_event_reason(title)
        unverified = row.get("unverified_tls") is True
        if unverified:
            unverified_count += 1
        if event_reason:
            guard_counts[event_reason] += 1
        reasons[decision.reason] += 1
        decisions.append({
            "source": row.get("source"),
            "title": title,
            "url": row.get("url"),
            "accepted_by_primary_filter": decision.accepted,
            "filter_reason": decision.reason,
            "event_guard_reason": event_reason,
            "possible_cross_source_duplicate": str(row.get("url") or "") in repeated_urls,
            "unverified_tls": unverified,
            "requires_source_verification": unverified,
        })
        if decision.accepted:
            # Preserve all rows in raw collection; filtering does not perform
            # semantic clustering or silently exclude the user's TLS sources.
            accepted.append(dict(row))

    report = {
        "read_only": True,
        "publishing_performed": False,
        "training_performed": False,
        "model_predictions_performed": False,
        "human_verified_count": 0,
        "as_of": as_of.isoformat(),
        "max_age_days": max_age_days,
        "summary": {
            "input_articles": len(rows),
            "primary_filter_accepted": len(accepted),
            "primary_filter_rejected": len(rows) - len(accepted),
            "by_filter_reason": dict(sorted(reasons.items())),
            "headline_event_guard_counts": dict(sorted(guard_counts.items())),
            "unverified_tls_articles": unverified_count,
            "possible_cross_source_duplicate_groups": len(title_groups),
            "possible_cross_source_duplicate_articles": len(repeated_urls),
        },
        "possible_cross_source_duplicates": [
            {
                "title": group[0].get("title"),
                "sources": sorted({str(r.get("source") or "") for r in group}),
                "urls": [r.get("url") for r in group],
            }
            for _key, group in sorted(title_groups.items())
        ],
        "decisions": decisions,
        "notes": [
            "Первичный фильтр использует существующий processing.topic_filter.",
            "Проверка типа события по заголовку только диагностическая.",
            "Совпадение заголовков не является доказательством дублирования.",
            "Непроверенный TLS помечается, но статья не исчезает из теста.",
            "Ни один прогноз модели или ручной эталон не создан.",
        ],
    }
    candidates = {
        "read_only": True,
        "filters_applied": True,
        "source_file_collected_at": payload.get("collected_at"),
        "as_of": as_of.isoformat(),
        "articles": accepted,
    }
    return report, candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Аудит первичной обработки без БД, MAX и локальных LLM"
    )
    parser.add_argument("news_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True,
                        help="JSON после первичного фильтра для сравнения SBERT/Qwen")
    parser.add_argument("--as-of", type=date.fromisoformat,
                        help="Дата оценки свежести YYYY-MM-DD (по умолчанию collected_at)")
    parser.add_argument("--max-age-days", type=int, default=30)
    args = parser.parse_args()
    if args.max_age_days < 0:
        parser.error("--max-age-days должен быть >= 0")
    if args.output.resolve() == args.candidates.resolve():
        parser.error("--output и --candidates должны быть разными файлами")
    if args.news_json.resolve() in (args.output.resolve(), args.candidates.resolve()):
        parser.error("Входной файл не может быть выходным")

    payload = json.loads(args.news_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        parser.error("Ожидался JSON-объект")
    try:
        as_of = args.as_of or date.fromisoformat(
            str(payload.get("collected_at") or "")[:10]
        )
    except ValueError:
        as_of = date.today()

    report, candidates = audit_payload(
        payload, as_of=as_of, max_age_days=args.max_age_days,
    )
    for path, content in ((args.output, report), (args.candidates, candidates)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(content, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Результат фильтра: {args.candidates}")
    print(f"Подробный аудит: {args.output}")


if __name__ == "__main__":
    main()
