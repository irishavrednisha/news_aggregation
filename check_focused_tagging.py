"""Read-only inspection of focused SBERT tags for news JSON files.

Supports raw collector diagnostics (reactivated_sources_live.json)
and the old check_news_collection.py {"articles": [...]} format.

Examples:
    python check_focused_tagging.py collector_test_results/reactivated_sources_live.json --limit 15
    python check_focused_tagging.py collector_test_results/news_fixed.json --limit 45

No database connection, status updates, topic filter or MAX publishing.
Tags in this report are only proposals, not automatic ground truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

from processing.tagger import score_tags_detailed, select_tags_by_score_gap
from processing.tagging_event_guard import nonindustrial_event_reason


def get_articles(payload: dict) -> list[dict]:
    if isinstance(payload.get("articles"), list):
        return payload["articles"]
    results = payload.get("results", {})
    if isinstance(results, dict):
        return [
            {"source": source, **example}
            for source, details in results.items()
            if isinstance(details, dict)
            for example in details.get("examples", [])
            if isinstance(example, dict)
        ]
    return []


def main():
    parser = argparse.ArgumentParser(
        description="Проверка соответствия тэгов текстам без БД и публикации",
    )
    parser.add_argument("input", type=Path, help="JSON с полными текстами статей")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--max-tags", type=int, default=2)
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument(
        "--summary-mode", choices=("textrank", "lead"), default="textrank",
        help="TextRank как в основном приложении или первые абзацы",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("collector_test_results/focused_tagging_audit.json"),
    )
    args = parser.parse_args()
    if args.limit < 1 or args.max_tags < 1:
        parser.error("--limit и --max-tags должны быть >= 1")

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    articles = get_articles(payload)
    if args.source:
        articles = [a for a in articles if a.get("source") == args.source]
    articles = articles[:args.limit]

    from check_ai_tagging import prepare_text_for_ai

    rows = []
    for article in articles:
        title = str(article.get("title") or "").strip()
        body = str(article.get("text") or "").strip()
        if not title or not body:
            continue

        summary, _lead = prepare_text_for_ai(
            title, body, args.summary_mode,
        )
        if not summary.strip():
            summary = body[:700]
        focus = title + "\n" + summary
        # Первые две строки — заголовок и summary; дальше идёт тело.
        scoring_text = "\n".join([title, summary, body[:2500]])
        ranked = score_tags_detailed(
            scoring_text, focus_text=focus,
        )
        raw_selected = select_tags_by_score_gap(
            ranked, min_score=args.min_score,
            max_tags=args.max_tags, max_gap_from_best=0.05,
        )
        event_reason = nonindustrial_event_reason(title)
        selected = [] if event_reason else raw_selected
        print(f"\n[{article.get('source') or 'Источник неизвестен'}] {title}")
        if event_reason:
            print(f"  Основное событие не промышленное: {event_reason}")
        print("  Теги: " + (", ".join(selected) if selected else "не определены"))
        for entry in ranked[:5]:
            print(
                f"  {entry['tag']}: {entry['final_score']:.3f}"
                f" (смысл заголовка+summary={entry['focus_similarity']:.3f},"
                f" смысл статьи={entry['body_similarity']:.3f},"
                f" совпадения={entry['focus_matches']}/{entry['body_matches']})"
            )
            if entry["focus_keywords"] or entry["body_keywords"]:
                print(
                    "    Основания: "
                    + ", ".join(
                        (entry["focus_keywords"][:4] + entry["body_keywords"][:3])
                    )
                )
        rows.append({
            "source": article.get("source"),
            "title": title,
            "url": article.get("url"),
            "summary_used": summary,
            "tags": selected,
            "raw_tags_before_event_guard": raw_selected,
            "event_guard_reason": event_reason,
            "ranked": ranked,
            "manual_tags": [],
        })

    report = {
        "read_only": True,
        "filtering_applied": False,
        "mode": "focused_sbert_title_summary_body",
        "summary_mode": args.summary_mode,
        "threshold": args.min_score,
        "max_tags": args.max_tags,
        "note": (
            "Одиночные новости: summary сгенерировано из статьи; "
            "у кластеров с несколькими статьями оно может отличаться. "
            "Точность нужно оценить по ручной разметке."
        ),
        "counts": {
            "articles": len(rows),
            "with_tags": sum(bool(r["tags"]) for r in rows),
            "blocked_by_event_guard": sum(bool(r["event_guard_reason"]) for r in rows),
            "tag_frequency": dict(Counter(
                tag for r in rows for tag in r["tags"]
            )),
        },
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\nОтчёт: {args.output}")


if __name__ == "__main__":
    main()
