"""Безопасный аудит AI-теггера на сохранённых статьях.

По умолчанию для одиночной новости вычисляет тот же TextRank-summary,
что и рабочий summarize_cluster(): title + text[:2500], 4 предложения,
максимум 800 символов. В производстве кластеры могут содержать несколько
новостей — их summary может отличаться. Режим --summary-mode lead
воспроизводит прежний эксперимент с первыми абзацами.

Не обращается к базе PostgreSQL и не публикует в MAX.

Пример:
  python check_ai_tagging.py collector_test_results/news_fixed_20261106_223822.json --limit 45 --output ai_real_summary.json
  python check_ai_tagging.py collector_test_results/news_fixed_20261106_223822.json --primary-filter --as-of 2026-11-06 --output ai_accepted_only.json
"""
import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace

from processing.ai_tagger import (
    DEFAULT_MODEL, load_classifier, predict_tags,
)


def prepare_text_for_ai(title: str, text: str, summary_mode: str):
    """Input for the two available audit modes, with no DB dependencies."""
    title = title or ""
    text = text or ""
    if summary_mode == "lead":
        # Для воспроизведения предыдущих результатов v2_45.
        return text[:780], text[780:1110]
    if summary_mode != "textrank":
        raise ValueError("summary_mode должен быть textrank или lead")

    # Зависимости networkx/sklearn загружаем только при TextRank.
    from processing.summarizer import build_textrank_summary
    # get_cluster_text() для одиночного кластера добавляет заголовок
    # и максимум 2500 символов полного текста этой новости.
    cluster_text = title + "\n" + text[:2500]
    summary = build_textrank_summary(
        text=cluster_text, title=title, sentence_count=4, max_chars=800,
    )
    return summary, text[:330]


def selected_articles(articles, source=None, primary_filter=False, as_of=None):
    """Keep original order, return (accepted, rejected with reasons)."""
    if source:
        articles = [a for a in articles if a.get("source") == source]
    if not primary_filter:
        return articles, []
    from processing.topic_filter import classify_article
    accepted, rejected = [], []
    for a in articles:
        decision = classify_article(SimpleNamespace(**a), now=as_of)
        if decision.accepted:
            accepted.append(a)
        else:
            rejected.append({
                "source": a.get("source"),
                "title": a.get("title"),
                "reason": decision.reason,
            })
    return accepted, rejected


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальное тестирование NLI-теггера")
    parser.add_argument("input", type=Path, help="JSON из check_news_collection.py")
    parser.add_argument("--limit", type=int, default=10, help="Сколько статей проверить после первичного отбора")
    parser.add_argument("--source", type=str, default=None, help="Название источника")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--tag-threshold", type=float, default=0.12)
    parser.add_argument("--relevance-threshold", type=float, default=0.60)
    parser.add_argument("--max-tags", type=int, default=2)
    parser.add_argument(
        "--summary-mode", choices=("textrank", "lead"), default="textrank",
        help="textrank (по умолчанию — как в рабочем pipeline) или lead (старые проверки)",
    )
    parser.add_argument(
        "--primary-filter", action="store_true",
        help="Проверять только статьи, которые прошли основной фильтр",
    )
    parser.add_argument(
        "--as-of", type=date.fromisoformat, default=None,
        help="Дата первичного фильтра YYYY-MM-DD для воспроизводимости",
    )
    parser.add_argument("--output", type=Path, default=None, help="Сохранить результаты в JSON")
    args = parser.parse_args()

    if args.limit < 1 or args.max_tags < 1:
        parser.error("--limit и --max-tags должны быть >= 1")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    articles = data.get("articles", [])
    if not isinstance(articles, list):
        parser.error("Файл должен содержать articles: []")
    accepted, rejected = selected_articles(
        articles, source=args.source,
        primary_filter=args.primary_filter, as_of=args.as_of,
    )
    total_after_filter = len(accepted)
    articles = accepted[:args.limit]

    print(f"Модель: {args.model}")
    print(f"Режим summary: {args.summary_mode}")
    print(f"После первичного отбора: {total_after_filter}; отклонено: {len(rejected)}")
    print("Это только проверка; PostgreSQL и MAX не используются.\n")
    if not articles:
        print("Подходящих статей в файле нет.")
        return

    classifier = load_classifier(args.model)
    results = []
    for i, a in enumerate(articles, start=1):
        title = a.get("title", "")
        text = a.get("text", "")
        summary, lead = prepare_text_for_ai(title, text, args.summary_mode)
        result = predict_tags(
            title=title, summary=summary, lead=lead,
            max_tags=args.max_tags, tag_threshold=args.tag_threshold,
            relevance_threshold=args.relevance_threshold,
            classifier=classifier,
        )
        print(f"\n{i}. [{a.get('source')}] {title}")
        print(f"   AI: {', '.join(result.tags) if result.tags else 'НЕТ ТЕГОВ'}"
              f"; reason={result.reason}; industry_score={result.relevance_score:.3f}")
        print(f"   Главное событие: {result.event_type or 'не определено'}")
        print("   Типы событий: " + ", ".join(
            f"{event} {score:.2f}" for event, score in (result.ranked_events or [])[:4]
        ))
        print("   Отрасли: " + ", ".join(
            f"{tag} {score:.2f}" for tag, score in result.ranked_tags[:4]
        ))
        results.append({
            "source": a.get("source"),
            "title": title, "url": a.get("url"),
            "summary_used": summary,
            "tags": result.tags, "reason": result.reason,
            "industry_score": result.relevance_score,
            "event_type": result.event_type,
            "ranked_events": [
                {"event": event, "score": round(score, 4)}
                for event, score in (result.ranked_events or [])
            ],
            "ranked_tags": [
                {"tag": tag, "score": round(score, 4)}
                for tag, score in result.ranked_tags
            ],
        })

    print(f"\nСтатей: {len(results)}, с тегами: "
          f"{sum(bool(r['tags']) for r in results)}, без тегов: "
          f"{sum(not r['tags'] for r in results)}")
    print("Главные события: " + ", ".join(
        f"{key}: {count}" for key, count in Counter(
            r["event_type"] for r in results
        ).most_common(8)
    ))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({
                "model": args.model,
                "algorithm": "event_first_multiclass_v2",
                "summary_mode": args.summary_mode,
                "primary_filter": args.primary_filter,
                "as_of": args.as_of.isoformat() if args.as_of else None,
                "tag_threshold": args.tag_threshold,
                "relevance_threshold": args.relevance_threshold,
                "total_after_primary_filter": total_after_filter,
                "skipped_by_primary_filter": rejected,
                "note": (
                    "single-article TextRank proxy; multi-article cluster summaries may differ"
                    if args.summary_mode == "textrank"
                    else "lead proxy: first article characters, not real TextRank summary"
                ),
                "results": results,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Отчёт: {args.output}")


if __name__ == "__main__":
    main()
