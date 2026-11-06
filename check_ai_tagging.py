"""Безопасная офлайн-проверка AI-тегирования по collector JSON.

Никаких записей в PostgreSQL, публикации в MAX и изменения cluster.status.
В этом инструменте вместо TextRank-summary используется начало статьи —
оценки могут отличаться от результата в рабочем конвейере.

Пример:
  python check_ai_tagging.py collector_test_results/news_fixed_20261106_223822.json --limit 10
"""
import argparse
import json
from pathlib import Path

from processing.ai_tagger import (
    DEFAULT_MODEL, load_classifier, predict_tags,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Локальное тестирование NLI-теггера")
    parser.add_argument("input", type=Path, help="JSON из check_news_collection.py")
    parser.add_argument("--limit", type=int, default=10, help="Сколько статей проверить")
    parser.add_argument("--source", type=str, default=None, help="Название источника")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--tag-threshold", type=float, default=0.12)
    parser.add_argument("--relevance-threshold", type=float, default=0.60)
    parser.add_argument("--max-tags", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None, help="Сохранить результаты в JSON")
    args = parser.parse_args()

    if args.limit < 1 or args.max_tags < 1:
        parser.error("--limit и --max-tags должны быть >= 1")

    data = json.loads(args.input.read_text(encoding="utf-8"))
    articles = data.get("articles", [])
    if not isinstance(articles, list):
        parser.error("Файл должен содержать articles: []")
    if args.source:
        articles = [x for x in articles if x.get("source") == args.source]
    articles = articles[:args.limit]

    print(f"Модель: {args.model}")
    print("При первом запуске веса скачиваются в кэш Hugging Face.")
    print("Это предварительное тестирование, без публикации в MAX.\n")
    if not articles:
        print("Подходящих статей в файле нет.")
        return

    classifier = load_classifier(args.model)
    results = []
    for i, a in enumerate(articles, start=1):
        title = a.get("title", "")
        text = a.get("text", "")
        # Это НЕ TextRank-summary, а только прокси для пилотного аудита.
        result = predict_tags(
            title=title, summary=text[:780], lead=text[780:1110],
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
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({
                "model": args.model,
                "algorithm": "event_first_multiclass_v2",
                "tag_threshold": args.tag_threshold,
                "relevance_threshold": args.relevance_threshold,
                "note": "NLI v2 event-first, relative tag scores, article lead instead of TextRank summary",
                "results": results,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Отчёт: {args.output}")


if __name__ == "__main__":
    main()
