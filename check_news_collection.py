"""
Проверка новой версии сборщика без записи в PostgreSQL.
Примеры:
  python check_news_collection.py --limit 5
  python check_news_collection.py --source НедраДВ --limit 5
  python check_news_collection.py --source "ЛУКОЙЛ" --limit 3
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from collectors.data_collector import collect_news
from database.db import get_session
from database.models import Source

OUTPUT_DIR = Path(__file__).resolve().parent / "collector_test_results"


def main():
    parser = argparse.ArgumentParser(description="Проверка сбора промышленных новостей")
    parser.add_argument("--source", default=None, help="Имя источника из базы")
    parser.add_argument("--limit", type=int, default=5, help="Лимит на источник")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit должен быть >= 1")

    session = get_session()
    try:
        query = session.query(Source).filter(Source.is_active.is_(True))
        if args.source:
            query = query.filter(Source.name == args.source)
        sources = [
            SimpleNamespace(
                name=s.name,
                url=s.url,
                rss_url=s.rss_url,
                source_type=s.source_type,
            )
            for s in query.all()
        ]
    finally:
        session.close()

    if not sources:
        print("Нет активных источников для проверки.")
        return

    # Никакой сессии для collect_news: он не проверяет и не пишет URL в БД.
    articles = collect_news(
        sources=sources,
        limit_per_source=args.limit,
        session=None,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = datetime.now().strftime("news_fixed_%Y%m%d_%H%M%S.json")
    destination = OUTPUT_DIR / filename
    counts = {source.name: 0 for source in sources}

    records = []
    for article in articles:
        counts[article.source] = counts.get(article.source, 0) + 1
        records.append({
            "source": article.source,
            "title": article.title,
            "url": article.url,
            "published_at": (
                article.published_at.isoformat() if article.published_at else None
            ),
            "text_source": article.text_source,
            "text_length": len(article.text or ""),
            "text": article.text,
        })

    with destination.open("w", encoding="utf-8") as handle:
        json.dump({
            "collected_at": datetime.now().isoformat(),
            "source_limits": args.limit,
            "sources": [s.name for s in sources],
            "counts": counts,
            "total_articles": len(records),
            "articles": records,
        }, handle, ensure_ascii=False, indent=2)

    print("\nИТОГИ СБОРА:")
    for name, count in counts.items():
        print(f"  {name}: {count}")
    print(f"\nСохранено: {len(records)}")
    print(f"Файл: {destination}")


if __name__ == "__main__":
    main()
