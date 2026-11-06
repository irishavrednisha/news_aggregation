"""
Проверка новой версии сборщика без записи в PostgreSQL.
Примеры:
  python check_news_collection.py --limit 5
  python check_news_collection.py --source НедраДВ --limit 5
  python check_news_collection.py --source "ЛУКОЙЛ" --limit 3
"""
import argparse
import json
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from collectors.data_collector import collect_news
from database.db import get_session
from database.models import Source

OUTPUT_DIR = Path(__file__).resolve().parent / "collector_test_results"


def main():
    parser = argparse.ArgumentParser(description="Проверка сбора промышленных новостей")
    parser.add_argument(
        "--source", action="append", default=None,
        help="Имя источника из базы (можно указать несколько раз)"
    )
    parser.add_argument("--limit", type=int, default=5, help="Лимит на источник")
    parser.add_argument("--dump-html", action="store_true", help="Сохранить HTML неудачных источников в ZIP")
    args = parser.parse_args()

    if args.limit < 1:
        parser.error("--limit должен быть >= 1")

    session = get_session()
    try:
        query = session.query(Source).filter(Source.is_active.is_(True))
        if args.source:
            query = query.filter(Source.name.in_(args.source))
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
    diagnostics = {}
    articles = collect_news(
        sources=sources,
        limit_per_source=args.limit,
        session=None,
        diagnostics=diagnostics,
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
            "diagnostics": diagnostics,
            "total_articles": len(records),
            "articles": records,
        }, handle, ensure_ascii=False, indent=2)

    print("\nИТОГИ СБОРА:")
    for name, count in counts.items():
        d = diagnostics.get(name, {})
        print(
            f"  {name}: {count} статей, "
            f"{d.get('candidates', 0)} кандидатов, "
            f"{d.get('short_or_empty', 0)} без текста, "
            f"статус={d.get('status', 'unknown')}"
        )
        for err in d.get("errors", [])[:3]:
            print(f"    причина: {err}")
    print(f"\nСохранено: {len(records)}")
    print(f"Файл: {destination}")

    if args.dump_html:
        # Сохраняем неудачные страницы, а также один образец статьи,
        # если источник собирается без даты публикации.
        from collections import defaultdict
        samples_without_date = defaultdict(list)
        for item in records:
            if not item["published_at"]:
                samples_without_date[item["source"]].append(item["url"])

        # Если источник вернул 0 статей, сохраняем HTML его ленты;
        # если есть ссылки без текста, также сохраняем первые две
        # страницы конкретных новостей. Это публичные страницы сайтов.
        from collectors.data_collector import fetch_html
        archive_path = OUTPUT_DIR / (
            datetime.now().strftime("collector_html_%Y%m%d_%H%M%S.zip")
        )
        snapshot_log = []
        with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
            for idx, src in enumerate(sources, start=1):
                info = diagnostics.get(src.name, {})
                missing_date_urls = samples_without_date.get(src.name, [])
                if info.get("accepted", 0) >= args.limit and not missing_date_urls:
                    continue
                addresses = [src.url] if info.get("accepted", 0) < args.limit else []
                if src.name == "Производства.рф":
                    addresses.append("https://производства.рф/articles/")
                if src.name == "Ростех":
                    addresses.append("https://www.rostec.ru/media/news/")
                if src.name == "Роснефть":
                    addresses.append("https://www.rosneft.ru/press/news/2026/")
                addresses += info.get("rejected_urls", [])[:2]
                addresses += missing_date_urls[:1]
                for n, url in enumerate(dict.fromkeys(addresses)):
                    try:
                        html = fetch_html(url)
                        archive.writestr(
                            f"source_{idx:02}_page_{n:02}.html",
                            html or "<!-- FETCH FAILED: EMPTY HTML -->",
                        )
                        record = {
                            "source": src.name, "url": url,
                            "html_length": len(html or ""),
                            "snapshot": f"source_{idx:02}_page_{n:02}.html",
                        }
                        if html and "Error establishing a database connection" in html:
                            record["page_error"] = "origin_database_connection_error"
                        if not html and n == 0:
                            # Если браузер и requests вернули пустой результат,
                            # сохранение отдельной ошибки requests полезнее
                            # пустой HTML. Только один короткий HTTP-пробник.
                            import requests
                            from collectors.data_collector import HEADERS
                            try:
                                resp = requests.get(
                                    url, headers=HEADERS, timeout=12,
                                )
                                record["http_status"] = resp.status_code
                                record["http_final_url"] = resp.url
                            except requests.RequestException as exc:
                                record["http_exception"] = (
                                    f"{type(exc).__name__}: {exc}"
                                )[:500]
                        snapshot_log.append(record)
                    except Exception as error:
                        snapshot_log.append({
                            "source": src.name, "url": url,
                            "error": repr(error),
                        })
            archive.writestr(
                "manifest.json",
                json.dumps(snapshot_log, ensure_ascii=False, indent=2),
            )
        print(f"Снимки проблемных страниц: {archive_path}")



if __name__ == "__main__":
    main()
