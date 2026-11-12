"""Preview main images from 1 source. Never modifies PostgreSQL or posts to MAX.

Example:
  python scripts/check_news_images.py --source "НедраДВ" --limit 2
  python scripts/check_news_images.py --source "НедраДВ" --limit 2 --download
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from collectors.data_collector import collect_news
from database.db import get_session
from database.models import Source
from processing.image_downloader import download_main_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only article/image audit")
    parser.add_argument("--source", default="НедраДВ", help="Имя источника из таблицы sources")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--download", action="store_true",
                        help="Download images to local data/news_images (no DB writes)")
    args = parser.parse_args()
    if not 1 <= args.limit <= 10:
        parser.error("--limit must be between 1 and 10")

    session = get_session()
    try:
        source = session.query(Source).filter(Source.name == args.source).first()
        if source is None:
            parser.error(f"Источник не найден: {args.source}")
        print(f"Проверка изображений: {source.name}")
        # Intentionally pass session=None: no DB duplicate check, no DB writes.
        articles = collect_news(
            [source], limit_per_source=args.limit,
            session=None, apply_filters=False,
        )
        print(f"Получено статей: {len(articles)}")
        for index, article in enumerate(articles, 1):
            print(f"\n[{index}] {article.title}\n{article.url}")
            candidates = getattr(article, "image_candidates", []) or []
            print(f"Кандидатов изображений: {len(candidates)}")
            for candidate in candidates[:3]:
                print(f"  [{candidate.origin}] {candidate.url}")
            if args.download:
                downloaded = download_main_image(candidates)
                if downloaded:
                    print(
                        f"Сохранено: {downloaded.local_path} "
                        f"({downloaded.width}x{downloaded.height})"
                    )
                    print("Права: НЕ ПОДТВЕРЖДЕНЫ — публикация запрещена.")
                else:
                    print("Подходящей картинки нет; текстовая новость допустима.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
