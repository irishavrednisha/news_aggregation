"""Тест первичной фильтрации готового collector JSON без БД и публикации.

Пример:
  python check_news_filter.py collector_test_results/news_fixed_20261106_223822.json
  python check_news_filter.py collector_test_results/news_fixed_20261106_223822.json --max-age-days 7
  python check_news_filter.py collector_test_results/news_fixed_20261106_223822.json --show-kept
"""
import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace

from processing.topic_filter import classify_article


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Офлайн-проверка свежести и релевантности собранных новостей"
    )
    parser.add_argument("input", type=Path, help="JSON из check_news_collection.py")
    parser.add_argument(
        "--max-age-days", type=int, default=None,
        help="Максимальный возраст (по умолчанию 30, 0 — без ограничения)",
    )
    parser.add_argument(
        "--date", type=date.fromisoformat, default=None,
        help="Дата для воспроизводимого сравнения YYYY-MM-DD",
    )
    parser.add_argument(
        "--show-kept", action="store_true",
        help="Также вывести оставленные новости с причинами",
    )
    args = parser.parse_args()
    if args.max_age_days is not None and args.max_age_days < 0:
        parser.error("--max-age-days не может быть отрицательным")

    with args.input.open("r", encoding="utf-8") as handle:
        doc = json.load(handle)
    records = doc.get("articles", [])
    if not isinstance(records, list):
        parser.error("Входной JSON не содержит массива articles")

    results = []
    for record in records:
        article = SimpleNamespace(**record)
        decision = classify_article(
            article, now=args.date, max_age_days=args.max_age_days
        )
        results.append((record, decision))

    reasons = Counter(decision.reason for _, decision in results)
    accepted = [(record, decision) for record, decision in results if decision.accepted]
    rejected = [(record, decision) for record, decision in results if not decision.accepted]

    print(f"Всего: {len(results)}; оставить: {len(accepted)}; убрать: {len(rejected)}")
    print("Причины:")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")

    print("\nОТКЛОНЕНЫ:")
    for record, decision in rejected:
        print(f"[{decision.reason}] {record.get('source')} — {record.get('title')}")

    if args.show_kept:
        print("\nОСТАВЛЕНЫ:")
        for record, decision in accepted:
            print(f"[{decision.reason}] {record.get('source')} — {record.get('title')}")

    print(
        "\nЭто только предварительный фильтр; кластеризация, "
        "тегирование и публикация здесь не выполняются."
    )


if __name__ == "__main__":
    main()
