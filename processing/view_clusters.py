import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from database.db import get_session
from database.models import Cluster


def show_clusters(limit: int = 100):
    session = get_session()

    clusters = (
        session.query(Cluster)
        .order_by(Cluster.id.desc())
        .limit(limit)
        .all()
    )

    print("\nПОСЛЕДНИЕ КЛАСТЕРЫ")
    print("=" * 140)

    for cluster in clusters:
        news_count = len(cluster.news)

        print(
            f"[{cluster.id}] "
            f"{cluster.title or 'Без заголовка'} | "
            f"Новостей: {news_count} | "
            f"Теги: {cluster.tags or 'нет'} | "
            f"Статус: {cluster.status or 'unknown'}\n"
            f"Summary: {cluster.summary_text}"
        )

    while True:
        print("\nВведите ID кластера для просмотра (или q для выхода):")

        user_input = input(">>> ").strip()

        if user_input.lower() == "q":
            break

        if not user_input.isdigit():
            print("Введите число")
            continue

        cluster_id = int(user_input)

        cluster = (
            session.query(Cluster)
            .filter(Cluster.id == cluster_id)
            .first()
        )

        if not cluster:
            print("Кластер не найден")
            continue

        print("\n" + "=" * 140)
        print(f"КЛАСТЕР {cluster.id}")
        print("=" * 140)

        print(f"\nЗаголовок:\n{cluster.title}")

        print(f"\nТеги:\n{cluster.tags or 'нет'}")

        print(f"\nСтатус:\n{cluster.status}")

        print(f"\nSummary:\n{cluster.summary_text or 'нет summary'}")

        print("\nНОВОСТИ КЛАСТЕРА")
        print("-" * 140)

        sorted_news = sorted(
            cluster.news,
            key=lambda n: n.published_at or ""
        )

        for i, news in enumerate(sorted_news, start=1):

            print(f"\n[{i}] {news.title}")

            print(
                f"Источник: "
                f"{news.source.name if news.source else 'неизвестно'}"
            )

            print(f"Дата: {news.published_at}")

            print(f"Ссылка: {news.url}")

            print(f"Теги: {cluster.tags or 'нет'} | ")

            if news.text:
                preview = news.text[:700].replace("\n", " ")

                print(f"\nТекст:\n{preview}...")

            print("-" * 140)

    session.close()


if __name__ == "__main__":
    show_clusters(limit=100)