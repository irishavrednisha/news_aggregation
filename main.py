import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from collectors.data_collector import collect_news
from database.cleanup import delete_old_not_for_publication_clusters
from database.db import create_tables, get_session
from database.repository import (
    seed_sources,
    get_active_sources,
    save_articles,
)

from database.models import News

from processing.topic_filter import (
    filter_articles,
    print_filter_report,
)

from processing.clustering import (
    cluster_recent_news,
    print_recent_clusters,
    print_cluster_details,
)

from processing.summarizer import summarize_new_clusters
from processing.tagger import tag_new_clusters

from publishing.publisher import publish_ready_clusters

def main():
    print("Создание таблиц...")
    create_tables()

    session = get_session()

    try:
        print("Подготовка источников...")
        seed_sources(session)

        print("Получение активных источников...")
        sources = get_active_sources(session)

        news_count = session.query(News).count()
        is_first_run = news_count == 0

        if is_first_run:
            limit_per_source = 10
            clustering_hours = 48
        else:
            limit_per_source = 10
            clustering_hours = 6

        print(f"Новостей в базе: {news_count}")
        print(f"Режим запуска: {'первичный' if is_first_run else 'регулярный'}")
        print(f"Лимит на источник: {limit_per_source}")
        print(f"Окно кластеризации: {clustering_hours} ч.")

        print(f"Активных источников: {len(sources)}")

        print("Сбор новостей...")
        articles = collect_news(
            sources=sources,
            limit_per_source=limit_per_source
        )

        print(f"\nСобрано материалов: {len(articles)}")

        print("\nФильтрация служебных и нерелевантных материалов...")
        filtered_articles = filter_articles(articles)

        print_filter_report(articles, filtered_articles)

        print("\nСохранение новостей в базу данных...")
        saved_count = save_articles(session, filtered_articles)

        print(f"Сохранено новых новостей: {saved_count}")

        print("\nКластеризация свежих новостей...")
        cluster_recent_news(
            session=session,
            hours=clustering_hours,
            threshold=0.85
        )

        print_recent_clusters(
            session=session,
            limit=20
        )
        print("\nСуммаризация новых кластеров...")
        summarize_new_clusters(
            session=session,
            sentence_count=4,
            max_chars=700
        )
        print("\nТегирование новых кластеров...")
        tag_new_clusters(
            session=session,
            min_score=0.5

        )

        print("\nПубликация готовых кластеров в MAX...")
        publish_ready_clusters(
            limit=30
        )

    finally:
        session.close()


if __name__ == "__main__":
    main()