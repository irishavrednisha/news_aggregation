from datetime import datetime, timedelta

from database.db import get_session
from database.models import Cluster, News


def delete_old_not_for_publication_clusters(
    session,
    hours: int = 48,
    dry_run: bool = False
):
    border_time = datetime.now() - timedelta(hours=hours)

    clusters = (
        session.query(Cluster)
        .filter(Cluster.status == "not_for_publication")
        .filter(Cluster.updated_at < border_time)
        .all()
    )

    print(f"Кластеров для удаления: {len(clusters)}")

    deleted_news_count = 0

    for cluster in clusters:
        news_items = (
            session.query(News)
            .filter(News.cluster_id == cluster.id)
            .all()
        )

        print("-" * 100)
        print(f"Кластер #{cluster.id}")
        print(f"Заголовок: {cluster.title}")
        print(f"Новостей в кластере: {len(news_items)}")

        deleted_news_count += len(news_items)

        if not dry_run:
            for news in news_items:
                session.delete(news)

            session.delete(cluster)

    if not dry_run:
        session.commit()
        print("Удаление выполнено.")
    else:
        print("Тестовый режим: ничего не удалено.")

    print(f"Удалено кластеров: {len(clusters)}")
    print(f"Удалено новостей: {deleted_news_count}")


if __name__ == "__main__":
    session = get_session()

    try:
        delete_old_not_for_publication_clusters(
            session=session,
            hours=48,
            dry_run=False
        )
    finally:
        session.close()