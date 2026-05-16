import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from database.db import get_session
from database.models import News

from processing.clustering import (
    cluster_recent_news,
    print_recent_clusters,
    print_cluster_details
)

session = get_session()

# Проверяем сколько новостей есть
news_count = session.query(News).count()

print(f"Новостей в БД: {news_count}")

# Запускаем кластеризацию
changed_clusters = cluster_recent_news(
    session=session,
    hours=72,
    threshold=0.78
)

print("\n")
print("=" * 100)
print(f"Изменённых кластеров: {len(changed_clusters)}")

# Печать последних кластеров
print_recent_clusters(
    session=session,
    limit=20
)
print("\n\nДЕТАЛИ КЛАСТЕРОВ\n")

for cluster in changed_clusters:
    print_cluster_details(session, cluster.id)

session.close()