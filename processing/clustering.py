import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from datetime import datetime, timedelta, timezone

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from database.models import News, Cluster


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_model = None


def get_model():
    global _model

    if _model is None:
        print("Загрузка модели эмбеддингов...")
        _model = SentenceTransformer(MODEL_NAME)

    return _model


def get_time_limit(hours: int = 48) -> datetime:
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    return now_utc - timedelta(hours=hours)


def get_news_text(news: News) -> str:
    """
    Текст для построения embedding.
    Используем заголовок + полный текст статьи.
    """

    title = news.title or ""
    text = news.text or ""

    return f"{title}. {text}"


def build_embedding(text: str) -> list[float]:
    model = get_model()

    vector = model.encode(
        text,
        normalize_embeddings=True
    )

    return vector.tolist()


def ensure_news_embedding(session, news: News):
    """
    Создаёт embedding новости, если его ещё нет.
    """
    if news.embedding is None:
        news.embedding = build_embedding(get_news_text(news))
        session.commit()


def cosine_sim(vec1, vec2) -> float:
    if vec1 is None or vec2 is None:
        return 0.0

    a = np.array(vec1, dtype=float).reshape(1, -1)
    b = np.array(vec2, dtype=float).reshape(1, -1)

    return float(cosine_similarity(a, b)[0][0])


def get_recent_clusters(session, hours: int = 48) -> list[Cluster]:
    """
    Берём свежие кластеры.
    Лучше смотреть updated_at, потому что старый кластер мог обновиться новой новостью.
    """
    time_limit = get_time_limit(hours)

    return (
        session.query(Cluster)
        .filter(Cluster.updated_at >= time_limit)
        .all()
    )


def choose_representative_news(news_items: list[News]) -> News:
    """
    Главная новость кластера — самая ранняя по published_at.
    Если published_at нет, берём самую раннюю по collected_at.
    """
    with_dates = [news for news in news_items if news.published_at is not None]

    if with_dates:
        return min(with_dates, key=lambda news: news.published_at)

    return min(news_items, key=lambda news: news.collected_at)


def update_cluster_primary_news(session, cluster: Cluster):
    news_items = (
        session.query(News)
        .filter(News.cluster_id == cluster.id)
        .all()
    )

    if not news_items:
        return

    representative = choose_representative_news(news_items)

    for news in news_items:
        news.is_primary = news.id == representative.id

    if representative.title:
        cluster.title = representative.title

    session.commit()


def update_cluster_embedding(session, cluster: Cluster):
    """
    Embedding кластера = средний embedding всех новостей кластера.
    """
    news_items = (
        session.query(News)
        .filter(News.cluster_id == cluster.id)
        .filter(News.embedding.isnot(None))
        .all()
    )

    if not news_items:
        return

    vectors = [np.array(news.embedding, dtype=float) for news in news_items]
    mean_vector = np.mean(vectors, axis=0)

    norm = np.linalg.norm(mean_vector)

    if norm > 0:
        mean_vector = mean_vector / norm

    cluster.embedding = mean_vector.tolist()
    cluster.cluster_size = len(news_items)

    session.commit()


def attach_news_to_cluster(session, news: News, cluster: Cluster):
    news.cluster_id = cluster.id
    news.is_primary = False

    session.commit()

    update_cluster_embedding(session, cluster)
    update_cluster_primary_news(session, cluster)


def create_cluster_with_news(session, news: News) -> Cluster:
    cluster = Cluster(
        title=news.title,
        summary_text=None,
        tags=None,
        cluster_size=1,
        status="new",
        embedding=news.embedding
    )

    session.add(cluster)
    session.commit()
    session.refresh(cluster)

    news.cluster_id = cluster.id
    news.is_primary = True

    session.commit()

    return cluster


def find_best_cluster(
    news: News,
    clusters: list[Cluster],
    threshold: float = 0.78
):
    best_cluster = None
    best_score = 0.0

    for cluster in clusters:
        if cluster.embedding is None:
            continue

        score = cosine_sim(news.embedding, cluster.embedding)

        if score > best_score:
            best_score = score
            best_cluster = cluster

    if best_cluster is not None and best_score >= threshold:
        return best_cluster, best_score

    return None, best_score


def cluster_recent_news(
    session,
    hours: int = 48,
    threshold: float = 0.78
):
    """
    Кластеризует свежие новости без cluster_id.
    Новая новость сравнивается с уже существующими свежими кластерами.
    """
    time_limit = get_time_limit(hours)

    news_items = (
        session.query(News)
        .filter(News.cluster_id.is_(None))
        .filter(News.collected_at >= time_limit)
        .order_by(News.collected_at.asc())
        .all()
    )

    print(f"Новых новостей без кластера за последние {hours} часов: {len(news_items)}")

    if not news_items:
        return []

    recent_clusters = get_recent_clusters(session, hours=hours)
    changed_clusters = []

    print(f"Свежих кластеров для сравнения: {len(recent_clusters)}")

    for news in news_items:
        ensure_news_embedding(session, news)

        best_cluster, score = find_best_cluster(
            news=news,
            clusters=recent_clusters,
            threshold=threshold
        )

        if best_cluster:
            print("-" * 100)
            print(f"Добавляем в кластер #{best_cluster.id}")
            print(f"Новость: {news.title}")
            print(f"Сходство: {score:.3f}")

            attach_news_to_cluster(session, news, best_cluster)

            if best_cluster not in changed_clusters:
                changed_clusters.append(best_cluster)

        else:
            print("-" * 100)
            print("Создаём новый кластер")
            print(f"Новость: {news.title}")
            print(f"Максимальное сходство: {score:.3f}")

            new_cluster = create_cluster_with_news(session, news)
            recent_clusters.append(new_cluster)
            changed_clusters.append(new_cluster)

    return changed_clusters


def print_recent_clusters(session, limit: int = 20):
    clusters = (
        session.query(Cluster)
        .order_by(Cluster.created_at.desc())
        .limit(limit)
        .all()
    )

    print("\n" + "=" * 100)
    print("Последние кластеры новостей")

    clusters = list(reversed(clusters))

    for cluster in clusters:
        news_items = (
            session.query(News)
            .filter(News.cluster_id == cluster.id)
            .all()
        )

        print("-" * 100)
        print(f"Кластер #{cluster.id}")
        print(f"Заголовок: {cluster.title}")
        print(f"Размер: {len(news_items)}")
        print(f"Статус: {cluster.status}")
        print(f"Теги: {cluster.tags}")

        for news in news_items:
            mark = "[PRIMARY]" if news.is_primary else "         "
            print(f"{mark} {news.title}")
            print(f"          {news.url}")


def print_cluster_details(session, cluster_id: int):
    cluster = session.query(Cluster).filter(Cluster.id == cluster_id).first()

    if not cluster:
        print(f"Кластер #{cluster_id} не найден.")
        return

    news_items = (
        session.query(News)
        .filter(News.cluster_id == cluster.id)
        .order_by(News.published_at.asc().nullslast(), News.collected_at.asc())
        .all()
    )

    print("\n" + "=" * 100)
    print(f"Кластер #{cluster.id}")
    print(f"Заголовок: {cluster.title}")
    print(f"Размер кластера: {len(news_items)}")
    print(f"Статус: {cluster.status}")
    print(f"Summary: {cluster.summary_text}")
    print(f"Tags: {cluster.tags}")

    for index, news in enumerate(news_items, start=1):
        mark = "ОСНОВНАЯ" if news.is_primary else "дубликат"

        print("-" * 100)
        print(f"Новость #{index} ({mark})")
        print(f"Источник: {news.source.name if news.source else 'неизвестно'}")
        print(f"Заголовок: {news.title}")
        print(f"Ссылка: {news.url}")
        print(f"Дата публикации: {news.published_at}")
        print(f"Дата сбора: {news.collected_at}")
        print(f"Текст: {news.text[:700]}...")