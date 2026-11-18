import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import text as sql_text
from database.models import News, Cluster, Post
from processing.event_matching import (
    MODEL_NAME, MERGE_THRESHOLD, cosine, embedding_text,
    representative_key, choose_existing_cluster,
)


_model = None


def get_model():
    global _model

    if _model is None:
        print("Загрузка модели эмбеддингов...")
        # Lazy import: collection and lightweight audits never need torch/sklearn.
        from sentence_transformers import SentenceTransformer
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

    return embedding_text(news)


def build_embedding(text: str) -> list[float]:
    model = get_model()

    vector = model.encode(
        text,
        normalize_embeddings=True
    )

    return vector.tolist()


def ensure_news_embedding(session, news: News, *, recalculate: bool = False):
    """Rebuild unversioned legacy vectors when explicitly requested.

    The old pipeline persisted FULL-TEXT embeddings; those are incompatible
    with new title+lead embeddings, despite identical vector dimensions.
    """
    if news.embedding is None or recalculate:
        news.embedding = build_embedding(get_news_text(news))
        session.flush()


def cosine_sim(vec1, vec2) -> float:
    return cosine(vec1, vec2)


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
    """Most informative source; deterministic independent of collection order."""
    if not news_items:
        raise ValueError("Cannot select a primary from an empty cluster")
    return sorted(news_items, key=representative_key)[0]


def _frozen_cluster(session, cluster: Cluster) -> bool:
    """Published/approved Post snapshots are immutable by construction."""
    return (session.query(Post.id)
            .filter(Post.cluster_id == cluster.id)
            .filter(Post.status.in_(("approved", "published", "sending", "uncertain")))
            .first() is not None)


def update_cluster_primary_news(session, cluster: Cluster):
    news_items = (session.query(News)
                  .filter(News.cluster_id == cluster.id).all())
    if not news_items:
        return
    if cluster.status != "new" or _frozen_cluster(session, cluster):
        return  # Never change editorially prepared or approved titles/primary.
    chosen = choose_representative_news(news_items)
    for item in news_items:
        item.is_primary = (item.id == chosen.id)
    if chosen.title:
        cluster.title = chosen.title
    session.flush()


def update_cluster_embedding(session, cluster: Cluster, *, embedding_cache=None):
    news_items = (session.query(News)
                  .filter(News.cluster_id == cluster.id).all())
    embedding_cache = embedding_cache or {}
    vectors = [np.asarray(embedding_cache.get(n.id, n.embedding), dtype=float)
               for n in news_items
               if embedding_cache.get(n.id, n.embedding) is not None]
    cluster.cluster_size = len(news_items)  # count ALL attached sources
    if vectors:
        mean = np.mean(vectors, axis=0)
        norm = float(np.linalg.norm(mean))
        cluster.embedding = (mean / norm).tolist() if norm else None
    cluster.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    session.flush()


def attach_news_to_cluster(session, news: News, cluster: Cluster, *, embedding_cache=None):
    """One caller-owned transaction, no intermediate commit."""
    if news.cluster_id is not None:
        return
    news.cluster_id = cluster.id
    news.is_primary = False
    session.flush()
    update_cluster_embedding(session, cluster, embedding_cache=embedding_cache)
    update_cluster_primary_news(session, cluster)


def create_cluster_with_news(session, news: News) -> Cluster:
    """Create cluster+membership atomically; caller must commit."""
    cluster = Cluster(
        title=news.title, summary_text=None, tags=None, cluster_size=1,
        status="new", embedding=news.embedding
    )
    session.add(cluster)
    session.flush()
    news.cluster_id = cluster.id
    news.is_primary = True
    session.flush()
    return cluster


def find_best_cluster(news: News, clusters: list[Cluster],
                      threshold: float = MERGE_THRESHOLD, session=None,
                      embedding_cache=None):
    """Legacy-compatible function with factual verification; no centroid-only merge."""
    if session is None:
        return None, 0.0  # no evidence -> no potentially destructive merge
    embedding_cache = embedding_cache if embedding_cache is not None else {}
    candidates = []
    for cluster in sorted(clusters, key=lambda c: c.id):
        items = (session.query(News)
                 .filter(News.cluster_id == cluster.id)
                 .order_by(News.id.asc()).all())
        members = []
        for member in items:
            # Cache normalized NEW text encoding in memory only. Historical
            # DB embeddings can come from the old full-text model path.
            if member.id not in embedding_cache:
                embedding_cache[member.id] = build_embedding(get_news_text(member))
            members.append((member, embedding_cache[member.id]))
        candidates.append({
            "id": cluster.id, "cluster": cluster, "items": members,
        })
    match, decision = choose_existing_cluster(news, news.embedding, candidates,
                                              threshold=threshold)
    return (match["cluster"] if match else None), decision.similarity


def cluster_recent_news(session, hours: int = 48,
                        threshold: float = MERGE_THRESHOLD):
    """Conservative stage 3 database writer; never called by audit script.

    Transaction-scoped PostgreSQL advisory lock serializes competing workers.
    A crash rolls back cluster creation + news membership together.
    """
    time_limit = get_time_limit(hours)
    recent = (session.query(News)
              .filter(News.cluster_id.is_(None))
              .filter(News.collected_at >= time_limit)
              .order_by(News.published_at.asc().nullslast(),
                        News.collected_at.asc(), News.id.asc()).all())
    print(f"Новых новостей без кластера: {len(recent)}")
    changed = []
    changed_ids = set()
    embedding_cache = {}
    for original in recent:
        try:
            # Lock must be acquired BEFORE fresh cluster candidate reads.
            if session.get_bind().dialect.name == "postgresql":
                session.execute(sql_text("SELECT pg_advisory_xact_lock(2026111803)"))
            news = (session.query(News).filter(News.id == original.id)
                    .with_for_update().first())
            if news is None or news.cluster_id is not None:
                session.rollback()
                continue
            ensure_news_embedding(session, news, recalculate=True)
            embedding_cache[news.id] = news.embedding
            # Larger than the incoming collection window to allow late mirrors.
            cluster_limit = get_time_limit(max(hours, 24 * 14))
            clusters = (session.query(Cluster)
                        .filter(Cluster.updated_at >= cluster_limit)
                        .order_by(Cluster.id.asc()).all())
            candidate, score = find_best_cluster(
                news, clusters, threshold, session=session,
                embedding_cache=embedding_cache
            )
            if candidate is not None:
                attach_news_to_cluster(session, news, candidate,
                                       embedding_cache=embedding_cache)
                result = candidate
                print(f"Присоединено #{news.id} к кластеру #{result.id} ({score:.3f})")
            else:
                result = create_cluster_with_news(session, news)
                print(f"Создан кластер #{result.id} для новости #{news.id}")
            session.commit()
            if result.id not in changed_ids:
                changed_ids.add(result.id)
                changed.append(result)
        except Exception:
            session.rollback()
            raise
    return changed


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