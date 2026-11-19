"""Real PostgreSQL integration tests in an isolated, disposable CI database.

Never point this at an existing user database. Tests intentionally TRUNCATE the
named fixture DB after asserting explicit opt-in and an exact DB name match.
They exercise the application's real PostgreSQL advisory transaction locks,
SQLAlchemy ORM, Alembic-migrated intake records, and immutable Post snapshots.
The embedding model is mocked: semantic SBERT quality is tested elsewhere.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import os
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from database.models import Cluster, News, Post, Source
from database.intake_models import NewsIntakeRecord
from processing.clustering import cluster_recent_news, get_cluster_source_links


class Stage3PostgresIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value = os.getenv("STAGE3_POSTGRES_TEST_URL")
        if not value or os.getenv("STAGE3_TEST_WRITE_ACK") != "ephemeral-only":
            raise unittest.SkipTest("Only enabled for an explicitly isolated PostgreSQL test service")
        url = make_url(value)
        if url.get_backend_name() != "postgresql" or url.database != "stage3_ci":
            raise RuntimeError("Refusing to write: PostgreSQL test database MUST be named stage3_ci")
        cls.engine = create_engine(value, pool_pre_ping=True)
        if cls.engine.dialect.name != "postgresql":
            raise RuntimeError("This test must use PostgreSQL, never SQLite")
        with cls.engine.connect() as conn:
            version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            if version != "20261118_03":
                raise AssertionError(f"Real Alembic upgrade missing: {version!r}")
            existing = set(inspect(conn).get_table_names())
            expected = {"news_intake_records", "sources", "news", "posts", "clusters",
                        "moderation_items", "publication_attempts"}
            if not expected.issubset(existing):
                raise AssertionError(f"Missing migrated tables: {expected-existing}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine"):
            cls.engine.dispose()

    def setUp(self):
        # The explicit DB-name and write-ack guards in setUpClass are mandatory.
        with self.engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE sources, clusters RESTART IDENTITY CASCADE"))
        self.session = Session(bind=self.engine)
        for label in ("MASHNEWS", "Промышленный вестник", "ФлотПром", "Отраслевое.рф"):
            self.session.add(Source(name=label, url=f"https://{label[:3]}.example"))
        self.session.commit()
        self.sources = {x.name: x.id for x in self.session.query(Source).all()}

    def tearDown(self):
        self.session.close()

    def news(self, title, url, source="MASHNEWS", text_body="", days_ago=0):
        now = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago)
        instance = News(source_id=self.sources[source], title=title,
                        url=url, text=text_body, published_at=now,
                        collected_at=datetime.now(timezone.utc).replace(tzinfo=None))
        self.session.add(instance)
        self.session.commit()
        return instance

    def test_real_postgres_approval_source_links_idempotence(self):
        a = self.news("ЧМЗ выпустил партию автокранов 25 тонн", "https://unit.example/a",
                      text_body="Первая публикация.")
        b = self.news("Челябинский механический завод изготовил партию автокранов 25 тонн",
                      "https://unit.example/b", source="Промышленный вестник",
                      text_body="Подробности поставки автокранов. " * 80)
        different = self.news("ЧМЗ построил новый цех для автокранов",
                              "https://unit.example/new-shop",
                              source="Отраслевое.рф", text_body="Строительство нового цеха.")

        with patch("processing.clustering.build_embedding", return_value=[1.0, 0.0]):
            cluster_recent_news(self.session)
        self.session.expire_all()
        self.assertEqual(self.session.query(Cluster).count(), 2)
        self.assertEqual(a.cluster_id, b.cluster_id)
        self.assertNotEqual(a.cluster_id, different.cluster_id)
        self.assertTrue(b.is_primary)
        cluster_id = a.cluster_id
        cluster = self.session.get(Cluster, cluster_id)
        saved_title = cluster.title
        cluster.summary_text = "СОХРАНЁННАЯ СВОДКА"
        cluster.tags = "машиностроение"
        post = Post(cluster_id=cluster.id, status="approved",
                    post_text="УТВЕРЖДЁННЫЙ ТЕКСТ АДМИНИСТРАТОРА")
        self.session.add(post)
        self.session.commit()
        post_id = post.id

        raw = NewsIntakeRecord(
            source_id=self.sources["ФлотПром"],
            duplicate_of_news_id=a.id,
            raw_url="https://mirror.example/same-event",
            raw_title=a.title, raw_text=a.text or "",
            canonical_url="https://mirror.example/same-event",
            url_sha256="a"*64, payload_sha256="b"*64,
            status="rejected", reason="duplicate_text",
            decision_stage="exact_duplicate")
        self.session.add(raw)
        self.session.commit()

        c = self.news("ЧМЗ изготовил партию автокранов 25 тонн",
                      "https://unit.example/c", source="ФлотПром",
                      text_body="Новейшие подробности о партии автокранов. " * 250)
        with patch("processing.clustering.build_embedding", return_value=[1.0, 0.0]):
            cluster_recent_news(self.session)
        self.session.expire_all()
        self.assertEqual(c.cluster_id, cluster_id)
        cluster = self.session.get(Cluster, cluster_id)
        self.assertEqual(cluster.title, saved_title)
        self.assertEqual(cluster.summary_text, "СОХРАНЁННАЯ СВОДКА")
        self.assertEqual(cluster.tags, "машиностроение")
        self.assertEqual(cluster.cluster_size, 3)
        self.assertTrue(b.is_primary)
        self.assertFalse(c.is_primary)
        self.assertEqual(self.session.get(Post, post_id).post_text,
                         "УТВЕРЖДЁННЫЙ ТЕКСТ АДМИНИСТРАТОРА")
        self.assertEqual(self.session.get(Post, post_id).status, "approved")

        links = get_cluster_source_links(self.session, cluster_id)
        self.assertEqual(len(links), 4)
        self.assertEqual({x["url"] for x in links},
                         {a.url, b.url, c.url, "https://mirror.example/same-event"})
        self.assertEqual([x["link_type"] for x in links].count("stage2_exact_copy"), 1)
        before = (self.session.query(News).count(), self.session.query(Cluster).count(),
                  self.session.query(NewsIntakeRecord).count(), cluster.cluster_size)
        with patch("processing.clustering.build_embedding", return_value=[1.0, 0.0]):
            cluster_recent_news(self.session)
            cluster_recent_news(self.session)
        self.session.expire_all()
        after = (self.session.query(News).count(), self.session.query(Cluster).count(),
                 self.session.query(NewsIntakeRecord).count(),
                 self.session.get(Cluster, cluster_id).cluster_size)
        self.assertEqual(before, after)

    def test_real_postgres_rollback_and_resume(self):
        a = self.news("ЧМЗ выпустил партию автокранов 25 тонн", "https://rollback.example/one")
        with patch("processing.clustering.build_embedding", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                cluster_recent_news(self.session)
        self.session.expire_all()
        self.assertIsNone(a.cluster_id)
        self.assertEqual(self.session.query(Cluster).count(), 0)
        with patch("processing.clustering.build_embedding", return_value=[1.0, 0.0]):
            cluster_recent_news(self.session)
        self.session.expire_all()
        self.assertEqual(self.session.query(Cluster).count(), 1)
        self.assertIsNotNone(a.cluster_id)

    def test_real_postgres_parallel_workers_do_not_duplicate_clusters(self):
        a = self.news("ЧМЗ выпустил партию автокранов 25 тонн", "https://race.example/one")
        b = self.news("ЧМЗ изготовил партию автокранов 25 тонн", "https://race.example/two",
                      source="Промышленный вестник")
        def process(_):
            with Session(bind=self.engine) as session:
                cluster_recent_news(session)
        with patch("processing.clustering.build_embedding", return_value=[1.0, 0.0]):
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(process, [0, 1]))
        self.session.expire_all()
        self.assertEqual(self.session.query(Cluster).count(), 1)
        self.assertEqual(a.cluster_id, b.cluster_id)
        self.assertEqual(self.session.query(News).filter(News.cluster_id.is_(None)).count(), 0)
        self.assertEqual(self.session.get(Cluster, a.cluster_id).cluster_size, 2)


if __name__ == "__main__":
    unittest.main()
