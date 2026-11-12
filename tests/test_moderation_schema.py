"""Stage 2 migration tests: SQLite in memory; never touch the local PostgreSQL."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.db import Base
from database.models import Cluster, News, Post, Source
from database.moderation_models import (
    ModerationAction,
    ModerationItem,
    NewsImage,
    PublicationAttempt,
)


MIGRATION = (
    Path(__file__).resolve().parent.parent
    / "migrations"
    / "versions"
    / "20261112_01_moderation_images.py"
)


def apply_migration(engine):
    spec = importlib.util.spec_from_file_location("stage2_migration", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    with engine.begin() as connection:
        migration_context = MigrationContext.configure(connection)
        with Operations.context(migration_context):
            module.upgrade()


class ModerationSchemaTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")

    def tearDown(self):
        self.engine.dispose()

    def _legacy_tables(self):
        Base.metadata.create_all(
            bind=self.engine,
            tables=[
                Source.__table__,
                Cluster.__table__,
                News.__table__,
                Post.__table__,
            ],
        )

    def _seed_legacy(self):
        with Session(self.engine) as session:
            source = Source(
                name="Тестовый источник", url="https://example.org",
                source_type="rss",
            )
            cluster = Cluster(title="Новость", status="ready")
            session.add_all([source, cluster])
            session.flush()
            news = News(
                source_id=source.id, cluster_id=cluster.id,
                title="Завод запустил линию",
                text="Описание события", url="https://example.org/news/1",
            )
            post = Post(
                cluster_id=cluster.id, post_text="Заголовок\n\nСводка",
                status="draft",
            )
            session.add_all([news, post])
            session.commit()
            return news.id, post.id

    def test_does_not_apply_to_empty_database(self):
        with self.assertRaisesRegex(RuntimeError, "отсутствуют исходные таблицы"):
            apply_migration(self.engine)
        self.assertNotIn("moderation_items", inspect(self.engine).get_table_names())

    def test_migration_adds_only_new_tables_and_preserves_rows(self):
        self._legacy_tables()
        news_id, post_id = self._seed_legacy()
        apply_migration(self.engine)
        tables = set(inspect(self.engine).get_table_names())
        self.assertTrue({
            "news_images", "moderation_items",
            "moderation_actions", "publication_attempts"
        }.issubset(tables))

        with Session(self.engine) as session:
            self.assertEqual(session.query(News).count(), 1)
            self.assertEqual(session.query(Post).count(), 1)
            self.assertEqual(session.get(Post, post_id).status, "draft")

            image = NewsImage(
                news_id=news_id, source_url="https://example.org/photo.jpg",
            )
            session.add(image)
            session.flush()
            item = ModerationItem(
                post_id=post_id, snapshot_sha256="a" * 64, image_id=image.id
            )
            session.add(item)
            session.flush()
            session.add(
                ModerationAction(
                    moderation_item_id=item.id, moderator_telegram_id=12345678,
                    action="reject", details={"reason": "test"},
                )
            )
            session.add(PublicationAttempt(post_id=post_id, status="started"))
            session.commit()
            self.assertEqual(item.status, "pending")
            self.assertEqual(item.image_rights_decision, "pending")
            self.assertEqual(image.rights_status, "unknown")

    def test_one_main_image_per_news(self):
        self._legacy_tables()
        news_id, _ = self._seed_legacy()
        apply_migration(self.engine)
        with Session(self.engine) as session:
            session.add(NewsImage(news_id=news_id, source_url="https://example.org/1.jpg"))
            session.commit()
            session.add(NewsImage(news_id=news_id, source_url="https://example.org/2.jpg"))
            with self.assertRaises(IntegrityError):
                session.commit()
            session.rollback()

    def test_reapplying_migration_fails_safely(self):
        self._legacy_tables()
        apply_migration(self.engine)
        with self.assertRaisesRegex(RuntimeError, "уже существуют"):
            apply_migration(self.engine)


if __name__ == "__main__":
    unittest.main()
