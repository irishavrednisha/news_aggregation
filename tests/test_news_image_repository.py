"""SQLite integration checks: saving image metadata must not lose the news."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from collectors.image_extractor import ImageCandidate
from database.db import Base
from database.models import Cluster, News, Post, Source
from database.moderation_models import NewsImage
from database.repository import save_news
from processing.image_downloader import DownloadedImage


class RepositoryImageTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(
            self.engine,
            tables=[
                Source.__table__, Cluster.__table__, News.__table__,
                Post.__table__, NewsImage.__table__,
            ],
        )
        self.session = sessionmaker(bind=self.engine)()
        self.session.add(Source(
            name="Тест", url="https://example.org",
            source_type="html", is_active=True,
        ))
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def article(self, url="https://example.org/news/1"):
        return SimpleNamespace(
            source="Тест", title="Завод построил цех", url=url,
            published_at=None, text="Промышленное событие",
            text_source="html",
            image_candidates=[ImageCandidate("https://example.org/photo.png")],
        )

    def test_disabled_images_still_save_news(self):
        with patch.dict(os.environ, {"NEWS_IMAGES_ENABLED": "false"}):
            row = save_news(self.session, self.article())
        self.assertIsNotNone(row)
        self.assertEqual(self.session.query(NewsImage).count(), 0)

    def test_image_metadata_saved_unknown_rights(self):
        sample = DownloadedImage(
            source_url="https://example.org/photo.png",
            local_path="data/news_images/test.png",
            mime_type="image/png", width=600, height=400, sha256="f" * 64,
        )
        with patch.dict(os.environ, {"NEWS_IMAGES_ENABLED": "true"}):
            with patch("processing.image_downloader.download_main_image",
                       return_value=sample):
                news = save_news(self.session, self.article())
        self.assertIsNotNone(news)
        saved = self.session.query(NewsImage).one()
        self.assertEqual(saved.news_id, news.id)
        self.assertEqual(saved.rights_status, "unknown")
        self.assertEqual(saved.local_path, sample.local_path)

    def test_image_exception_does_not_rollback_news(self):
        with patch.dict(os.environ, {"NEWS_IMAGES_ENABLED": "true"}):
            with patch("processing.image_downloader.download_main_image",
                       side_effect=ValueError("invalid image")):
                news = save_news(self.session, self.article())
        self.assertIsNotNone(news)
        self.assertEqual(self.session.query(News).count(), 1)
        self.assertEqual(self.session.query(NewsImage).count(), 0)


if __name__ == "__main__":
    unittest.main()
