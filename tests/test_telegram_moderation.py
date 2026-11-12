"""Moderation scenarios on disposable SQLite: NO Telegram/network/PostgreSQL/MAX."""
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from database.db import Base
from database.models import Cluster, News, Post, Source
from database.moderation_models import (
    ModerationAction, ModerationItem, NewsImage, PublicationAttempt,
)
from moderation.service import (
    create_pending_items, decide, get_item_preview, list_pending, valid_approval,
)
from publishing.publisher import get_approved_posts


class TelegramModerationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.cluster = Cluster(
            title="Открылся цех", summary_text="На предприятии открыт цех.",
            tags="машиностроение", status="ready",
        )
        self.session.add(self.cluster)
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def _new_item(self, image=False):
        if image:
            source = Source(name="Тест", url="https://example.org", source_type="rss")
            self.session.add(source)
            self.session.flush()
            news = News(
                source_id=source.id, cluster_id=self.cluster.id,
                title="Открылся цех", text="Открыт цех.",
                url="https://example.org/news/1", is_primary=True,
            )
            self.session.add(news)
            self.session.flush()
            self.session.add(NewsImage(
                news_id=news.id, source_url="https://example.org/1.png",
                local_path="data/news_images/1.png", rights_status="unknown",
            ))
            self.session.commit()
        self.assertEqual(create_pending_items(self.session), 1)
        self.assertEqual(create_pending_items(self.session), 0)
        return self.session.query(ModerationItem).one()

    def test_text_only_can_be_approved_with_real_audit(self):
        item = self._new_item()
        self.assertEqual(item.image_rights_decision, "no_image")
        post = self.session.get(Post, item.post_id)
        self.assertEqual(post.status, "draft")
        self.assertEqual(get_approved_posts(self.session), [])
        self.assertIn("Одобрено", decide(self.session, item.id, "approve", 555))
        self.assertTrue(valid_approval(self.session, post))
        self.assertEqual([p.id for p in get_approved_posts(self.session)], [post.id])
        self.assertEqual(
            self.session.query(ModerationAction).filter_by(action="approve").count(), 1
        )
        self.assertIn("уже обработан", decide(self.session, item.id, "approve", 555))
        self.assertEqual(
            self.session.query(ModerationAction).filter_by(action="approve").count(), 1
        )

    def test_image_requires_rights_or_removal_before_approval(self):
        item = self._new_item(image=True)
        self.assertIn("Сначала", decide(self.session, item.id, "approve", 555))
        self.assertIn("исключено", decide(self.session, item.id, "remove_image", 555))
        self.assertIn("Одобрено", decide(self.session, item.id, "approve", 555))
        post = self.session.get(Post, item.post_id)
        self.assertTrue(valid_approval(self.session, post))

    def test_approved_image_waits_for_image_capable_max_publisher(self):
        item = self._new_item(image=True)
        self.assertIn("Отмечено", decide(self.session, item.id, "rights_confirmed", 555))
        self.assertIn("Одобрено", decide(self.session, item.id, "approve", 555))
        post = self.session.get(Post, item.post_id)
        self.assertFalse(valid_approval(self.session, post))
        self.assertTrue(valid_approval(self.session, post, require_text_only=False))
        self.assertEqual(get_approved_posts(self.session), [])

    def test_rejection_blocks_requeue(self):
        item = self._new_item()
        self.assertIn("Отклонено", decide(self.session, item.id, "reject", 555))
        self.assertEqual(create_pending_items(self.session), 0)
        self.assertEqual(list_pending(self.session), [])
        self.assertEqual(get_approved_posts(self.session), [])

    def test_modified_post_invalidates_approval(self):
        item = self._new_item()
        post = self.session.get(Post, item.post_id)
        post.post_text = "Другая версия\n\nИзменённый текст"
        self.session.commit()
        self.assertIn("изменился", decide(self.session, item.id, "approve", 555))
        self.assertFalse(valid_approval(self.session, post))

    def test_invalid_moderator_not_allowed(self):
        item = self._new_item()
        self.assertIn("Недействительный", decide(self.session, item.id, "approve", 0))
        self.assertEqual(self.session.get(ModerationItem, item.id).status, "pending")

    def test_queue_tracks_delivered_messages(self):
        item = self._new_item()
        self.assertEqual(len(list_pending(self.session)), 1)
        item.telegram_message_id = 102
        self.session.commit()
        self.assertEqual(list_pending(self.session), [])


if __name__ == "__main__":
    unittest.main()
