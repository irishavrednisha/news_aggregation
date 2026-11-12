"""Database-level moderation tests (SQLite; no MAX/network activity)."""
import os
import unittest
from datetime import datetime
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.db import Base
from database.models import Cluster, Post
from database.moderation_models import ModerationItem, ModerationAction, NewsImage, PublicationAttempt
from moderation.service import snapshot_hash
from publishing.publisher import _claim_post, get_approved_posts, publish_approved_posts


class PublisherSafetyTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.session = sessionmaker(bind=self.engine)()
        cluster = Cluster(title="Test", status="ready")
        self.session.add(cluster)
        self.session.commit()
        self.cluster_id = cluster.id

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def add_post(self, status="draft", text="Заголовок\n\nСводка"):
        post = Post(
            cluster_id=self.cluster_id,
            post_text=text,
            status=status,
        )
        self.session.add(post)
        self.session.commit()
        if status == "approved":
            item = ModerationItem(
                post_id=post.id, status="approved",
                snapshot_sha256=snapshot_hash(post.post_text, None),
                image_rights_decision="no_image",
                moderator_telegram_id=12345,
                reviewed_at=datetime.utcnow(),
            )
            self.session.add(item)
            self.session.flush()
            self.session.add(ModerationAction(
                moderation_item_id=item.id,
                moderator_telegram_id=12345,
                action="approve",
            ))
            self.session.commit()
        return post

    def test_ready_cluster_and_draft_post_cannot_be_selected(self):
        self.add_post(status="draft")
        self.assertEqual(get_approved_posts(self.session), [])

    def test_only_valid_approved_post_is_selected(self):
        approved = self.add_post(status="approved")
        self.add_post(status="approved", text="only title")
        self.add_post(status="draft")
        self.assertEqual([p.id for p in get_approved_posts(self.session)], [approved.id])

    def test_post_marked_approved_without_moderation_is_ineligible(self):
        post = self.add_post(status="draft")
        post.status = "approved"
        self.session.commit()
        self.assertEqual(get_approved_posts(self.session), [])
        self.assertFalse(_claim_post(self.session, post.id))

    def test_max_disabled_does_not_even_open_a_db_session(self):
        self.add_post(status="approved")
        with patch.dict(os.environ, {"MAX_PUBLISHING_ENABLED": "false"}):
            with patch("publishing.publisher.get_session", side_effect=AssertionError("No DB access")):
                self.assertEqual(publish_approved_posts(), 0)

    def test_dry_run_never_changes_approval_or_opens_browser(self):
        approved = self.add_post(status="approved")
        post_id = approved.id
        with patch("publishing.publisher.get_session", return_value=self.session):
            with patch("publishing.publisher.sync_playwright", side_effect=AssertionError("No browser")):
                self.assertEqual(publish_approved_posts(dry_run=True), 1)
        # Reopen a fresh session to prove the persisted state is unchanged.
        check = sessionmaker(bind=self.engine)()
        try:
            self.assertEqual(check.get(Post, post_id).status, "approved")
        finally:
            check.close()

    def test_claim_is_atomic_and_cannot_be_repeated(self):
        approved = self.add_post(status="approved")
        self.assertTrue(_claim_post(self.session, approved.id))
        self.assertFalse(_claim_post(self.session, approved.id))
        self.session.refresh(approved)
        self.assertEqual(approved.status, "sending")
        self.assertEqual(get_approved_posts(self.session), [])


if __name__ == "__main__":
    unittest.main()
