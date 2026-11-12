"""Database-level moderation tests (SQLite; no MAX/network activity)."""
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.db import Base
from database.models import Cluster, Post
from publishing.publisher import _claim_post, get_approved_posts


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
        return post

    def test_ready_cluster_and_draft_post_cannot_be_selected(self):
        self.add_post(status="draft")
        self.assertEqual(get_approved_posts(self.session), [])

    def test_only_valid_approved_post_is_selected(self):
        approved = self.add_post(status="approved")
        self.add_post(status="approved", text="only title")
        self.add_post(status="draft")
        self.assertEqual([p.id for p in get_approved_posts(self.session)], [approved.id])

    def test_claim_is_atomic_and_cannot_be_repeated(self):
        approved = self.add_post(status="approved")
        self.assertTrue(_claim_post(self.session, approved.id))
        self.assertFalse(_claim_post(self.session, approved.id))
        self.session.refresh(approved)
        self.assertEqual(approved.status, "sending")
        self.assertEqual(get_approved_posts(self.session), [])


if __name__ == "__main__":
    unittest.main()
