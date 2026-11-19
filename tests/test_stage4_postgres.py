"""Stage 4 real Postgres persistence, disposable stage4_ci database ONLY."""
import os
import unittest

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from database.models import Cluster, News, Post, Source
from database.summary_models import SummaryRun
from processing.abstractive_summary import summarize_event_cluster
from scripts.run_stage4_summaries import _persist


class Stage4PostgresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        url = os.getenv("STAGE4_TEST_URL")
        if not url or os.getenv("STAGE4_TEST_WRITE_ACK") != "ephemeral-only":
            raise unittest.SkipTest("Requires explicit disposable Stage 4 Postgres")
        parsed = make_url(url)
        if parsed.get_backend_name() != "postgresql" or parsed.database != "stage4_ci":
            raise RuntimeError("Must use isolated database named stage4_ci")
        cls.engine = create_engine(url)
        with cls.engine.connect() as connection:
            revision = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
            if revision != "20261119_04":
                raise AssertionError("Stage 4 migration missing: " + revision)
            self_tables = set(inspect(connection).get_table_names())
            if "summary_runs" not in self_tables:
                raise AssertionError("Summary audit table missing")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine"):
            cls.engine.dispose()

    def test_append_only_audits_preserve_source_and_approval(self):
        with Session(bind=self.engine) as session:
            source = Source(name="Тестовый", url="https://example.org")
            session.add(source)
            session.flush()
            cluster = Cluster(title="Готовая новость", summary_text="APPROVED BASELINE",
                              status="ready", tags="машиностроение", cluster_size=1)
            session.add(cluster)
            session.flush()
            article = News(source_id=source.id, cluster_id=cluster.id,
                           title="Промышленное событие",
                           text="Источник сообщает о производственном событии. " * 8,
                           url="https://example.org/article", is_primary=True)
            post = Post(cluster_id=cluster.id, post_text="HUMAN APPROVED",
                        status="approved")
            session.add_all([article, post])
            session.commit()
            row = summarize_event_cluster(
                cluster.id, [{"id": article.id, "title": article.title,
                              "text": article.text, "url": article.url}], "",
                request=lambda *a, **kw: {"facts": []},
            ).to_dict()
            self.assertEqual(row["status"], "summary_review")
            _persist(session, row)
            _persist(session, row)
            session.expire_all()
            self.assertEqual(session.query(SummaryRun).filter_by(cluster_id=cluster.id).count(), 2)
            self.assertEqual(session.get(Cluster, cluster.id).summary_text, "APPROVED BASELINE")
            self.assertEqual(session.get(Post, post.id).post_text, "HUMAN APPROVED")
            self.assertEqual(session.get(Post, post.id).status, "approved")
            self.assertEqual(session.get(News, article.id).text, article.text)
            self.assertEqual(session.query(SummaryRun).first().source_materials[0]["text"],
                             article.text)


if __name__ == "__main__":
    unittest.main()
