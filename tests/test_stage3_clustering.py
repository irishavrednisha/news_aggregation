"""Stage 3 regressions; no SBERT download, no PostgreSQL or network required."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from processing.event_matching import (
    same_event, extract_facts, choose_existing_cluster, representative_key,
)
from scripts.audit_stage3 import run as audit_run


def row(title, *, date="2026-11-17", text="", url="https://example.org/article"):
    return SimpleNamespace(title=title, published_at=date, text=text, url=url)


class Stage3FactsTests(unittest.TestCase):
    def test_syndicated_osk_story_one_event(self):
        titles = [
            "ОСК усовершенствовала систему подготовки наставников",
            "ОСК улучшила систему подготовки наставников",
            "Объединенная судостроительная корпорация усовершенствовала систему подготовки наставников",
            "ОСК обновила систему подготовки наставников",
        ]
        for other in titles[1:]:
            decision = same_event(row(titles[0]), row(other), 0.94)
            self.assertTrue(decision.merge, (other, decision))

    def test_same_company_different_object_and_action(self):
        a = row("ЧМЗ выпустил партию автокранов 25 тонн")
        b = row("ЧМЗ построил новый цех для промышленного производства")
        self.assertEqual(same_event(a, b, 0.99).reason, "different_action")
        self.assertFalse(same_event(a, b, 0.99).merge)

    def test_different_quantities_do_not_merge(self):
        a = row("ЧМЗ выпустил партию автокранов 25 тонн")
        b = row("ЧМЗ изготовил партию автокранов 50 тонн")
        self.assertEqual(same_event(a, b, 0.97).reason, "different_quantity")

    def test_dates_repeated_batched_events(self):
        a = row("ЧМЗ выпустил партию автокранов 25 тонн",
                date="2026-10-01")
        b = row("ЧМЗ изготовил партию автокранов 25 тонн",
                date="2026-11-17")
        self.assertEqual(same_event(a, b, 0.98).reason, "publication_gap")

    def test_same_number_different_object(self):
        a = row("ЧМЗ выпустил партию автокранов 25 тонн")
        b = row("ЧМЗ выпустил партию станков 25 тонн")
        self.assertEqual(same_event(a, b, 0.98).reason, "different_object")

    def test_same_company_not_evidence(self):
        a = row("ОСК построила судно для морских перевозок")
        b = row("ОСК разработала двигатель для судна")
        self.assertEqual(same_event(a, b, 0.99).reason, "different_action")

    def test_different_company_same_object(self):
        a = row("ОСК разработала двигатель для судна")
        b = row("Ростех разработал двигатель для судна")
        self.assertEqual(same_event(a, b, 0.99).reason, "different_company")

    def test_real_seligdar_paraphrase(self):
        a = row('"Селигдар" достиг отметки в 8 тонн по производству золота',
                text='Холдинг Селигдар произвел золото, плановые показатели на Рябиновом.')
        b = row('8 тонн золота произвел «Селигдар» с начала 2026 года',
                text='Показатели месторождений Рябиновое и Самолазовское.')
        self.assertTrue(same_event(a, b, 0.842).merge)

    def test_real_kret_shared_technical_specifications(self):
        a = row("В России разработали компактные электродвигатели для дронов и роботов",
                text="Ростех, КРЭТ: бесколлекторные модели, 2500 Вт и 600 кг.")
        b = row("КРЭТ представил линейку ультралегких электродвигателей для БПЛА и робототехники",
                text="Концерн КРЭТ, входящий в Ростех: бесколлекторные модели, 2500 Вт и 600 кг.")
        self.assertTrue(same_event(a, b, 0.72759).merge)

    def test_low_similarity_gate(self):
        a = row("ЧМЗ выпустил партию автокранов")
        b = row("ЧМЗ изготовил партию автокранов")
        self.assertEqual(same_event(a, b, 0.4).reason, "low_embedding")

    def test_complete_link_guards_against_chaining(self):
        a = row("ЧМЗ выпустил партию автокранов 25 тонн")
        b = row("ЧМЗ изготовил партию автокранов 50 тонн")
        incoming = row("ЧМЗ изготовил партию автокранов 25 тонн")
        group = {"id": 1, "items": [(a, [1.0, 0.0]), (b, [1.0, 0.0])]}
        match, decision = choose_existing_cluster(incoming, [1.0, 0.0], [group])
        self.assertIsNone(match)
        self.assertEqual(decision.reason, "different_quantity")

    def test_informative_primary(self):
        a = row("ЧМЗ выпустил автокраны", text="Коротко.")
        b = row("ЧМЗ выпустил автокраны для заказчика",
                text="Детали производства. " * 120, url="https://example.org/full")
        self.assertEqual(min((a,b), key=representative_key), b)


class Stage3DatabaseTests(unittest.TestCase):
    def setUp(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from database.db import Base
        import database.models  # noqa
        import database.intake_models  # noqa: stage 2 raw-mirror compatibility
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session = sessionmaker(bind=self.engine)()
        from database.models import Source
        self.session.add(Source(name="TEST", url="https://test.example", source_type="html"))
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def add(self, title, text, url):
        from database.models import News, Source
        source = self.session.query(Source).first()
        instance = News(source_id=source.id, title=title, text=text,
                        url=url, published_at=datetime.utcnow(),
                        collected_at=datetime.utcnow())
        self.session.add(instance)
        self.session.commit()
        return instance

    def test_exact_mirror_source_urls_remain_available(self):
        from database.models import Cluster, Source
        from database.intake_models import NewsIntakeRecord
        from processing.clustering import get_cluster_source_links
        news = self.add("ОСК усовершенствовала систему подготовки наставников",
                        "Подробная статья", "https://test.example/original")
        cluster = Cluster(title=news.title, status="new", cluster_size=1)
        self.session.add(cluster)
        self.session.flush()
        news.cluster_id = cluster.id
        source = self.session.query(Source).first()
        mirror = NewsIntakeRecord(
            source_id=source.id, duplicate_of_news_id=news.id,
            raw_url="https://flotprom.ru/2026/Судостроение48/",
            raw_title=news.title, raw_text=news.text,
            canonical_url="https://flotprom.ru/2026/Судостроение48/",
            url_sha256="a" * 64, payload_sha256="b" * 64,
            status="rejected", reason="duplicate_text",
            decision_stage="exact_duplicate"
        )
        self.session.add(mirror)
        self.session.commit()
        links = get_cluster_source_links(self.session, cluster.id)
        self.assertEqual(len(links), 2)
        self.assertEqual(links[0]["link_type"], "accepted_event_article")
        self.assertEqual(links[1]["link_type"], "stage2_exact_copy")
        self.assertEqual(self.session.query(NewsIntakeRecord).count(), 1)

    def test_legacy_fulltext_embedding_cannot_block_true_event(self):
        from database.models import Cluster, News
        from processing.clustering import cluster_recent_news
        older = self.add("ЧМЗ выпустил партию автокранов 25 тонн",
                         "Текст компании " * 80, "https://test.example/legacy")
        # Before stage 3 this was encoded from a DIFFERENT representation.
        group = Cluster(title=older.title, status="new", cluster_size=1,
                        embedding=[0.0, 1.0])
        self.session.add(group)
        self.session.flush()
        older.cluster_id = group.id
        older.embedding = [0.0, 1.0]
        self.session.commit()
        incoming = self.add("Челябинский механический завод изготовил партию автокранов 25 тонн",
                            "Краткое описание события", "https://test.example/fresh")
        with patch("processing.clustering.build_embedding", return_value=[1.0, 0.0]):
            cluster_recent_news(self.session)
        self.assertEqual(incoming.cluster_id, older.cluster_id)
        self.assertEqual(self.session.get(Cluster, group.id).cluster_size, 2)

    def test_atomic_merge_primary_freeze_and_rerun(self):
        from database.models import News, Cluster, Post
        from processing.clustering import cluster_recent_news
        self.add("ЧМЗ выпустил партию автокранов 25 тонн", "Короткий текст", "https://test.example/1")
        self.add("Челябинский механический завод изготовил партию автокранов 25 тонн",
                 "Подробный материал о выпуске автокранов. " * 80, "https://test.example/2")
        self.add("ЧМЗ построил новый цех для производства автокранов",
                 "Новый цех для сборки автокранов." * 20, "https://test.example/3")
        with patch("processing.clustering.build_embedding", return_value=[1.0, 0.0]):
            cluster_recent_news(self.session, hours=48)
            groups = self.session.query(Cluster).order_by(Cluster.id).all()
            self.assertEqual(len(groups), 2)
            first = self.session.query(News).filter(News.url == "https://test.example/1").one()
            second = self.session.query(News).filter(News.url == "https://test.example/2").one()
            self.assertEqual(first.cluster_id, second.cluster_id)
            self.assertTrue(second.is_primary)
            original_title = self.session.get(Cluster, first.cluster_id).title
            post = Post(cluster_id=first.cluster_id, status="approved",
                        post_text="УТВЕРЖДЁННАЯ ВЕРСИЯ")
            self.session.add(post)
            self.session.commit()
            self.add("ЧМЗ изготовил партию автокранов 25 тонн",
                     "Детальная статья " * 400, "https://test.example/4")
            cluster_recent_news(self.session, hours=48)
            self.assertEqual(self.session.query(Cluster).count(), 2)
            self.assertEqual(self.session.get(Cluster, first.cluster_id).title, original_title)
            self.assertTrue(second.is_primary)
            self.assertEqual(self.session.get(Post, post.id).post_text,
                             "УТВЕРЖДЁННАЯ ВЕРСИЯ")
            count = self.session.query(News).count()
            cluster_recent_news(self.session, hours=48)
            self.assertEqual(self.session.query(News).count(), count)
            self.assertEqual(self.session.query(Cluster).count(), 2)
            self.assertEqual(self.session.get(Cluster, first.cluster_id).cluster_size, 3)


class Stage3AuditTests(unittest.TestCase):
    def test_read_only_replay_and_gold_pair_diagnostics(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "news.json"
            gold = Path(temp) / "gold.json"
            rows = [
                {"title": "ЧМЗ выпустил партию автокранов 25 тонн",
                 "url": "https://test.example/1", "text": "Машины для заказчика.",
                 "published_at": "2026-11-17", "status": "accepted"},
                {"title": "ЧМЗ изготовил партию автокранов 25 тонн",
                 "url": "https://test.example/2", "text": "Партия техники на заказ.",
                 "published_at": "2026-11-17", "status": "accepted"},
                {"title": "ЧМЗ построил новый цех для производства автокранов",
                 "url": "https://test.example/3", "text": "Производственный корпус.",
                 "published_at": "2026-11-17", "status": "accepted"},
            ]
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            gold.write_text(json.dumps({"pairs": [
                {"a": rows[0]["url"], "b": rows[1]["url"], "same_event": True},
                {"a": rows[0]["url"], "b": rows[2]["url"], "same_event": False},
            ]}), encoding="utf-8")
            before = path.read_bytes()
            first = audit_run(path, gold=gold, backend="lexical", min_decisions=3)
            second = audit_run(path, gold=gold, backend="lexical", min_decisions=3)
            self.assertEqual(first, second)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(first["db_writes"], 0)
            self.assertEqual(first["cluster_decisions_checked"], 3)
            self.assertEqual(first["gold_real_metrics"]["revised"]["counts"]["labeled_pairs"], 2)


if __name__ == "__main__":
    unittest.main()
