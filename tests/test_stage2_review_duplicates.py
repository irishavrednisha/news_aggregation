"""Regression: exact copies of review/rejected reports across publisher mirrors.

The original remains auditable, and review cannot suppress later acceptance.
"""
import os
from datetime import date
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from processing.exact_duplicates import normalize_url
from scripts.audit_stage2 import audit_real

TODAY = date(2026, 11, 18)
TEXT = ("Специалисты описали технологический комплекс для предприятий, "
        "сведения о технических условиях внедрения и практических испытаниях. "
        "Отдельно приводятся результаты оценки производственных решений. ") * 5


def article(title, url, source="ФлотПром", text=TEXT):
    return SimpleNamespace(source=source, url=url, title=title, text=text,
                           published_at="2026-11-17", text_source="html")


class Stage2RawDuplicateAudit(unittest.TestCase):
    def test_read_only_includes_nonpromoted_mirror_duplicates(self):
        title = "ОСК усовершенствовала систему подготовки наставников"
        first = article(title, "https://flotprom.ru/2026/industry51")
        second = article(title, "https://flot.com/2026/industry51", "ФлотКом")
        third = article("Распродажа техники для покупателей",
                        "https://flot.com/2026/promo52", "ФлотКом")
        fourth = article("Распродажа техники для покупателей",
                         "https://flotprom.ru/2026/promo52")
        result = audit_real([first, second, third, fourth], today=TODAY)
        rows = result["articles"]
        self.assertEqual([r["status"] for r in rows],
                         ["review", "duplicate", "rejected", "duplicate"])
        self.assertEqual([r["reason"] for r in rows if r["status"] == "duplicate"],
                         ["duplicate_text", "duplicate_text"])
        self.assertEqual(rows[1]["duplicate_of_url"], normalize_url(first.url))
        self.assertEqual(rows[3]["duplicate_of_url"], normalize_url(third.url))
        self.assertEqual(result["repeatability"]["replay_new_news"], 0)

    def test_different_intake_events_not_merged(self):
        a = article("ОСК обсудила подготовку наставников",
                    "https://flotprom.ru/2026/industry51")
        b = article("ОСК обсудила подготовку наставников",
                    "https://flot.com/2026/industry51", "ФлотКом",
                    text=TEXT.replace("практических", "лабораторных") +
                         " В этом году речь идет о другом объекте.")
        result = audit_real([a, b], today=TODAY)
        self.assertEqual([r["status"] for r in result["articles"]],
                         ["review", "review"])


class Stage2RawDuplicatePersistence(unittest.TestCase):
    def setUp(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session
        from database.db import Base
        from database.models import Source, News
        from database.intake_models import NewsIntakeRecord
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        Base.metadata.create_all(self.engine, tables=[
            Source.__table__, News.__table__, NewsIntakeRecord.__table__,
        ])
        self.session = Session(bind=self.engine)
        self.News = News
        self.Intake = NewsIntakeRecord
        for name in ("ФлотПром", "ФлотКом"):
            self.session.add(Source(name=name, url="https://example.org",
                                    source_type="html", is_active=True))
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def test_review_and_rejected_exact_mirrors_are_linked(self):
        from database.intake_repository import save_intake_batch
        title = "ОСК усовершенствовала систему подготовки наставников"
        arr = [
            article(title, "https://flotprom.ru/2026/industry51"),
            article(title, "https://flot.com/2026/industry51", "ФлотКом"),
            article("Распродажа техники", "https://flot.com/2026/promo52", "ФлотКом"),
            article("Распродажа техники", "https://flotprom.ru/2026/promo52"),
        ]
        with patch.dict(os.environ, {"NEWS_IMAGES_ENABLED": "false"}):
            stats = save_intake_batch(self.session, arr, now=TODAY)
        self.assertEqual(stats.saved, 0)
        records = self.session.query(self.Intake).order_by(self.Intake.id).all()
        self.assertEqual([r.status for r in records],
                         ["review", "duplicate", "rejected", "duplicate"])
        self.assertEqual(records[1].duplicate_of_intake_id, records[0].id)
        self.assertEqual(records[3].duplicate_of_intake_id, records[2].id)
        self.assertEqual(self.session.query(self.News).count(), 0)
        # The original text and exact source/URL are not destroyed.
        self.assertEqual(records[1].raw_text, TEXT)
        self.assertEqual(records[1].raw_url, arr[1].url)
        with patch.dict(os.environ, {"NEWS_IMAGES_ENABLED": "false"}):
            again = save_intake_batch(self.session, arr, now=TODAY)
        self.assertEqual(again.saved, 0)
        self.assertEqual(self.session.query(self.Intake).count(), 4)
        self.assertEqual([r.seen_count for r in records], [2, 2, 2, 2])

    def test_existing_review_never_blocks_new_accepted_news(self):
        from database.intake_repository import save_intake_batch
        r = article("ОСК усовершенствовала систему подготовки наставников",
                    "https://flot.com/2026/industry51", "ФлотКом")
        accepted = article("Верфь построила новый ледокол",
                           "https://flotprom.ru/2026/industry51")
        with patch.dict(os.environ, {"NEWS_IMAGES_ENABLED": "false"}):
            result = save_intake_batch(self.session, [r, accepted], now=TODAY)
        self.assertEqual(result.saved, 1)
        self.assertEqual(self.session.query(self.News).count(), 1)
        self.assertEqual([x.status for x in self.session.query(self.Intake).order_by(self.Intake.id)],
                         ["review", "accepted"])


if __name__ == "__main__":
    unittest.main()
