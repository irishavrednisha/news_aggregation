"""Stage 2 independently labeled controls + SQLite, rerun and mirror regressions."""
from __future__ import annotations

from datetime import date, datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from processing.news_intake import classify_intake, technical_decision, semantic_decision
from processing.exact_duplicates import (
    ExactDuplicateIndex, match_exact, normalize_url, normalize_title,
)

TODAY = date(2026, 11, 18)
BODY = (
    "На российском заводе инженеры внедрили цифровую систему контроля качества "
    "и испытали новые технологии для обработки металлов. Предприятие модернизировало "
    "промышленное оборудование, увеличило скорость обработки сырья, открыло участок "
    "производства деталей для двигателей и провело испытания экспериментального "
    "оборудования. Измерения подтвердили производительность комплекса. "
    "Научный коллектив подготовил техническую документацию для дальнейшей "
    "эксплуатации и передал результаты специалистам. "
)
BOILER = (("Политика конфиденциальности. Все права защищены. " * 12))


def article(title, *, url="https://sample.example/news/a", source="MASHNEWS",
            text=BODY, published="2026-11-18", text_source="html"):
    return SimpleNamespace(title=title, url=url, source=source, text=text,
                           published_at=published, text_source=text_source)


def from_gold(case):
    kinds = {"full": BODY, "short": "На заводе испытали станок.",
             "empty": "", "boiler": BOILER, "rss": BODY}
    return article(case["title"], url=case["url"], text=kinds[case["body_kind"]],
                   published=case["published_at"],
                   text_source="rss" if case["body_kind"] == "rss" else "html")


class CuratedBoundaryTests(unittest.TestCase):
    def test_50_labeled_boundary_cases(self):
        fixture = Path(__file__).parent / "fixtures" / "stage2_gold_cases.json"
        cases = json.loads(fixture.read_text(encoding="utf-8"))["cases"]
        self.assertGreaterEqual(len(cases), 30)
        for case in cases:
            with self.subTest(case=case["id"], title=case["title"]):
                predicted = classify_intake(from_gold(case), now=TODAY)
                self.assertEqual(predicted.status, case["expected_status"])
                self.assertEqual(predicted.reason, case["expected_reason"])

    def test_no_russia_only_rejection(self):
        value = article("В Китае построили новую фабрику")
        self.assertEqual(classify_intake(value, now=TODAY).status, "accepted")

    def test_review_short_and_rss_not_silently_deleted(self):
        for kwargs, reason in (
            ({"text": "Важное сообщение"}, "short_text"),
            ({"text_source": "rss"}, "rss_preview_only"),
        ):
            with self.subTest(reason=reason):
                decision = classify_intake(article("Завод запустил новую линию", **kwargs), now=TODAY)
                self.assertEqual((decision.status, decision.reason), ("review", reason))

    def test_production_mentioned_in_ad_is_still_ad(self):
        v = article("Скидки на промышленное оборудование для заводов")
        self.assertEqual(classify_intake(v, now=TODAY).reason, "advertisement")

    def test_decisions_are_pure_and_repeatable(self):
        v = article("Предприятие внедрило новую производственную линию")
        before = dict(vars(v))
        first = classify_intake(v, now=TODAY)
        second = classify_intake(v, now=TODAY)
        self.assertEqual(first, second)
        self.assertEqual(vars(v), before)


class ExactDuplicateTests(unittest.TestCase):
    def test_canonical_url_strips_trackers_and_fragments(self):
        left = "http://WWW.Example.Org/news/a/?utm_source=mail&obj=42&gclid=x#image"
        right = "https://example.org/news/a?obj=42"
        self.assertEqual(normalize_url(left), normalize_url(right))

    def test_canonical_url_keeps_nedradv_obj_and_other_content_parameters(self):
        one = "https://nedradv.ru/nedradv/ru/news?obj=115e"
        two = "https://nedradv.ru/nedradv/ru/news?obj=115f"
        self.assertNotEqual(normalize_url(one), normalize_url(two))
        self.assertIn("obj=115e", normalize_url(one))

    def test_idn_domains_and_aliases(self):
        a = normalize_url("https://военное.рф/2026/11/18/new-ship/")
        b = normalize_url("https://xn--b1aga5aadd.xn--p1ai/2026/11/18/new-ship")
        self.assertEqual(a, b)

    def test_trailing_slash_and_query_order(self):
        self.assertEqual(
            normalize_url("https://example.org/n/?b=2&a=1"),
            normalize_url("https://example.org/n?a=1&b=2"),
        )

    def test_same_url_different_tracking_is_duplicate(self):
        a = article("Завод запустил линию", url="https://example.org/n?id=33")
        b = article("Завод изменил линию", url="https://example.org/n?utm_source=news&id=33")
        self.assertEqual(match_exact(a, b), "duplicate_url")

    def test_cross_domain_mirror_article_path(self):
        a = article("Верфь построила новый корабль", url="https://военное.рф/2026/11/18/unique-article-name")
        b = article("Верфь построила новый корабль", url="https://flotprom.ru/2026/11/18/unique-article-name")
        self.assertEqual(match_exact(a, b), "duplicate_mirror")

    def test_mirrors_unrelated_path_not_equal(self):
        a = article("Верфь построила корабль", url="https://flot.com/2026/11/18/unique-one")
        b = article("Верфь построила корабль", url="https://flotprom.ru/2026/11/18/unique-two",
                    text="Совершенно другой отчёт об аварии на порту и закупке старого судна. " * 8)
        self.assertIsNone(match_exact(a, b))

    def test_exact_full_text_duplicate_different_headline(self):
        a = article("Завод запустил новый цех", url="https://s1.example/news/a")
        b = article("На предприятии начали выпуск продукции", url="https://s2.example/news/b")
        self.assertEqual(match_exact(a, b), "duplicate_text")

    def test_equal_generic_headline_different_events_not_merged(self):
        a = article("Завод запустил новую линию", url="https://a.example/news/1")
        b = article("Завод запустил новую линию", url="https://b.example/news/2",
                    text="На предприятии совсем другого региона открыта линия по сборке турбин. " * 8)
        self.assertIsNone(match_exact(a, b))

    def test_equal_body_repeated_next_year_not_merged(self):
        a = article("Завод запустил новую линию", published="2025-11-01")
        b = article("Завод запустил новую линию", url="https://example.org/news/b",
                    published="2026-11-01")
        self.assertIsNone(match_exact(a, b))

    def test_headline_normalization(self):
        self.assertEqual(normalize_title("«Завод» — запустил!"), normalize_title("ЗАВОД запустил"))

    def test_index_returns_canonical_news_id(self):
        index = ExactDuplicateIndex()
        one = article("Завод запустил линию", url="https://example.org/news/abc")
        one.id = 9
        index.add(one)
        duplicate = article("Завод запустил линию", url="https://example.org/news/abc/?utm_source=site")
        match = index.find(duplicate)
        self.assertEqual((match.news_id, match.reason), (9, "duplicate_url"))


class DatabaseIntakeTests(unittest.TestCase):
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
        self.Record = NewsIntakeRecord
        for name in ("MASHNEWS", "ФлотПром"):
            self.session.add(Source(name=name, url="https://example.org",
                                    source_type="html", is_active=True))
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def test_store_accept_review_reject_and_exact_copy(self):
        from database.intake_repository import save_intake_batch
        first = article("Завод запустил новый цех", url="https://first.example/a")
        review = article("Завод обсуждает планы", url="https://first.example/b")
        rejected = article("Скидки на товары", url="https://first.example/c")
        copy = article("Завод запустил новый цех", source="ФлотПром",
                       url="https://second.example/a")
        with patch.dict(os.environ, {"NEWS_IMAGES_ENABLED": "false"}):
            stats = save_intake_batch(self.session, [first, review, rejected, copy], now=TODAY)
        self.assertEqual(stats.saved, 1)
        self.assertEqual(self.session.query(self.News).count(), 1)
        records = self.session.query(self.Record).order_by(self.Record.id).all()
        self.assertEqual([x.status for x in records],
                         ["accepted", "review", "rejected", "duplicate"])
        self.assertEqual(records[-1].duplicate_of_news_id, records[0].news_id)
        self.assertEqual(records[-1].reason, "duplicate_text")

    def test_idempotent_double_run_no_new_news(self):
        from database.intake_repository import save_intake_batch
        sample = article("Завод запустил новый цех")
        with patch.dict(os.environ, {"NEWS_IMAGES_ENABLED": "false"}):
            first = save_intake_batch(self.session, [sample], now=TODAY)
            second = save_intake_batch(self.session, [sample], now=TODAY)
        self.assertEqual((first.saved, second.saved), (1, 0))
        self.assertEqual(self.session.query(self.News).count(), 1)
        self.assertEqual(self.session.query(self.Record).count(), 1)
        record = self.session.query(self.Record).one()
        self.assertEqual(record.seen_count, 2)

    def test_canonical_variants_have_distinct_raw_snapshots(self):
        from database.intake_repository import save_intake_batch
        a = article("Завод запустил новый цех",
                    url="https://example.org/n?id=7&utm_source=one")
        b = article("Завод запустил новый цех",
                    url="https://example.org/n?utm_source=two&id=7")
        with patch.dict(os.environ, {"NEWS_IMAGES_ENABLED": "false"}):
            result = save_intake_batch(self.session, [a, b], now=TODAY)
        self.assertEqual(result.saved, 1)
        self.assertEqual(self.session.query(self.Record).count(), 2)
        self.assertEqual(self.session.query(self.News).count(), 1)

    def test_legacy_news_index_is_honored_without_schema_recreation(self):
        from database.intake_repository import save_intake_batch
        source = self.session.query(__import__("database.models", fromlist=["Source"]).Source).first()
        old = self.News(source_id=source.id, url="https://example.org/old",
                        text=BODY, title="Завод запустил новый цех",
                        published_at=datetime(2026, 11, 18))
        self.session.add(old)
        self.session.commit()
        sample = article("Завод запустил новый цех",
                         url="https://example.org/old?utm_source=rss")
        with patch.dict(os.environ, {"NEWS_IMAGES_ENABLED": "false"}):
            stats = save_intake_batch(self.session, [sample], now=TODAY)
        self.assertEqual(stats.saved, 0)
        self.assertEqual(self.session.query(self.News).count(), 1)
        self.assertEqual(self.session.query(self.Record).one().duplicate_of_news_id, old.id)


if __name__ == "__main__":
    unittest.main()
