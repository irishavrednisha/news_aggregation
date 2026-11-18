"""Offline stability contracts: no ML on collection path, isolation, TLS and reporting."""
from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from collectors.article_quality import is_error_page_text
from collectors.data_collector import Article, fetch_html_with_requests
from collectors.tls_policy import ca_bundle_for_url
from scripts.diagnose_all_sources import article_metrics, inspect_source


class Stage1StabilityTests(unittest.TestCase):
    def test_main_import_does_not_import_ml(self):
        script = (
            "import sys; import main; "
            "assert not any(x in sys.modules for x in "
            "('sklearn', 'torch', 'sentence_transformers')); "
            "print('COLLECTOR_IMPORT_OK')"
        )
        process = subprocess.run([sys.executable, "-c", script],
                                 capture_output=True, text=True, timeout=12)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("COLLECTOR_IMPORT_OK", process.stdout)

    def test_collection_continues_after_source_failure(self):
        import main
        sources = [SimpleNamespace(name="Broken", source_type="html"),
                   SimpleNamespace(name="Working", source_type="html")]
        session = MagicMock()
        good = Article(
            source="Working", title="Открыт новый российский промышленный завод",
            url="https://example.org/news/factory", published_at=datetime(2026, 11, 18),
            text="Подробная информация о работе нового промышленного предприятия. " * 6,
            text_source="html",
        )
        with patch("database.db.get_session", return_value=session), \
             patch("database.repository.get_active_sources", return_value=sources), \
             patch("collectors.data_collector.collect_news",
                   side_effect=[RuntimeError("unavailable"), [good]]) as collected, \
             patch("processing.topic_filter.filter_articles", side_effect=lambda a: a), \
             patch("database.repository.save_articles", return_value=1):
            saved = main.collect(limit=1)
        self.assertEqual(saved, 1)
        self.assertEqual(collected.call_count, 2)
        session.rollback.assert_called()
        session.close.assert_called_once()

    def test_normal_collection_uses_browser_warning_only_for_three_sources(self):
        import main
        names = ("Ростех", "Роснефть", "Газпром", "ИАПН", "Норильский никель")
        sources = [SimpleNamespace(name=name, source_type="html") for name in names]
        with patch("database.db.get_session", return_value=MagicMock()), \
             patch("database.repository.get_active_sources", return_value=sources), \
             patch("collectors.data_collector.collect_news", return_value=[]) as collector, \
             patch("processing.topic_filter.filter_articles", side_effect=lambda items: items), \
             patch("database.repository.save_articles", return_value=0), \
             patch.dict(os.environ, {"NEWS_BROWSER_CERT_WARNING_FALLBACK": "1"}):
            main.collect(limit=1)
        self.assertEqual(collector.call_count, len(sources))
        flags = {
            names[i]: call.kwargs["allow_unverified_tls"]
            for i, call in enumerate(collector.call_args_list)
        }
        self.assertEqual(flags, {
            "Ростех": True, "Роснефть": True, "Газпром": True,
            "ИАПН": False, "Норильский никель": False,
        })

    def test_certificate_warning_can_be_disabled_with_environment(self):
        import main
        for setting in ("0", "false", "off"):
            with self.subTest(setting=setting), \
                 patch.dict(os.environ, {"NEWS_BROWSER_CERT_WARNING_FALLBACK": setting}):
                self.assertFalse(main._browser_cert_warning_enabled())

    def test_untrusted_browser_never_crosses_official_site_boundaries(self):
        from collectors.data_collector import _cert_warning_same_source
        self.assertTrue(_cert_warning_same_source(
            "https://www.rosneft.ru/press/news/", "https://limited.rosneft.ru/press/news/2026/",
        ))
        self.assertTrue(_cert_warning_same_source(
            "https://rostec.ru/media/news/", "https://www.rostec.ru/media/file.jpg",
        ))
        for url in (
            "https://rostec.ru.evil.invalid/steal",
            "https://www.rosneft.ru/press/news/",
            "https://cdn.rostec.ru/file.js",
            "http://rostec.ru/media/news/",
            "https://user:password@rostec.ru/media/news/",
            "file:///etc/passwd",
        ):
            with self.subTest(url=url):
                self.assertFalse(_cert_warning_same_source(
                    "https://rostec.ru/media/news/", url,
                ))

    def test_error_pages_never_count_as_news(self):
        for text in (
            "",
            "Error establishing a database connection",
            "Access Denied",
            "Just a moment...",
            "Please enable JavaScript to view this page",
        ):
            self.assertTrue(is_error_page_text(text), text)
        self.assertFalse(is_error_page_text(
            "Инженеры запустили новое производство компонентов для отечественных машин. " * 12
        ))

    def test_requests_always_starts_with_verified_tls(self):
        from requests.exceptions import SSLError
        page = MagicMock()
        page.status_code = 200
        page.encoding = "utf-8"
        page.text = "<html><article>verified from system</article></html>"
        with patch("collectors.data_collector.requests.get",
                   side_effect=SSLError("untrusted")) as req, \
             patch("collectors.data_collector.retry_with_trusted_roots",
                   return_value=page) as trusted:
            result = fetch_html_with_requests("https://www.gazprom.ru/press/news/")
        self.assertIn("system", result)
        self.assertTrue(req.call_args.kwargs["verify"])
        trusted.assert_called_once()

    def test_ca_bundle_matches_exact_host_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            ca = Path(tmp) / "root.pem"
            ca.write_text("test", encoding="utf-8")
            with patch.dict(os.environ, {
                "NEWS_SOURCE_CA_BUNDLES": json.dumps({"www.rostec.ru": str(ca)}),
                "NEWS_CA_BUNDLE": "",
            }):
                self.assertEqual(ca_bundle_for_url("https://www.rostec.ru/media/news/"), str(ca))
                self.assertIsNone(ca_bundle_for_url("https://www.rostec.ru.attacker.test/news"))
                self.assertIsNone(ca_bundle_for_url("https://rostec.ru/media/news"))
                self.assertIsNone(ca_bundle_for_url("https://user:pass@www.rostec.ru/media/news"))

    def test_diagnostics_reports_date_title_image_and_extraction(self):
        source = SimpleNamespace(id=7, name="Example", source_type="html",
                                 url="https://example.com", rss_url=None, is_active=False)
        sample = Article(
            source="Example", title="Новая установка заработала на промышленном заводе",
            url="https://example.com/article", text="Новые технологии применены на заводе. " * 20,
            published_at=datetime(2026, 11, 18), text_source="html",
        )
        with patch("collectors.data_collector.collect_news", return_value=[sample]) as fetch:
            result = inspect_source(source, limit=2)
        self.assertEqual(result["source_id"], 7)
        self.assertFalse(result["configured_active"])
        self.assertEqual(result["articles_extracted"], 1)
        self.assertEqual(result["full_html_texts_250_plus"], 1)
        self.assertEqual(result["valid_dates"], 1)
        self.assertEqual(result["articles_with_image_candidates"], 0)
        self.assertIsNone(fetch.call_args.kwargs["session"])
        self.assertFalse(fetch.call_args.kwargs["apply_filters"])

    def test_empty_html_or_rss_preview_is_not_complete_html_text(self):
        rss = Article("Example", "Промышленное строительство в России",
                      "https://example.com/n", datetime(2026, 11, 18),
                      "Предварительный анонс выпуска нового оборудования. " * 6,
                      "rss")
        self.assertFalse(article_metrics(rss)["full_text_heuristic"])


    def test_rss_excerpt_is_not_saved_as_a_complete_article(self):
        from collectors.data_collector import collect_news, ArticlePreview
        source = SimpleNamespace(name="ТАСС", source_type="rss",
                                 url="https://tass.ru/", rss_url="https://tass.ru/rss/v2.xml")
        preview = ArticlePreview(source="ТАСС", title="Российский завод изготовил новое оборудование",
                                 url="https://tass.ru/example", published_at=None,
                                 preview_text="Текст из RSS")
        article = Article(source="ТАСС", title=preview.title, url=preview.url,
                          published_at=None, text="Длинное описание из RSS. " * 30,
                          text_source="rss")
        with patch("collectors.data_collector.collect_rss_previews", return_value=[preview]), \
             patch("collectors.data_collector.build_full_article", return_value=article), \
             patch("collectors.data_collector.time.sleep"):
            diagnostic = {}
            raw = collect_news([source], limit_per_source=1, diagnostics=diagnostic,
                               apply_filters=False)
            self.assertEqual(len(raw), 1)
            self.assertEqual(diagnostic["ТАСС"]["status"], "rss_preview_only")
            self.assertEqual(diagnostic["ТАСС"]["full_html_accepted"], 0)
            stored = collect_news([source], limit_per_source=1, apply_filters=True)
            self.assertEqual(stored, [])
            self.assertFalse(article_metrics(article)["full_text_heuristic"])

    def test_browser_first_source_has_verified_requests_fallback(self):
        from collectors.data_collector import fetch_html
        html = "<html><body><h1>Новости Торгово-промышленной палаты</h1>" + (
            "<p>Промышленное оборудование произведено на отечественном заводе.</p>" * 5
        ) + "</body></html>"
        with patch("collectors.data_collector.fetch_html_with_browser",
                   return_value="") as browser, \
             patch("collectors.data_collector.fetch_html_with_requests",
                   return_value=html) as requests_mock:
            text = fetch_html("https://news.tpprf.ru/ru/main/")
        self.assertEqual(text, html)
        browser.assert_called_once()
        requests_mock.assert_called_once()


    def test_second_run_does_not_duplicate_news_or_erase_prior_insert(self):
        """Use real SQLAlchemy/SQLite constraints, no PostgreSQL mutation."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from database.db import Base
        from database.models import Cluster, News, Source
        from database.repository import save_articles

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(
            bind=engine,
            tables=[Source.__table__, Cluster.__table__, News.__table__],
        )
        session = sessionmaker(bind=engine)()
        try:
            session.add(Source(
                name="Example", url="https://example.org", source_type="html",
                is_active=True,
            ))
            session.commit()
            first = Article(
                source="Example", title="Российское предприятие запустило завод",
                url="https://example.org/n1",
                published_at=datetime(2026, 11, 18),
                text="Новое производственное оборудование. " * 20,
                text_source="html",
            )
            second = Article(
                source="Example", title="Российское предприятие расширило завод",
                url="https://example.org/n2",
                published_at=datetime(2026, 11, 18),
                text="Производственная мощность нового оборудования. " * 20,
                text_source="html",
            )
            with patch.dict(os.environ, {"NEWS_IMAGES_ENABLED": "false"}):
                self.assertEqual(save_articles(session, [first]), 1)
                self.assertEqual(save_articles(session, [first, second]), 1)
            self.assertEqual(session.query(News).count(), 2)
            self.assertEqual({n.url for n in session.query(News).all()},
                             {first.url, second.url})
        finally:
            session.close()
            engine.dispose()


    def test_iapn_tries_configured_rss_when_html_listing_fails(self):
        from collectors.data_collector import ArticlePreview, collect_news
        source = SimpleNamespace(
            name="ИАПН", source_type="html", url="https://iapn.ru",
            rss_url="https://iapn.ru/feed/", is_active=True,
        )
        preview = ArticlePreview(
            source="ИАПН", title="Российский завод выпустил новую линию станков",
            url="https://iapn.ru/article-123/", published_at=datetime(2026, 11, 18),
            preview_text="",
        )
        article = Article(
            source="ИАПН", title=preview.title, url=preview.url,
            published_at=preview.published_at, text="На заводе запустили станки. " * 18,
            text_source="html",
        )
        diagnostics = {}
        with patch("collectors.data_collector.collect_html_previews", return_value=[]), \
             patch("collectors.data_collector.collect_rss_previews", return_value=[preview]) as rss, \
             patch("collectors.data_collector.build_full_article", return_value=article), \
             patch("collectors.data_collector.time.sleep"):
            collected = collect_news(
                [source], limit_per_source=1, diagnostics=diagnostics,
                apply_filters=False,
            )
        self.assertEqual(len(collected), 1)
        self.assertTrue(diagnostics["ИАПН"]["rss_fallback_attempted"])
        self.assertTrue(diagnostics["ИАПН"]["rss_fallback_succeeded"])
        self.assertEqual(diagnostics["ИАПН"]["status"], "ok")
        rss.assert_called_once()

    def test_rss_fetch_uses_verified_tls_first_and_truststore_on_failure(self):
        from requests.exceptions import SSLError
        from collectors.data_collector import collect_rss_previews
        source = SimpleNamespace(
            name="ИАПН", url="https://iapn.ru", rss_url="https://iapn.ru/feed/",
        )
        feed_xml = """<?xml version="1.0" encoding="utf-8"?>
          <rss version="2.0"><channel><title>News</title><item>
          <title>Российский завод ввёл новую производственную линию</title>
          <link>https://iapn.ru/test-industrial-news/</link>
          <description>Российский производитель оборудования</description>
          </item></channel></rss>""".encode("utf-8")
        response = MagicMock()
        response.content = feed_xml
        response.raise_for_status.return_value = None
        with patch("collectors.data_collector.requests.get",
                   side_effect=SSLError("untrusted CA")) as request, \
             patch("collectors.data_collector.retry_with_trusted_roots",
                   return_value=response) as trusted:
            previews = collect_rss_previews(source, 2, apply_filters=False)
        self.assertEqual(len(previews), 1)
        self.assertEqual(previews[0].url, "https://iapn.ru/test-industrial-news/")
        self.assertIs(request.call_args.kwargs["verify"], True)
        trusted.assert_called_once()

    def test_diagnostics_does_not_call_old_news_fresh(self):
        old_news = Article(
            source="Вестник промышленности",
            title="Модернизацию машиностроения обсудят летом в Казани",
            url="https://vestnikprom.ru/2026/04/21/industry/",
            published_at=datetime(2026, 4, 20),
            text="Выпуск промышленных машин. " * 20,
            text_source="html",
        )
        metric = article_metrics(old_news)
        self.assertTrue(metric["date_valid"])
        self.assertFalse(metric["fresh_within_30_days"])
        self.assertGreater(metric["age_days"], 30)
        self.assertIn("older_than_30_days", metric["issues"])

    def test_gazprom_tls_diagnostic_is_explicit_and_exact_host_only(self):
        from collectors.data_collector import _ALLOW_UNVERIFIED_TLS, _may_ignore_https_errors
        self.assertFalse(_may_ignore_https_errors("https://www.gazprom.ru/press/news/"))
        token = _ALLOW_UNVERIFIED_TLS.set(True)
        try:
            self.assertTrue(_may_ignore_https_errors("https://www.gazprom.ru/press/news/"))
            self.assertFalse(_may_ignore_https_errors("https://www.gazprom.ru.attacker.net/"))
            self.assertFalse(_may_ignore_https_errors("http://www.gazprom.ru/press/news/"))
            self.assertFalse(_may_ignore_https_errors("https://u:p@www.gazprom.ru/press/news/"))
        finally:
            _ALLOW_UNVERIFIED_TLS.reset(token)


if __name__ == "__main__":
    unittest.main()
