"""Регрессионные тесты: запускать python -m unittest discover -s tests -v."""
import unittest
from datetime import datetime
from unittest.mock import patch

from bs4 import BeautifulSoup

from collectors.data_collector import (
    ArticlePreview,
    _page_date,
    build_full_article,
    collect_html_previews,
    extract_text_nedradv,
)


class CollectorTests(unittest.TestCase):
    def test_nedradv_does_not_capture_related_news_or_auctions(self):
        html = """
        <h1>На карьере запущена новая линия</h1>
        <p>На карьере запустили новую линию переработки руды с
           производительностью более тысячи тонн в сутки.</p>
        <blockquote>Предприятие завершило основные испытания.</blockquote>
        <h2>Читайте также</h2>
        <p>В начале года прежнего руководителя задержали по подозрению
           в получении взяток.</p>
        <h2>Последние аукционы</h2>
        <p>Одинокое, рудно-россыпное месторождение олова.</p>
        """
        soup = BeautifulSoup(html, "html.parser")
        text = extract_text_nedradv(soup, "На карьере запущена новая линия")
        self.assertIn("переработки руды", text)
        self.assertIn("испытания", text)
        self.assertNotIn("взяток", text)
        self.assertNotIn("Одинокое", text)

    def test_nedradv_publication_date(self):
        soup = BeautifulSoup("""
        <h1>Новость о добыче</h1>
        <p>5 ноября 2026 года, NEDRADV. На участке начата новая
           добыча полезных ископаемых.</p>
        <h2>Читайте также</h2>
        <p>6 ноября 2026 года, NEDRADV. Другая статья.</p>
        """, "html.parser")
        self.assertEqual(_page_date(soup, {}), datetime(2026, 11, 5))

    def test_metadata_date_and_article_body(self):
        preview = ArticlePreview(
            source="Ростех", title="Начат выпуск новых двигателей",
            url="https://rostec.ru/media/news/new-engines/",
            published_at=None, preview_text=""
        )
        html = """
        <html><head>
        <meta property="article:published_time" content="2026-11-05T12:00:00+03:00">
        </head><body><h1>Начат выпуск новых двигателей</h1>
        <article><p>Предприятие запустило производство двигателей,
        которые предназначены для новых отечественных самолётов.</p>
        <p>Новые двигатели прошли испытания на заводе и поставляются
        российским заказчикам.</p>
        <section class="related"><p>Читайте также новость о налоговом
        форуме, которая не относится к двигателям.</p></section>
        </article></body></html>
        """
        with patch("collectors.data_collector.fetch_html", return_value=html):
            article = build_full_article(preview)
        self.assertEqual(article.published_at, datetime(2026, 11, 5, 9))
        self.assertIn("производство двигателей", article.text)
        self.assertNotIn("налоговом", article.text)

    def test_iapn_article_urls(self):
        html = """
        <main><h3><a href="/dvigateli-odk-obespechili-zapusk-rakety/">
        Двигатели ОДК обеспечили запуск ракеты</a></h3>
        <h3><a href="/contact/">Связаться с редакцией сайта</a></h3>
        </main>
        """
        source = type("Source", (), {
            "name": "ИАПН", "url": "https://iapn.ru"
        })()
        with patch("collectors.data_collector.fetch_html", return_value=html):
            previews = collect_html_previews(source, 5)
        self.assertEqual(len(previews), 1)
        self.assertIn("Двигатели", previews[0].title)

    def test_promvest_article_urls(self):
        html = """
        <main><h2><a href="/ru/novosti-promyishlennosti/sollers-zapustil-proizvodstvo/">
        СОЛЛЕРС запустил производство двигателей в России</a></h2></main>
        """
        source = type("Source", (), {
            "name": "Промышленный вестник / ПВ.РФ",
            "url": "https://promvest.info/ru/novosti-promyishlennosti/"
        })()
        with patch("collectors.data_collector.fetch_html", return_value=html):
            previews = collect_html_previews(source, 5)
        self.assertEqual(len(previews), 1)

    def test_kommersant_only_theme_not_general_feed(self):
        html = """
        <h2>Лента</h2><a href="/doc/100">Новость общего раздела</a>
        <h1>Промышленность</h1>
        <a href="/doc/200">Арматура подешевела на российском рынке</a>
        <h2>Новости компаний</h2>
        <a href="/doc/300">Банк увеличил прибыль за квартал</a>
        """
        source = type("Source", (), {
            "name": "Коммерсантъ — Промышленность",
            "url": "https://www.kommersant.ru/theme/3631"
        })()
        with patch("collectors.data_collector.fetch_html", return_value=html):
            previews = collect_html_previews(source, 5)
        self.assertEqual([p.url for p in previews], [
            "https://www.kommersant.ru/doc/200"
        ])

    def test_legacy_generic_fallback_restores_main_paragraphs(self):
        # В ряде корпоративных CMS нет article / itemprop=articleBody:
        # предыдущий сборщик использовал main p.
        preview = ArticlePreview(
            source="Ростех", title="Завод запустил выпуск продукции",
            url="https://rostec.ru/media/news/new-plant/",
            published_at=None, preview_text=""
        )
        html = """
        <h1>Завод запустил выпуск продукции</h1>
        <main><div class="cms-content"><p>
        Российское предприятие завершило монтаж новой технологической
        линии и начало выпуск продукции для отечественных заказчиков.
        Завод получил необходимое оборудование и провёл испытания.
        Производство будет постепенно увеличивать объёмы поставок
        на внутренний рынок и повышать долю местных комплектующих.
        </p></div></main>
        """
        with patch("collectors.data_collector.fetch_html", return_value=html):
            article = build_full_article(preview)
        self.assertGreater(len(article.text), 200)
        self.assertIn("Российское предприятие", article.text)

    def test_lukoil_legacy_extractor_restored(self):
        preview = ArticlePreview(
            source="ЛУКОЙЛ", title="Пресс-релиз",
            url="https://lukoil.ru/PressCenter/Pressreleases/pressrelease-1",
            published_at=None, preview_text=""
        )
        html = """
        <h1>Новая скважина введена в эксплуатацию</h1>
        <div class="main-content">
        <p>Предприятие ЛУКОЙЛ ввело в эксплуатацию новую скважину
        на нефтяном месторождении в Российской Федерации. Работы
        включали модернизацию оборудования и подготовку инфраструктуры.</p>
        <p>В результате производительность добычи увеличилась,
        а технологические процессы стали более автоматизированными.
        Компания продолжит внедрение отечественных разработок.</p>
        </div>
        """
        with patch("collectors.data_collector.fetch_html", return_value=html):
            article = build_full_article(preview)
        self.assertIn("скважину", article.text)

    def test_promvest_plain_anchor_without_nested_heading(self):
        html = """
        <div><a href="/ru/vazhno/rukovoditeli-proekta-osk/">
        Руководители судостроительных проектов ОСК прошли обучение
        </a></div>
        """
        source = type("Source", (), {
            "name": "Промышленный вестник / ПВ.РФ",
            "url": "https://promvest.info/ru/novosti-promyishlennosti/"
        })()
        with patch("collectors.data_collector.fetch_html", return_value=html):
            previews = collect_html_previews(source, 5)
        self.assertEqual(len(previews), 1)
        self.assertIn("ОСК", previews[0].title)

    def test_unknown_container_rejected_instead_of_mixing_news(self):
        preview = ArticlePreview(
            source="НедраДВ", title="Первая новость",
            url="https://nedradv.ru/nedradv/ru/page_news?obj=abc",
            published_at=None, preview_text=""
        )
        with patch("collectors.data_collector.fetch_html",
                   return_value="<h1>Первая новость</h1><h2>Читайте также</h2><p>Чужая статья о другом проекте</p>"):
            article = build_full_article(preview)
        self.assertFalse(article.text)


if __name__ == "__main__":
    unittest.main()
