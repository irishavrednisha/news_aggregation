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
        <h2>НОВОСТИ</h2><main><h3><a href="/dvigateli-odk-obespechili-zapusk-rakety/">
        Двигатели ОДК обеспечили запуск ракеты</a></h3>
        <h3><a href="/contact/">Связаться с редакцией сайта</a></h3>
        </main><h2>СЕВЕРО-ЗАПАДНЫЙ ФЕДЕРАЛЬНЫЙ ОКРУГ</h2>
        <h3><a href="/old-archive-article/">Очень старая статья из архива</a></h3>
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

    def test_lukoil_excludes_recommendations_and_extracts_date(self):
        preview = ArticlePreview(
            source="ЛУКОЙЛ",
            title="Пресс-релиз",
            url="https://lukoil.ru/PressCenter/Pressreleases/Pressrelease/example",
            published_at=None,
            preview_text="",
        )
        html = """
        <h1>Пресс-релиз</h1>
        <h1>Лаявожское месторождение ввели в эксплуатацию</h1>
        <span>9 октября 2026 г.</span>
        <div>
          <p>ПАО ЛУКОЙЛ приступило к эксплуатации нового нефтяного
          месторождения в Ненецком округе и запустило необходимую
          производственную инфраструктуру.</p>
          <p>Компания построила нефтепромысловую инфраструктуру,
          включая технологические площадки и энергетический центр.</p>
        </div>
        <h4>Вас может заинтересовать</h4>
        <p>Другая новость о запуске совершенно другого месторождения,
        которая должна быть исключена из текста первой новости.</p>
        """
        with patch("collectors.data_collector.fetch_html", return_value=html):
            article = build_full_article(preview)
        self.assertIn("технологические площадки", article.text)
        self.assertNotIn("совершенно другого", article.text)
        self.assertEqual(article.published_at, datetime(2026, 10, 9))

    def test_eprussia_publication_date_from_footer(self):
        preview = ArticlePreview(
            source="Энергетика и промышленность России",
            title="ГОСТ для подземных хранилищ газа",
            url="https://www.eprussia.ru/news/base/2026/100.html",
            published_at=None, preview_text="",
        )
        html = """
        <h1>ГОСТ для подземных хранилищ газа</h1>
        <main>
          <p>В России впервые утверждён новый отраслевой стандарт для
          разработки проектной документации подземных газовых хранилищ.
          Он определяет требования к эксплуатации энергетических объектов.</p>
          <p>Специалисты отмечают существенное повышение надёжности
          системы и безопасность проектирования газовых хранилищ.</p>
          <p>Опубликовано: 05 ноября 2026, 09:05</p>
        </main>
        """
        with patch("collectors.data_collector.fetch_html", return_value=html):
            article = build_full_article(preview)
        self.assertEqual(article.published_at, datetime(2026, 11, 5, 9, 5))
        self.assertNotIn("Опубликовано:", article.text)

    def test_iapn_ignores_archival_sections(self):
        source = type("Source", (), {
            "name": "ИАПН", "url": "https://iapn.ru/"
        })()
        html = """
        <h2>НОВОСТИ</h2>
        <h3><a href="/novaya-tekhnologiya-rossii/">
        В России представили новую промышленную технологию</a></h3>
        <h2>ПРОЕКТЫ ИАПН</h2>
        <h3><a href="/staryj-proekt-proizvodstva/">
        Японские проекты по промышленному развитию</a></h3>
        """
        with patch("collectors.data_collector.fetch_html", return_value=html):
            previews = collect_html_previews(source, 5)
        self.assertEqual(len(previews), 1)
        self.assertIn("промышленную технологию", previews[0].title)

    def test_zavody_article_container_and_local_date(self):
        # Реальная структура из collector_html_20261106_035534.zip:
        # main > .article-txt-wrap > .text-bl и span.date.
        preview = ArticlePreview(
            source="Заводы.рф", title="Завод передан государству",
            url="https://заводы.рф/publication/test-plant",
            published_at=None, preview_text="",
        )
        html = """
        <main>
          <div class="article-head__top"><span class="date">03 сентября 2026 г.</span></div>
          <h1>Завод передан государству</h1>
          <div class="article-txt-wrap">
            <div class="text-bl">
              <div class="article-head"><p>Одно из крупных предприятий
              области перешло в собственность государства по решению суда.
              На заводе трудится более пятисот сотрудников.</p></div>
              <p>Предприятие работает в обычном режиме и выполняет
              обязательства перед поставщиками и заказчиками.
              Производственные планы не изменились после решения суда.</p>
              <p>Обсудить новости промышленности можно в нашем чате в Telegram.</p>
            </div>
          </div>
          <div class="author">Последние статьи автора:
          <p>Совершенно чужая статья про банкротство другой компании.</p></div>
        </main>
        """
        with patch("collectors.data_collector.fetch_html", return_value=html):
            article = build_full_article(preview)
        self.assertIn("выполняет обязательства", article.text)
        self.assertNotIn("другой компании", article.text)
        self.assertNotIn("Обсудить новости", article.text)
        self.assertEqual(article.published_at, datetime(2026, 9, 3))

    def test_kommersant_text_blocks_not_image_captions(self):
        # Актуальная структура Kommersant: текст в div, а не p.
        preview = ArticlePreview(
            source="Коммерсантъ — Промышленность",
            title="Цены на арматуру снизились",
            url="https://www.kommersant.ru/doc/9079360",
            published_at=None, preview_text="",
        )
        html = """
        <meta property="article:published_time" content="2026-11-03T01:36:00Z">
        <h1>У проката остыл спрос</h1>
        <div class="article_text_wrapper">
          <div class="doc-content__text-block">
            <div>К концу октября стоимость арматуры на российском рынке
            опустилась до минимального уровня за полгода.
            Причина — избыток предложения на рынке металлопроката.</div>
          </div>
          <div class="doc-content__image-block"><p>Фото рекламного баннера.</p></div>
          <div class="doc-content__text-block">
            <div>Представители металлургических предприятий отмечают
            сезонное снижение спроса и ожидают восстановления закупок
            с началом нового строительного сезона в следующем году.</div>
          </div>
        </div>
        """
        with patch("collectors.data_collector.fetch_html", return_value=html):
            article = build_full_article(preview)
        self.assertIn("избыток предложения", article.text)
        self.assertIn("восстановления закупок", article.text)
        self.assertNotIn("Фото рекламного баннера", article.text)
        self.assertEqual(article.published_at, datetime(2026, 11, 3, 1, 36))

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
