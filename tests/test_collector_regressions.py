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
    def test_abbreviations_do_not_delete_short_street_names(self):
        from utils.text_utils import normalize_article_text

        original = (
            '5 ноября 1975 года первый троллейбус проехал по маршруту '
            '"площадь Калинина - ул. Мастеровая" (ныне ул. Маршала Борзова). '
            'С этого дня троллейбус вошёл в транспортную систему города.'
        )
        cleaned = normalize_article_text(original)
        self.assertIn("Мастеровая", cleaned)
        self.assertIn("Маршала Борзова", cleaned)
        self.assertIn("С этого дня", cleaned)


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

    def test_real_lukoil_date_selector_ignores_script(self):
        """Реальный HTML: дата после JS-скрипта в span.pressrelise-article-date."""
        preview = ArticlePreview(
            source="ЛУКОЙЛ",
            title="На АЗС запущена новая акция",
            url="https://lukoil.ru/PressCenter/Pressreleases/Pressrelease/example",
            published_at=None, preview_text="",
        )
        html = """
        <h1>Пресс-релиз</h1>
        <div class="content">
            <h1 class="press-release-page-title">На АЗС запущена новая акция</h1>
            <script>var x = "js".repeat(500);</script>
            <p><span class="pressrelise-article-date">28 октября 2026 г.</span></p>
            <p>Компания запустила новую программу лояльности для
            покупателей топлива на российских автозаправочных
            станциях и опубликовала правила проведения акции.</p>
            <p>Участники программы могут получить скидку на топливо
            при приобретении дополнительных товаров в магазинах.</p>
            <h4>Вас может заинтересовать</h4>
            <p>Совершенно другая новость, которую включать нельзя.</p>
        </div>
        """
        with patch("collectors.data_collector.fetch_html", return_value=html):
            article = build_full_article(preview)
        self.assertEqual(article.published_at, datetime(2026, 10, 28))
        self.assertNotIn("Совершенно другая", article.text)

    def test_real_mashnews_date_selector(self):
        preview = ArticlePreview(
            source="MASHNEWS",
            title="Испытания водородного двигателя",
            url="https://mashnews.ru/test.html",
            published_at=None, preview_text="",
        )
        html = """
        <div class="grarticle-header"><div class="grarticle-items">
            <div class="grarticle-item">05 ноября 2026 23:00</div>
            <div class="grarticle-item">читать 2 минуты</div>
        </div></div>
        <h1>Испытания водородного двигателя</h1>
        <article>
            <p>В России начались испытания нового двигателя для
            отечественного водородного поезда. Разработка
            ориентирована на дальнейшее промышленное внедрение.</p>
            <p>Опытный образец проходит стендовые испытания и
            будет установлен на испытательном подвижном составе.</p>
        </article>
        """
        with patch("collectors.data_collector.fetch_html", return_value=html):
            article = build_full_article(preview)
        self.assertEqual(article.published_at, datetime(2026, 11, 5, 23, 0))

    def test_real_tpprf_date_selector_without_h1(self):
        preview = ArticlePreview(
            source="ТПП РФ",
            title="ТПП поддержала проект предприятия",
            url="https://news.tpprf.ru/ru/news/9793720/",
            published_at=None, preview_text="",
        )
        html = """
        <div class="news-detail__date">
            <div class="ribbon ribbon_fs-14">05.11.2026</div>
        </div>
        <div class="news-detail__text">
            <p>Региональная торгово-промышленная палата организовала
            мероприятие по развитию производственных предприятий
            и поддержке новых инвестиций в промышленность.</p>
            <p>Участники рассмотрели перспективы модернизации
            производственной инфраструктуры в регионе.</p>
        </div>
        """
        with patch("collectors.data_collector.fetch_html", return_value=html):
            article = build_full_article(preview)
        self.assertEqual(article.published_at, datetime(2026, 11, 5))

    def test_rostec_article_path_is_not_category(self):
        from collectors.data_collector import _is_real_article_link
        self.assertTrue(_is_real_article_link(
            "Ростех", "https://rostec.ru/media/news/odk-nagradila-luchshikh-inzhenerov/"
        ))
        self.assertFalse(_is_real_article_link(
            "Ростех", "https://rostec.ru/media/news/section/armiya/"
        ))

    def test_rosneft_official_light_site_canonicalization(self):
        from collectors.data_collector import _canonical_source_link, _rosneft_limited_url
        source = type("Source", (), {
            "name": "Роснефть", "url": "https://www.rosneft.ru/press/news/"
        })()
        self.assertEqual(
            _rosneft_limited_url(
                "https://www.rosneft.ru/press/news/item/224951/"
            ),
            "https://limited.rosneft.ru/press/news/item/224951/"
        )
        self.assertEqual(
            _canonical_source_link(
                source, "https://limited.rosneft.ru/press/news/item/224951/"
            ),
            "https://www.rosneft.ru/press/news/item/224951/"
        )
        self.assertIsNone(
            _canonical_source_link(
                source, "https://www.rosneft.ru.attacker.example/press/news/item/224951/"
            )
        )
        self.assertIsNone(
            _rosneft_limited_url("https://attacker.example/press/news/item/224951/")
        )

    def test_rosneft_fetch_tries_official_limited_mirror(self):
        from collectors.data_collector import fetch_html
        from unittest.mock import call
        url = "https://www.rosneft.ru/press/news/item/224951/"
        with patch("collectors.data_collector.fetch_html_with_requests",
                   side_effect=["", "<article>Test official article</article>"]) as req:
            with patch("collectors.data_collector.fetch_html_with_browser",
                       return_value="") as browser:
                html = fetch_html(url)
        self.assertIn("official article", html)
        self.assertEqual(req.call_args_list, [
            call(url), call("https://limited.rosneft.ru/press/news/item/224951/")
        ])
        browser.assert_called_once_with(url)

    def test_promvest_homepage_fallback_article_only(self):
        from collectors.data_collector import collect_html_previews
        source = type("Source", (), {
            "name": "Промышленный вестник / ПВ.РФ",
            "url": "https://promvest.info/ru/novosti-promyishlennosti/",
        })()
        homepage = """
        <a href="/ru/novosti-promyishlennosti/sinara-tramvay-71-233a/">
        «Синара» вывела на обкатку в Петербурге трехсекционный трамвай
        </a>
        <a href="/ru/reklama/">
        Условия размещения рекламы на промышленном портале
        </a>
        """
        def fake_html(url):
            return homepage if url == "https://promvest.info/" else ""
        with patch("collectors.data_collector.fetch_html", side_effect=fake_html):
            previews = collect_html_previews(source, 5)
        self.assertEqual(len(previews), 1)
        self.assertIn("Синара", previews[0].title)
        self.assertIn("/sinara-tramvay-71-233a/", previews[0].url)

    def test_site_database_error_is_not_collected_as_news(self):
        from collectors.data_collector import collect_html_previews
        source = type("Source", (), {
            "name": "Производства.рф",
            "url": "https://производства.рф/articles/",
        })()
        with patch("collectors.data_collector.fetch_html",
                   return_value="<html><body>Error establishing a database connection</body></html>"):
            previews = collect_html_previews(source, 5)
        self.assertEqual(previews, [])

    def test_tls_errors_do_not_disable_verification(self):
        import os
        from requests.exceptions import SSLError
        from collectors.data_collector import fetch_html_with_requests
        with patch.dict(os.environ, {"NEWS_CA_BUNDLE": ""}):
            with patch("collectors.data_collector.requests.get",
                       side_effect=SSLError("untrusted CA")) as req:
                html = fetch_html_with_requests("https://rostec.ru/media/news/")
        self.assertEqual(html, "")
        self.assertIs(req.call_args.kwargs["verify"], True)

    def test_rosneft_uses_nominative_date_and_discards_press_footer(self):
        from collectors.data_collector import _parse_published
        self.assertEqual(_parse_published("05 Ноябрь 2026"), datetime(2026, 11, 5))
        preview = ArticlePreview(
            source="Роснефть", title="Роснефть разработала новое оборудование",
            url="https://www.rosneft.ru/press/news/item/224951/",
            published_at=None, preview_text="",
        )
        html = """
        <h1>Роснефть разработала новое оборудование</h1>
        <div class="news-detail">
          <p>Специалисты компании создали новое устройство для
          повышения надёжности работы нефтедобывающих насосов и
          провели длительные испытания на нескольких скважинах.</p>
          <p>Оборудование позволило увеличить срок эксплуатации
          насосных агрегатов и повысить промышленную безопасность
          при освоении нефтяных месторождений.</p>
          <p>Департамент информации и рекламы ПАО НК Роснефть
          5 ноября 2026 г.</p>
        </div>
        """
        with patch("collectors.data_collector.fetch_html", return_value=html):
            article = build_full_article(preview)
        self.assertIn("промышленную безопасность", article.text)
        self.assertNotIn("Департамент информации", article.text)

    def test_rostec_listing_accepts_article_not_tags(self):
        from collectors.data_collector import collect_html_previews
        source = type("Source", (), {
            "name": "Ростех",
            "url": "https://rostec.ru/media/news/",
        })()
        html = """
        <h2><a href="/media/news/odk-nagradila-luchshikh-inzhenerov/">
        ОДК наградила лучших молодых инженеров</a></h2>
        <h2><a href="/media/news/section/armiya/">
        Вся информация о направлении</a></h2>
        """
        with patch("collectors.data_collector.fetch_html", return_value=html):
            previews = collect_html_previews(source, 5)
        self.assertEqual(len(previews), 1)
        self.assertEqual(previews[0].url,
                         "https://rostec.ru/media/news/odk-nagradila-luchshikh-inzhenerov/")

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
