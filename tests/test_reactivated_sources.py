"""Офлайн-регрессии восьми повторно подключаемых источников.

Запуск: python -m unittest discover -s tests -p 'test_reactivated_sources.py' -v
Реальные сайты, PostgreSQL и MAX не используются.
"""
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from requests.exceptions import HTTPError

from collectors.data_collector import (
    Article, ArticlePreview,
    _is_real_article_link,
    build_full_article,
    collect_milpress_previews,
    collect_news,
    clearly_foreign_milpress_article,
    is_foreign_general_rss,
    is_milpress_consumer_news,
    collect_rss_previews,
    fetch_html,
)
from database.repository import seed_sources


REACTIVATED = (
    "Газпром", "Военное.рф", "ФлотПром", "Отраслевое.рф",
    "ФлотКом", "РИА Новости", "ТАСС", "Интерфакс",
)


class FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


class ReactivatedSourcesTests(unittest.TestCase):
    def test_all_eight_are_enabled_in_seed(self):
        with patch("database.repository.get_or_create_source") as mock_get:
            seed_sources(object())
        flags = {
            call.kwargs["name"]: call.kwargs["is_active"]
            for call in mock_get.call_args_list
        }
        for name in REACTIVATED:
            with self.subTest(name=name):
                self.assertIs(flags[name], True)

    def test_milpress_ignores_archive_category_and_other_domains(self):
        year = datetime.now().year
        for name, base in (
            ("Военное.рф", "https://военное.рф"),
            ("ФлотПром", "https://flotprom.ru"),
            ("Отраслевое.рф", "https://отраслевое.рф"),
            ("ФлотКом", "https://flot.com"),
        ):
            with self.subTest(source=name):
                source = SimpleNamespace(
                    name=name,
                    url=f"{base}/{year}/" if name in ("Военное.рф", "ФлотПром") else base,
                )
                html = f"""
                    <a href="/{year}/">Новости {year} года и архив</a>
                    <a href="/{year}/Судостроение7/">
                        Важное Российский завод начал строительство нового судна
                    </a>
                    <a href="/{year-1}/Судостроение8/">
                        В архиве много старых новостей о кораблях
                    </a>
                    <a href="/category/technology/">
                        Новости машиностроительной отрасли России
                    </a>
                    <a href="https://example.com/{year}/other/">
                        Другая внешняя новость о промышленности России
                    </a>
                """
                with patch("collectors.data_collector.fetch_html", return_value=html):
                    previews = collect_milpress_previews(source, 10)
                self.assertEqual(len(previews), 1)
                self.assertEqual(previews[0].url, f"{base}/{year}/Судостроение7/")
                self.assertEqual(
                    previews[0].title,
                    "Российский завод начал строительство нового судна",
                )

    def test_rss_scans_past_ten_and_keeps_industry_only(self):
        for name, domain in (
            ("РИА Новости", "https://ria.ru"),
            ("ТАСС", "https://tass.ru"),
            ("Интерфакс", "https://www.interfax.ru"),
        ):
            with self.subTest(source=name):
                items = "".join(
                    f"<item><title>Новости спорта номер {i}</title>"
                    f"<link>{domain}/sport/{i}</link>"
                    "<description>Результаты соревнования</description></item>"
                    for i in range(15)
                )
                items += (
                    "<item><title>Российский завод запустил производство станков</title>"
                    f"<link>{domain}/business/20261107001</link>"
                    "<pubDate>Sat, 07 Nov 2026 12:00:00 +0300</pubDate>"
                    "<description>В цехе заработала новая линия</description></item>"
                )
                xml = f'<?xml version="1.0" encoding="utf-8"?><rss version="2.0"><channel>{items}</channel></rss>'
                source = SimpleNamespace(
                    name=name, url=domain, rss_url=domain + "/feed.xml"
                )
                with patch(
                    "collectors.data_collector.requests.get",
                    return_value=FakeResponse(xml.encode("utf-8")),
                ) as request:
                    previews = collect_rss_previews(source, 3)
                self.assertEqual(len(previews), 1)
                self.assertIn("завод", previews[0].title.lower())
                self.assertEqual(previews[0].published_at, datetime(2026, 11, 7, 9))
                self.assertEqual(request.call_args.kwargs["timeout"], (10, 30))

    def test_international_rss_requires_russian_connection(self):
        self.assertTrue(is_foreign_general_rss(
            "https://www.interfax.ru/world/12345",
            "Хуситы атаковали нефтяные объекты Саудовской Аравии",
            "Йеменские повстанцы сообщили об ударах.",
        ))
        self.assertTrue(is_foreign_general_rss(
            "https://tass.ru/mezhdunarodnaya-panorama/456",
            "В Европе построили иностранный оборонный завод",
            "Европейские компании инвестируют в производство.",
        ))
        self.assertFalse(is_foreign_general_rss(
            "https://www.interfax.ru/world/12345",
            "Российский завод заключил зарубежный контракт",
            "Ростех поставит оборудование.",
        ))
        self.assertFalse(is_foreign_general_rss(
            "https://www.interfax.ru/russia/12345",
            "Промышленное предприятие открыли в Пензе",
            "В Пензенской области заработала новая линия.",
        ))

    def test_rss_http_failure_does_not_return_fake_articles(self):
        source = SimpleNamespace(
            name="РИА Новости", url="https://ria.ru",
            rss_url="https://ria.ru/export/rss2/index.xml",
        )
        with patch(
            "collectors.data_collector.requests.get",
            side_effect=HTTPError("429 Too Many Requests"),
        ):
            self.assertEqual(collect_rss_previews(source, 3), [])

    def test_gazprom_accepts_articles_not_section_pages(self):
        self.assertTrue(_is_real_article_link(
            "Газпром",
            "https://www.gazprom.ru/press/news/2026/october/article585577/",
        ))
        self.assertFalse(_is_real_article_link(
            "Газпром", "https://www.gazprom.ru/press/news/",
        ))
        self.assertFalse(_is_real_article_link(
            "Газпром",
            "https://www.gazprom.ru/press/news/miller-journal/2026/646503/",
        ))

    def test_milpress_article_has_text_and_local_published_time(self):
        preview = ArticlePreview(
            source="ФлотПром",
            title='На "Янтаре" автоматизируют линию резки металла',
            url="https://flotprom.ru/2026/Янтарь5/",
            published_at=None,
            preview_text="",
        )
        html = """
            <main><h1>На "Янтаре" автоматизируют линию резки металла</h1>
            <span>Важное 5 ноября 2026 в 10:25</span>
            <article><p>На российском судостроительном заводе Янтарь
            установлена новая роботизированная линия резки металлического
            профиля, которая автоматизирует производственные операции.</p>
            <p>Оборудование сокращает трудозатраты и ускоряет подготовку
            комплектующих для новых судов. Сотрудники завершили наладку
            производственного комплекса и проверили параметры линии.</p>
            <aside class="related"><p>Соседняя публикация об ипотеке.</p></aside>
            </article></main>
        """
        with patch("collectors.data_collector.fetch_html", return_value=html):
            article = build_full_article(preview)
        self.assertEqual(article.published_at, datetime(2026, 11, 5, 10, 25))
        self.assertIn("роботизированная линия", article.text)
        self.assertNotIn("ипотеке", article.text)
        self.assertEqual(article.text_source, "html")


    def test_milpress_foreign_relevance_with_russia_exception(self):
        self.assertTrue(clearly_foreign_milpress_article(
            "Финляндия поставит донные мины для датского флота",
            "Финские компании поставляют мины военно-морским силам Дании.",
        ))
        self.assertFalse(clearly_foreign_milpress_article(
            "Финляндия поставит оборудование для России",
            "Российские предприятия закупили оборудование.",
        ))
        self.assertFalse(clearly_foreign_milpress_article(
            'На "Янтаре" автоматизируют линию резки металла',
            "Завод ОСК внедрил роботизированную линию.",
        ))
        self.assertTrue(clearly_foreign_milpress_article(
            "Акции немецкого оборонного гиганта подешевели",
            "В Германии участники торгов снизили цену бумаг компании.",
        ))
        self.assertTrue(clearly_foreign_milpress_article(
            "Начался новый этап европейского многоцелевого корвета",
            "Франция и Италия подписали договор о разработке корабля.",
        ))
        self.assertTrue(is_milpress_consumer_news(
            "Россиянам объяснили, почему дорожает курица",
        ))
        self.assertFalse(is_milpress_consumer_news(
            "На заводе начали выпуск нового оборудования",
        ))

    def test_milpress_does_not_stop_on_rejected_news(self):
        source = SimpleNamespace(
            name="ФлотПром", url="https://flotprom.ru/2026/",
            rss_url=None, source_type="html",
        )
        def preview(title, slug):
            return ArticlePreview(
                source="ФлотПром", title=title,
                url=f"https://flotprom.ru/2026/{slug}/",
                published_at=None, preview_text="",
            )
        previews = [
            preview("Главное в отраслевых СМИ за неделю: обзор событий", "Дайджест39"),
            preview("Финляндия поставит донные мины для датского флота", "Европа130"),
            preview('На "Янтаре" автоматизируют линию резки металла', "Янтарь5"),
        ]
        def make_article(p):
            if "Дайджест" in p.url:
                text = "Обзор отраслевых событий за неделю. " * 20
            elif "Европа" in p.url:
                text = "Финские компании поставляют военные технологии Дании. " * 15
            else:
                text = "На заводе Янтарь ОСК введена новая линия резки металла. " * 12
            return Article(
                source=p.source, title=p.title, url=p.url,
                published_at=None, text=text, text_source="html",
            )
        with (
            patch(
                "collectors.data_collector.collect_milpress_previews",
                return_value=previews,
            ),
            patch(
                "collectors.data_collector.build_full_article",
                side_effect=make_article,
            ),
            patch("collectors.data_collector.time.sleep"),
        ):
            diagnostics = {}
            articles = collect_news(
                [source], limit_per_source=1, diagnostics=diagnostics,
            )
        self.assertEqual(len(articles), 1)
        self.assertIn("Янтаре", articles[0].title)
        self.assertEqual(diagnostics["ФлотПром"]["accepted"], 1)
        self.assertEqual(len(diagnostics["ФлотПром"]["filtered_out"]), 2)


    def test_ria_uses_larger_verified_archive_feed(self):
        registered = []
        with patch(
            "database.repository.get_or_create_source",
            side_effect=lambda **kwargs: registered.append(kwargs),
        ):
            seed_sources(None)
        ria = next(s for s in registered if s["name"] == "РИА Новости")
        self.assertEqual(
            ria["rss_url"], "https://ria.ru/export/rss2/archive/index.xml",
        )

    def test_rss_excludes_foreign_actor_without_world_section(self):
        self.assertTrue(is_foreign_general_rss(
            "https://ria.ru/20261107/sudno-1000.html",
            "США вывели из строя судно в Оманском заливе",
            "",
        ))
        self.assertFalse(is_foreign_general_rss(
            "https://ria.ru/20261107/stanok-1000.html",
            "США закупили российские станки",
            "Российский завод отправил оборудование на экспорт.",
        ))

    def test_rss_collector_scans_more_than_requested_publication_limit(self):
        source = SimpleNamespace(
            name="РИА Новости", url="https://ria.ru",
            rss_url="https://ria.ru/export/rss2/archive/index.xml",
            source_type="rss",
        )
        previews = [
            ArticlePreview(
                source=source.name,
                title="В России отметили лучшие промышленные проекты",
                url=f"https://ria.ru/20261107/news-{i}.html",
                published_at=datetime(2026, 11, 7),
                preview_text="",
            )
            for i in range(3)
        ]
        def make_article(p):
            if p.url.endswith("news-0.html"):
                title = "Дайджест главных новостей промышленности"
            elif p.url.endswith("news-1.html"):
                title = "Новостной заголовок"
            else:
                title = "Российский завод открыл производство двигателей"
            return Article(
                source=p.source, title=title, url=p.url,
                published_at=p.published_at,
                text="Российский завод модернизировал производство. " * 12,
                text_source="html",
            )
        with (
            patch(
                "collectors.data_collector.collect_rss_previews",
                return_value=previews,
            ) as collector,
            patch(
                "collectors.data_collector.build_full_article",
                side_effect=make_article,
            ),
            patch("collectors.data_collector.time.sleep"),
        ):
            result = collect_news([source], limit_per_source=1)
        self.assertEqual(len(result), 1)
        self.assertIn("двигателей", result[0].title)
        self.assertGreaterEqual(collector.call_args.kwargs["limit"], 20)

    def test_ria_tass_empty_html_shell_uses_browser_or_fails_closed(self):
        shell = "<html><head><script>renderArticle()</script></head><body></body></html>"
        article = (
            "<html><h1>Российский завод ввёл в эксплуатацию новый станок</h1>"
            "<article><p>В России модернизирован завод по выпуску "
            "промышленного оборудования и новых станков для предприятий.</p>"
            "</article></html>"
        )
        for url in (
            "https://tass.ru/ekonomika/28295367",
            "https://ria.ru/20261107/stanok-123.html",
        ):
            with self.subTest(url=url):
                with (
                    patch(
                        "collectors.data_collector.fetch_html_with_requests",
                        return_value=shell,
                    ),
                    patch(
                        "collectors.data_collector.fetch_html_with_browser",
                        return_value=article,
                    ) as browser,
                ):
                    self.assertIn("Российский завод", fetch_html(url))
                    browser.assert_called_once()
                with (
                    patch(
                        "collectors.data_collector.fetch_html_with_requests",
                        return_value=shell,
                    ),
                    patch(
                        "collectors.data_collector.fetch_html_with_browser",
                        return_value=shell,
                    ),
                ):
                    self.assertEqual(fetch_html(url), "")


if __name__ == "__main__":
    unittest.main()
