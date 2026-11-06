"""Проверки структурного извлечения для ВСЕХ поддерживаемых HTML-источников.

Запуск: python -m unittest discover -s tests -v
Реальные сайты не запрашиваются; используем типовые HTML-фрагменты.
"""
import unittest

from bs4 import BeautifulSoup

from collectors.data_collector import (
    _ARTICLE_SELECTORS,
    _article_paragraphs,
    _extract_main_text,
    extract_by_selectors,
)


LEAD = (
    "Российское машиностроительное предприятие завершило модернизацию "
    "производственной линии и организовало изготовление новых комплектующих "
    "для промышленного оборудования в рамках программы импортозамещения."
)

DETAILS = (
    "Инженеры завода испытали оборудование на производственной площадке, "
    "измерили основные характеристики и подтвердили соответствие "
    "запланированным техническим требованиям и параметрам эксплуатации."
)


class ExtractionForAllSourcesTests(unittest.TestCase):
    def test_all_configured_sources_preserve_in_article_lists(self):
        """Общий путь: Ростех, Роснефть, ОСК, ТПП, MASHNEWS, Заводы и др."""
        html = f"""
            <nav><ul><li>Открыть меню сайта</li></ul></nav>
            <article>
              <p>{LEAD}</p>
              <div class="article-list-wrap">
                <ul><li><a href="#one">Оглавление страницы</a></li></ul>
              </div>
              <div class="ya-share2">
                <ul><li>Одноклассники</li><li>ВКонтакте</li></ul>
              </div>
              <h3>Технические параметры</h3>
              <ul>
                <li>Модель Ту-95РЦ</li>
                <li><a href="/model">Ту-95А</a></li>
                <li>500 МВт</li>
              </ul>
              <p>{DETAILS}</p>
              <aside class="related-news">
                <ul><li>Посторонняя новость из боковой ленты</li></ul>
              </aside>
            </article>
        """
        special = {"НедраДВ", "ЛУКОЙЛ", "Коммерсантъ — Промышленность"}
        for name in _ARTICLE_SELECTORS:
            if name in special:
                continue
            with self.subTest(source=name):
                text = _extract_main_text(
                    BeautifulSoup(html, "html.parser"),
                    source_name=name, title="Модернизация производства",
                    jsonld={},
                )
                self.assertIn("Модель Ту-95РЦ", text)
                self.assertIn("Ту-95А", text)
                self.assertIn("500 МВт", text)
                self.assertIn("Технические параметры", text)
                self.assertIn("измерили основные характеристики", text)
                self.assertNotIn("Оглавление страницы", text)
                self.assertNotIn("ВКонтакте", text)
                self.assertNotIn("Посторонняя новость", text)

    def test_nedradv_preserves_lists_until_related_news_section(self):
        html = f"""
            <h1>На карьере модернизировали добывающую технику</h1>
            <p>{LEAD}</p>
            <ul><li>Мощность оборудования 500 МВт</li>
                <li>Добыча увеличена на 20%</li></ul>
            <p>{DETAILS}</p>
            <h2>Читайте также</h2>
            <ul><li>Посторонний заголовок соседней новости</li></ul>
        """
        result = _extract_main_text(
            BeautifulSoup(html, "html.parser"),
            "НедраДВ", "На карьере модернизировали добывающую технику", {},
        )
        self.assertIn("500 МВт", result)
        self.assertIn("20%", result)
        self.assertNotIn("Посторонний заголовок", result)

    def test_lukoil_preserves_lists_until_related_news_section(self):
        html = f"""
            <h1>ЛУКОЙЛ внедрил новое промышленное оборудование</h1>
            <p>{LEAD}</p>
            <ol><li>Показатель эффективности 42%</li>
                <li>Мощность 500 МВт</li></ol>
            <p>{DETAILS}</p>
            <h2>Вас может заинтересовать</h2>
            <p>Посторонняя рекомендованная статья, которая не входит в релиз.</p>
        """
        result = _extract_main_text(
            BeautifulSoup(html, "html.parser"),
            "ЛУКОЙЛ", "ЛУКОЙЛ внедрил новое промышленное оборудование", {},
        )
        self.assertIn("42%", result)
        self.assertIn("500 МВт", result)
        self.assertNotIn("Посторонняя рекомендованная", result)

    def test_kommersant_preserves_standalone_content_lists(self):
        html = f"""
            <div class="article_text_wrapper">
                <div class="doc-content__text-block">{LEAD}</div>
                <ul class="doc-content__list">
                    <li>Введено оборудование 500 МВт</li>
                    <li>Сокращены затраты на 30%</li>
                </ul>
                <div class="doc-content__text-block">{DETAILS}</div>
            </div>
        """
        result = _extract_main_text(
            BeautifulSoup(html, "html.parser"),
            "Коммерсантъ — Промышленность", "Модернизация оборудования", {},
        )
        self.assertIn("500 МВт", result)
        self.assertIn("30%", result)
        self.assertIn("измерили основные характеристики", result)

    def test_jsonld_does_not_hide_lists_when_html_is_complete(self):
        html = f"""
            <article>
                <p>{LEAD}</p>
                <ul><li>Восстановлен станок Ту-95А</li>
                    <li>Мощность 500 МВт</li></ul>
                <p>{DETAILS}</p>
            </article>
        """
        result = _extract_main_text(
            BeautifulSoup(html, "html.parser"),
            "Ростех", "Модернизация оборудования",
            {"articleBody": LEAD + " " + DETAILS},
        )
        self.assertIn("Восстановлен станок Ту-95А", result)
        self.assertIn("500 МВт", result)

    def test_old_paragraph_selector_includes_adjacent_list(self):
        html = f"""
            <article>
                <p>{LEAD}</p>
                <ul><li>Скорость линии 1000 единиц в сутки</li>
                    <li>Расход энергии 500 МВт</li></ul>
                <p>{DETAILS}</p>
            </article>
        """
        result = extract_by_selectors(
            BeautifulSoup(html, "html.parser"),
            ["article p"], min_len=150, title="Модернизация",
        )
        self.assertIn("1000 единиц", result)
        self.assertIn("500 МВт", result)

    def test_nested_list_and_nested_paragraph_not_duplicated(self):
        block = BeautifulSoup(f"""
            <article>
              <p>{LEAD}</p>
              <ul>
                <li><p>Комплект оборудования модели А</p>
                    <ul><li>Подсистема измерения 500 МВт</li></ul>
                </li>
              </ul>
              <p>{DETAILS}</p>
            </article>
        """, "html.parser").article
        result = _article_paragraphs(block, "Модернизация")
        self.assertEqual(result.count("Комплект оборудования модели А"), 1)
        self.assertEqual(result.count("Подсистема измерения 500 МВт"), 1)


if __name__ == "__main__":
    unittest.main()
