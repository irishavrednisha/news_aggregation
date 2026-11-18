"""Follow-up stage 2 regressions from real-world review and URL edge cases.

These are selected diagnostics, NOT an independent blind estimate of accuracy.
"""
import unittest
from datetime import date
from types import SimpleNamespace

from processing.exact_duplicates import match_exact, normalize_url
from processing.news_intake import classify_intake

TODAY = date(2026, 11, 18)
BODY = ("Предприятие представило технический отчёт о производстве. "
        "Инженеры описали испытания, результаты измерений, объёмы выпуска "
        "продукции и работу технологического оборудования. ") * 5


def news(title, *, url="https://example.com/news/100", text=BODY, published="2026-11-18"):
    return SimpleNamespace(title=title, url=url, text=text, published_at=published,
                           text_source="html", source="AUDIT")


class Stage2RealWorldRecallRegressions(unittest.TestCase):
    def test_headline_contract(self):
        # Each tuple fixes one previously overlooked industrial or junk edge.
        cases = [
            ("«Уралхимпласт» вывел из эксплуатации и ликвидировал устаревшее производство", "accepted"),
            ("«Норникель» объявляет итоги производственной деятельности за 9М 2026 г.", "accepted"),
            ("Энергопотребление в России снизилось на 0,8% с начала года", "accepted"),
            ('АО "Птицефабрика Чайковская" перевыполнила план по производству ВТМ', "accepted"),
            ("В США построили модель добычи платины на астероидах", "review"),
            ("Ростех создал первый Центр высокотехнологичного протезирования", "review"),
            ("До юбилейной 25-й выставки Дефектоскопия осталось 2 месяца", "review"),
            ("Путин поздравил Группу Синара с 25-летием", "rejected"),
            ("Минсельхоз объяснил ситуацию с ростом цен на красную икру", "rejected"),
            ("«Норникель» открыл Центр подбора персонала в Мончегорске", "rejected"),
            ("На АЗС «Лукойл» запущена акция — скидка на дизельное топливо", "rejected"),
            ("Завод объявил распродажу товаров", "rejected"),
            ("КРЭТ представил линейку электродвигателей для БПЛА", "accepted"),
            ('"Росэл" увеличил выпуск панелей для "чистых помещений"', "accepted"),
            ("«Норникель» представил в Китае ИИ-лабораторию новых материалов", "accepted"),
            ("Ростех представил новый робот для промышленных предприятий", "accepted"),
            ("Завод построил цех в Германии", "accepted"),
            ("Верфь построила новый ледокол", "accepted"),
            ("На заводе прошли обыски", "review"),
            ("Купить смартфон в магазине", "rejected"),
            ("Смартфон подорожал после презентации", "rejected"),
            ("Объявлен конкурс инженеров", "review"),
            ("Прототип электродвигателя успешно испытали", "accepted"),
            ("Отдел закупок заводской компании объявил о распродаже", "rejected"),
            ("В лаборатории завода проходят испытания новых сплавов", "accepted"),
            ("ОСК подготовила будущих корабелов к практике на верфях", "review"),
            ("На предприятии обсудили итоги работы", "review"),
            ("Металлурги завершили модернизацию доменной печи", "accepted"),
            ("Министерство объявило о кадровом конкурсе на энергозаводе", "review"),
            ("Компания продала промышленное предприятие", "review"),
            ("Газпром расширяет мощность переработки нефти", "accepted"),
            ("Топ 10 новостей энергетики за неделю", "rejected"),
            ("В ОЭЗ «Технополис Москва» запущено российско-китайское производство SIM-карт", "accepted"),
            ("Курс на автономию: ОСК создает отечественное беспилотное судно «река-море»", "accepted"),
            ("На Севмаше оптимизировали вывод из эллинга атомных подлодок", "accepted"),
            ("2,9 млн тонн железной руды вывезли из Еврейской АО", "accepted"),
            ("Импортозамещенный МС-21 прошел больше половины сертификационных испытаний", "accepted"),
            ("СОЛЛЕРС запускает в России полный цикл выпуска 8-ступенчатых АКПП", "accepted"),
            ("В Северске стартовала сборка корзины активной зоны БРЕСТ-ОД-300", "accepted"),
            ("На грузовых тепловозах 3ТЭ28 прошли испытания интеллектуальных систем", "accepted"),
            ("ОСК начала ходовые испытания головного круизного лайнера «Николай Жарков»", "accepted"),
            ("Для немецкого флота заказали четыре дополнительных фрегата MEKO A-200", "accepted"),
            ("Канадский флот ввел в строй шестой патрульный ледокол", "accepted"),
            ("«Синара» вывела на обкатку в Петербурге трехсекционный трамвай 71-233А", "accepted"),
            ("В войска поступила партия новых Су-34", "accepted"),
            ("Австралия начала резку стали для больших десантных кораблей", "accepted"),
            ("Роботизированную платформу «Депеша» сделали выносливее", "accepted"),
            ("«Ареал» планирует добывать титан в Тамбовской области", "review"),
            ("Эстония решила закрыть границу с Россией после попытки диверсии на заводе по выпуску дронов", "review"),
        ]
        self.assertGreaterEqual(len(cases), 30)
        for title, expected in cases:
            with self.subTest(title=title):
                self.assertEqual(classify_intake(news(title), now=TODAY).status, expected)


class Stage2URLConservativeRegressions(unittest.TestCase):
    def test_keep_nedradv_obj_and_semantic_ref(self):
        self.assertNotEqual(normalize_url("https://nedradv.ru/nedradv/ru/news?obj=1"),
                            normalize_url("https://nedradv.ru/nedradv/ru/news?obj=2"))
        self.assertNotEqual(normalize_url("https://example.com/news?ref=story-a"),
                            normalize_url("https://example.com/news?ref=story-b"))

    def test_drop_only_actual_tracking(self):
        self.assertEqual(normalize_url("http://www.example.com/article/?id=5&utm_source=mail"),
                         normalize_url("https://example.com/article?id=5"))
        self.assertEqual(normalize_url("https://example.com/news?obj=8&utm_campaign=x"),
                         normalize_url("https://example.com/news?obj=8"))

    def test_encoded_delimiter_remains_distinct(self):
        self.assertNotEqual(normalize_url("https://example.com/a%2Fb"),
                            normalize_url("https://example.com/a/b"))
        self.assertEqual(normalize_url("https://example.com/%7Earticle"),
                         normalize_url("https://example.com/~article"))

    def test_mirror_requires_content_corroboration(self):
        a = news("Верфь построила ледокол", url="https://flot.com/2026/11/18/industrial-story")
        b = news("Верфь построила ледокол", url="https://flotprom.ru/2026/11/18/industrial-story",
                 text=("Другой проект, другое событие, особый корабль и иной регион. " * 10))
        self.assertIsNone(match_exact(a, b))
        c = news("Верфь построила ледокол", url="https://flotprom.ru/2026/11/18/industrial-story")
        self.assertEqual(match_exact(a, c), "duplicate_text")

    def test_same_general_headline_new_facts_not_duplicate(self):
        a = news("Завод увеличил выпуск", url="https://one.example/a", text=BODY + "Выпуск: 1300 шт.")
        b = news("Завод увеличил выпуск", url="https://two.example/b", text=BODY + "Выпуск: 1400 шт.")
        self.assertIsNone(match_exact(a, b))


if __name__ == "__main__":
    unittest.main()
