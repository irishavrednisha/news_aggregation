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
            ("В США построили модель добычи платины на астероидах", "accepted"),
            ("Ростех создал первый Центр высокотехнологичного протезирования", "accepted"),
            ("До юбилейной 25-й выставки Дефектоскопия осталось 2 месяца", "rejected"),
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
            ("На заводе прошли обыски", "rejected"),
            ("Купить смартфон в магазине", "rejected"),
            ("Смартфон подорожал после презентации", "rejected"),
            ("Объявлен конкурс инженеров", "rejected"),
            ("Прототип электродвигателя успешно испытали", "accepted"),
            ("Отдел закупок заводской компании объявил о распродаже", "rejected"),
            ("В лаборатории завода проходят испытания новых сплавов", "accepted"),
            ("ОСК подготовила будущих корабелов к практике на верфях", "rejected"),
            ("На предприятии обсудили итоги работы", "accepted"),
            ("Металлурги завершили модернизацию доменной печи", "accepted"),
            ("Министерство объявило о кадровом конкурсе на энергозаводе", "rejected"),
            ("Компания продала промышленное предприятие", "accepted"),
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
            ("«Ареал» планирует добывать титан в Тамбовской области", "accepted"),
            ("Эстония решила закрыть границу с Россией после попытки диверсии на заводе по выпуску дронов", "rejected"),
        ]
        self.assertGreaterEqual(len(cases), 30)
        for title, expected in cases:
            with self.subTest(title=title):
                self.assertEqual(classify_intake(news(title), now=TODAY).status, expected)



class Stage2CorroboratedLeadRegressions(unittest.TestCase):
    def test_real_articles_with_evidence_not_keyword_only(self):
        cases = [
            (
                "РЖД рассказали о первом водородном поезде в России",
                "Заканчивается сборка первого в России водородного поезда. "
                "Поезд строят российские предприятия совместно с РЖД.",
                "accepted",
            ),
            (
                "АЛРОСА впервые добыла голубой алмаз ювелирного качества",
                "АЛРОСА добыла первый в России природный голубой алмаз "
                "редкого типа IIb. Кристалл извлечен на месторождении.",
                "accepted",
            ),
            (
                "Пермскую разработку используют в автобусах Архангельской области",
                "Оборудование пермской компании Galileosky вошло в новую "
                "систему навигации. Специалисты протестировали технологию.",
                "accepted",
            ),
            (
                "С металла какой спрос",
                "Производство стали и проката в октябре сократилось на "
                "2–4,7% относительно прошлого года. Текущие данные.",
                "accepted",
            ),
            (
                "Единая цифровая среда шахты: решения НПФ «Гранч» на «Руднике-2026»",
                "Российский разработчик представил масштабные обновления "
                "программно-аппаратного комплекса Умная шахта.",
                "accepted",
            ),
            (
                "Mercedes-Benz заморозил инвестиции в производство в Германии "
                "и допустил закрытие заводов",
                "Автоконцерн решил заморозить дальнейшие инвестиции "
                "в производство на территории Германии.",
                "accepted",
            ),
            (
                "ArianeGroup представила баллистическую противокорабельную ракету",
                "Разработчики описали характеристики нового изделия.",
                "accepted",
            ),
            (
                "Две передачи Мичкова помогли Филадельфии обыграть Баффало",
                "Сообщение о спортивном матче и игровых очках.",
                "rejected",
            ),
            (
                "Три очка Подколзина принесли Эдмонтону победу над Флоридой",
                "Сообщение о результатах хоккейного матча.",
                "rejected",
            ),
            (
                "«Ареал» планирует добывать титан в Тамбовской области",
                "Предприятие собирается инвестировать в будущий запуск участка.",
                "accepted",
            ),
            (
                "Водородный поезд примет участие в фестивале",
                "Организаторы пригласили участников выставки.",
                "rejected",
            ),
        ]
        for title, lead, expected in cases:
            with self.subTest(title=title):
                article = news(title, text=(lead + " ") * 6)
                self.assertEqual(classify_intake(article, now=TODAY).status,
                                 expected)



class Stage2BinaryRealCorpusRegressions(unittest.TestCase):
    def test_avoid_incidental_industry_words_in_unrelated_news(self):
        # Taken from real corpus errors; a keyword in a quote or body is not
        # proof that this is a primary industrial news event.
        titles = [
            "В Кемеровской области упал военно-транспортный Ан-12",
            "Силы ПВО отражают атаку БПЛА над Краснодаром",
            "Российские регионы захлестнула рекордная волна украинских дронов",
            "Работа Кандинского эпохи Баухауса установила рекорд на аукционе Christie's",
            "В аэропорту Краснодара введены дополнительные ограничения",
            "Песков сообщил, что прямая линия с Путиным пройдет в декабре",
            "Оттава выделила дополнительные средства на военную помощь Киеву",
            "Армия Кувейта сообщила, что ведет отражение налета беспилотников",
            "Первопричина, приведшая к сбою в аэропортах США, пока не установлена",
            "Американские силы на Ближнем Востоке дополнила третья авианосная ударная группа",
            "Си Цзиньпин призвал развивать партнерство в сфере технологических инноваций",
            "Beget лишился статуса регистратора доменных имен в Рунете",
            "В Якутии обратили три бриллианта в доход государства",
        ]
        for title in titles:
            with self.subTest(title=title):
                article = news(title, text=(title + " В сообщении указаны "
                                             "технологические аспекты, "
                                             "средства производства, организации "
                                             "и промышленность. ") * 5)
                self.assertEqual(classify_intake(article, now=TODAY).status,
                                 "rejected")

    def test_industrial_news_not_lost(self):
        cases = [
            "В 2026 году вложения в химическую отрасль составят 2,3 трлн рублей",
            "Владимир Цитцер: Птицезащита должна закладываться в проекты ЛЭП изначально — это снизит затраты на 20%",
            "ОСК начала ходовые испытания головного круизного лайнера «Николай Жарков»",
            "Ростех оснастил аэродром в Санкт-Петербурге новым комплексом управления полетами",
            "Импортозамещенный МС-21 прошел больше половины сертификационных испытаний",
            "СОЛЛЕРС запустил серийное производство бензиновых турбодвигателей в Елабуге",
            "АриaneGroup представила баллистическую противокорабельную ракету",
        ]
        for title in cases:
            with self.subTest(title=title):
                article = news(title, text=(title + " Предприятия производят "
                                             "оборудование, проводят внедрение и "
                                             "инженерную работу в отрасли. ") * 5)
                self.assertEqual(classify_intake(article, now=TODAY).status,
                                 "accepted")

    def test_event_invitation_is_not_new_industrial_project(self):
        cases = [
            "Международные специализированные выставки «Металлоконструкции», «Металлургия.Литмаш», «Сервисные Металлоцентры»",
            "KIOSH 2026: ключевая площадка по охране труда и промышленной безопасности в Центральной Азии",
            "«Рексофт» проведёт бесплатный вебинар о внедрении компьютерного зрения в добывающей отрасли",
            "Модернизацию машиностроения обсудят летом в Казани",
        ]
        for title in cases:
            with self.subTest(title=title):
                text = ("С 26 по 28 мая состоится выставка промышленного "
                        "оборудования. Организаторы приглашают компании и "
                        "рассказывают о программе мероприятия. ") * 5
                self.assertEqual(
                    classify_intake(news(title, text=text, published="2026-11-17"),
                                    now=TODAY).status,
                    "rejected",
                )


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
