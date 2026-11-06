"""Регрессионные проверки отбора российских промышленных новостей.

Зафиксированы ситуации из пользовательского сбора 06.11.2026 (45 статей).
Тесты не требуют PostgreSQL, ML-модели или доступа к сайтам.
"""
import unittest
from datetime import date, datetime
from types import SimpleNamespace

from processing.topic_filter import (
    classify_article, filter_articles, has_bad_title,
    has_bad_content, has_bad_url, is_valid_article,
)

TODAY = date(2026, 11, 6)
ARTICLE_BODY = (
    "На российском машиностроительном предприятии установлено новое "
    "оборудование. Технология позволяет модернизировать производственную "
    "линию, увеличить выпуск продукции и снизить энергопотребление. "
    "Инженеры испытали агрегаты и подтвердили соответствие техническим "
    "параметрам. Создана инфраструктура для производства комплектующих."
)


def article(title, *, body=ARTICLE_BODY, published="2026-11-05",
            source="MASHNEWS", url="https://example.org/news/industrial-new-plant"):
    return SimpleNamespace(
        title=title, text=body, published_at=published,
        source=source, url=url
    )


class TopicFilterTests(unittest.TestCase):
    def decision(self, title, **kwargs):
        return classify_article(article(title, **kwargs), now=TODAY)

    def test_actual_stale_iapn_and_vestnik_excluded(self):
        titles = [
            ("Дроны следят за пожарами", "2025-03-11"),
            ("Модернизацию машиностроения обсудят летом в Казани", "2026-04-20"),
            ("В Невельске возрождают производство по переработке морепродуктов", "2026-09-02"),
        ]
        for title, published in titles:
            with self.subTest(title=title):
                result = self.decision(title, published=published)
                self.assertEqual(result.reason, "stale_article")
                self.assertFalse(result.accepted)

    def test_recent_infrastructure_and_production_results_preserved(self):
        titles = [
            "«Свеза» вложила 30 млн рублей в модернизацию производства ДСП в Прикамье",
            "«Свеза» на Вологодчине изменила схему отгрузки фанеры",
            "Лаявожское нефтегазоконденсатное месторождение запущено в опытно-промышленную эксплуатацию",
            "«Норникель» объявляет итоги производственной деятельности за 9М 2026 г.",
            "НТМК перевёл контроль зоны горения в домнах на видеонаблюдение",
            "Чебоксарский речной порт запустит новый контейнерный кран в 2027 году",
            "АЛРОСА впервые добыла голубой алмаз ювелирного качества",
            "«Гиредмет» подобрал технологию извлечения лития для узбекистанского месторождения",
            "Руководители представили новые технологии на выставке",
            "Ростех представил видеоаналитику для промышленных печей",
        ]
        for title in titles:
            with self.subTest(title=title):
                self.assertTrue(self.decision(title).accepted)

    def test_real_world_consumption_finance_and_pr_removed(self):
        titles_reasons = [
            ("На азс «лукойл» запущена акция – скидка на дизельное топливо",
             "offtopic_consumer_promotion"),
            ("Социальная деятельность лукойла отмечена престижными наградами",
             "offtopic_corporate_pr"),
            ("Акционеры АО «ХК Быстра» утвердили Устав компании в новой редакции",
             "offtopic_corporate_governance"),
            ("АКРА присвоило акциям «Роснефти» высший некредитный рейтинг",
             "offtopic_financial_markets"),
            ("Apple вернула себе звание самой дорогой компании в мире",
             "offtopic_financial_markets"),
            ("Котировки «Селигдара» рухнули на фоне обвинений",
             "offtopic_financial_markets"),
            ("Минсельхоз объяснил ситуацию с ростом цен на красную икру в России",
             "offtopic_consumer_prices"),
            ("ОСК провела фестиваль «Властелины глубин»",
             "offtopic_event_only"),
            ("ОСК приглашает к участию во Всероссийском инженерном конкурсе",
             "offtopic_event_only"),
            ("Руководители судостроительных проектов ОСК прошли обучение в РАНХиГС",
             "offtopic_training_only"),
            ("В ТПП Владимирской области подвели итоги работы за девять месяцев",
             "offtopic_internal_meeting"),
            ("В Казани стартует финал «Битвы роботов»",
             "offtopic_event_only"),
        ]
        for title, reason in titles_reasons:
            with self.subTest(title=title):
                decision = self.decision(title)
                self.assertEqual(decision.reason, reason)
                self.assertFalse(decision.accepted)

    def test_agricultural_support_without_production_project_is_offtopic(self):
        rejected = self.decision(
            "Запорожская область включат в федеральную программу поддержки виноделия"
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "offtopic_agriculture_policy_only")

        for title in (
            "Винодельческий завод модернизировал линию розлива",
            "Российские инженеры разработали оборудование для виноделия",
            "На предприятии запустили производство промышленных прессов "
            "для переработки винограда",
        ):
            with self.subTest(title=title):
                self.assertTrue(self.decision(title).accepted)

    def test_concrete_nonindustrial_mining_and_wine_policy_news(self):
        # Заголовки из выгрузки 06.11.2026: эти два сюжета прежде
        # попадали в предварительно принятые 21/45.
        self.assertEqual(
            self.decision(
                "Запорожская область включат в федеральную программу "
                "поддержки виноделия"
            ).reason,
            "offtopic_agricultural_policy",
        )
        self.assertEqual(
            self.decision(
                "Посольство КНР предупредило граждан об ответственности "
                "за незаконную добычу золота в России"
            ).reason,
            "offtopic_consular_notice",
        )

    def test_production_and_mine_safety_survive_narrow_exclusions(self):
        # Не баним ключевые слова «виноделие», «добыча», «золото»
        # без анализа того, о каком событии сообщает заголовок.
        for title in (
            "Винодельческий завод открыл новую автоматизированную линию розлива",
            "Предприятие разработало оборудование для переработки винограда",
            "Ростехнадзор остановил рудник «Дальполиметалла»",
            "Золотодобывающая компания запустила новую технологию извлечения золота",
            "«Яковлев» продолжает работу над гражданскими самолётами",
            "В Москву поставили 27 новых автобусов и электробусов для обучения водителей",
        ):
            with self.subTest(title=title):
                self.assertTrue(self.decision(title).accepted)

    def test_foreign_only_not_russian_technology_abroad(self):
        foreign = article(
            "В Бангладеш возобновились протесты рабочих швейных фабрик",
            body=("В Бангладеш возобновились протесты на текстильных "
                  "фабриках. Рабочие требуют повышения зарплат. " + ARTICLE_BODY.replace(
                      "российском", "местном"
                  ).replace("российского", "местного"))
        )
        # Иностранная тема без российских организаций.
        res = classify_article(foreign, now=TODAY)
        self.assertEqual(res.reason, "foreign_without_russian_connection")
        russian = self.decision(
            "В Китае «Норникель» представил новую лабораторию материалов",
            body="Норникель и российские ученые разработали технологию. " + ARTICLE_BODY
        )
        self.assertTrue(russian.accepted)
        uzbek = self.decision(
            "«Гиредмет» создал технологию лития для Узбекистана"
        )
        self.assertTrue(uzbek.accepted)

    def test_single_stopword_in_body_cannot_delete_good_news(self):
        body = (
            "Российское предприятие внедрило новую технологию. "
            "Ранее прототип демонстрировали на выставке, "
            "а кадровый резерв проходил обучение. "
            "Завод модернизирует производственные линии и "
            "выпускает новое оборудование. "
            "Также обсуждали дивиденды и вакансии работников. "
        ) + ARTICLE_BODY
        x = article("Завод внедрил систему контроля качества", body=body)
        self.assertFalse(has_bad_content(body))
        self.assertTrue(is_valid_article(x, now=TODAY))

    def test_missing_date_does_not_delete_news(self):
        d = self.decision("Завод ввел в эксплуатацию новый станок", published=None)
        self.assertTrue(d.accepted)
        self.assertEqual(d.reason, "accepted_undated")

    def test_configurable_age_boundary_and_future(self):
        x = article("Введена линия металлургического производства",
                    published="2026-10-07")
        self.assertTrue(classify_article(x, now=TODAY, max_age_days=30).accepted)
        self.assertFalse(classify_article(x, now=TODAY, max_age_days=29).accepted)
        self.assertTrue(classify_article(x, now=TODAY, max_age_days=0).accepted)
        future = article("На заводе запустили агрегат", published="2026-11-09")
        self.assertEqual(classify_article(future, now=TODAY).reason, "future_date")

    def test_navigation_rejects_exact_path_segments_not_article_word(self):
        self.assertTrue(has_bad_url("https://example.org/about/"))
        self.assertTrue(has_bad_url("https://example.org/category/industry/"))
        self.assertFalse(has_bad_url("https://example.org/news/about-new-motor"))
        self.assertFalse(has_bad_url("https://example.org/news/career-of-engineer"))
        self.assertFalse(has_bad_title("Видеоаналитика выявила отклонения в домнах"))
        self.assertTrue(has_bad_title("Дайджест новостей промышленности"))

    def test_bulk_filter_keeps_order_without_mutating_articles(self):
        keep1 = article("НТМК модернизировал оборудование")
        drop = article("На АЗС запущена акция – скидка на топливо")
        keep2 = article("На предприятии внедрен новый промышленный контроллер")
        self.assertEqual(
            filter_articles([keep1, drop, keep2], now=TODAY),
            [keep1, keep2],
        )


if __name__ == "__main__":
    unittest.main()
