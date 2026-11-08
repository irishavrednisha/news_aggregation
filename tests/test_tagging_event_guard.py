"""Pure offline tests for conservative headline event checks.

No model download, PostgreSQL, topic filtering or MAX access.
"""
import unittest

from processing.tagging_event_guard import nonindustrial_event_reason
from evaluate_tagging import compare_event_guard, threshold_tags


class TaggingEventGuardTests(unittest.TestCase):
    def test_unrelated_events_in_industrial_organizations_have_no_tags(self):
        examples = (
            ("ОСК провела фестиваль «Властелины глубин»", "public_or_staff_event"),
            ("Руководители судостроительных проектов прошли обучение",
             "public_or_staff_event"),
            ("На выставке представлены экспонаты истории авиации", "public_or_staff_event"),
            ("Акционеры «Норникеля» утвердили дивиденды", "corporate_finance_or_retail_promotion"),
            ("Суд передал акции завода в доход государства", "corporate_finance_or_retail_promotion"),
            ("На АЗС запущена акция – скидка на топливо", "corporate_finance_or_retail_promotion"),
            ("Погрузка угля на железную дорогу выросла", "transport_operations"),
            ("«Свеза» изменила схему отгрузки фанеры", "transport_operations"),
            ("В Калининграде пассажиропоток троллейбусов вырос", "transport_operations"),
            ("На птицефабрике завершается посев озимой пшеницы", "consumer_or_agriculture"),
            ("Приостановили работу аэропорты Пензы и Саратова", "airport_operations"),
        )
        for title, reason in examples:
            with self.subTest(title=title):
                self.assertEqual(nonindustrial_event_reason(title), reason)

    def test_technical_result_is_not_rejected_for_incidental_event(self):
        for title in (
            "На выставке представили новый станок для российской верфи",
            "На форуме разработали оборудование для очистки воды",
            "Завод запустил новую производственную линию после международной выставки",
            "Россия и Мьянма договорились о строительстве атомной станции",
            "«Свеза» вложила 30 млн рублей в модернизацию производства ДСП",
            "Ростехнадзор подтвердил готовность строительства Смоленской АЭС-2",
            "В 2027 году на маршруты Хабаровска выйдут два новых трамвая с ИИ",
            "НТМК перевёл контроль домен на видеонаблюдение",
        ):
            with self.subTest(title=title):
                self.assertIsNone(nonindustrial_event_reason(title))

    def test_guard_overrides_even_large_embedding_score_for_nonindustrial_event(self):
        row = {
            "title": "Акционеры «Норникеля» утвердили дивиденды",
            "event_guard_reason": "corporate_finance_or_retail_promotion",
            "ranked": [{
                "tag": "металлургия",
                "focus_similarity": 0.9,
                "body_similarity": 0.8,
                "keyword_boost": 0.05,
                "keyword_penalty": 0.0,
            }],
        }
        self.assertEqual(threshold_tags(row, 0.78, 0.35, 0.05, 2), [])

    def test_event_comparison_keeps_raw_scores_and_counts_zero_gold_articles(self):
        rows = [
            (
                {"title": "Акционеры завода утвердили дивиденды",
                 "tags": ["металлургия"], "url": "https://ex.test/dividends"},
                {"title": "Акционеры завода утвердили дивиденды",
                 "expected_tags": [], "url": "https://ex.test/dividends"},
            ),
            (
                {"title": "Построили новый ледокол",
                 "tags": ["судостроение"], "url": "https://ex.test/icebreaker"},
                {"title": "Построили новый ледокол",
                 "expected_tags": ["судостроение"],
                 "url": "https://ex.test/icebreaker"},
            ),
        ]
        c = compare_event_guard(rows)
        self.assertEqual(c["blocked_headlines"], 1)
        self.assertEqual(c["total"]["before"]["fp"], 1)
        self.assertEqual(c["total"]["after"]["fp"], 0)
        self.assertEqual(c["total"]["after"]["tp"], 1)


if __name__ == "__main__":
    unittest.main()
