"""Offline black/white regression cases for the Stage 4 relation-audit layer."""
from __future__ import annotations

import unittest

from processing.qwen_relation_audit import audit_relations
from processing.qwen_abstractive_summary import attach_evidence, validate_response


def material(text, title="Промышленная новость", published="2026-11-20"):
    return [{
        "id": "real:1", "title": title, "text": text, "published_at": published,
        "source": "Тест", "url": "https://example.org/1", "is_primary": True,
    }]


def issues(summary, source, title="Промышленная новость", published="2026-11-20"):
    return audit_relations(summary, material(source, title, published))["issues"]


class RelationsRegressionTests(unittest.TestCase):
    def test_reversed_growth_percentages_require_review(self):
        source = (
            "По результатам января-октября 2026 года на железную дорогу "
            "погружено 273 млн тонн, из которых 155,3 млн тонн отправлено "
            "на экспорт. В годовом сопоставлении результат вырос на 1,5% "
            "и на 4,7% соответственно."
        )
        wrong = ("Общий объем погрузки вырос на 4,7%, "
                 "а экспортный показатель — на 1,5%.")
        correct = ("Общий объем погрузки вырос на 1,5%, "
                   "а экспортный показатель — на 4,7%.")
        self.assertIn("metric_binding_mismatch:4.7:loading", issues(wrong, source))
        self.assertFalse(any(x.startswith("metric_binding_mismatch")
                             for x in issues(correct, source)))

    def test_voltage_cannot_be_misreported_as_power(self):
        source = ("Электротрактор оснащен инвертором с напряжением 220 В. "
                  "Он может служить источником питания.")
        self.assertIn("dimension_mismatch:power_vs_voltage",
                      issues("Мощность электростанции составляет 220 В.", source))
        self.assertNotIn("dimension_mismatch:power_vs_voltage",
                         issues("Напряжение на выходе составляет 220 В.", source))

    def test_project_cost_cannot_inherit_economic_effect(self):
        source = ("Экономический эффект модернизации оценивается в 1,5 млн рублей. "
                  "Затраты на проект не раскрывались.")
        wrong = "Модернизация обошлась предприятию в 1,5 млн рублей."
        correct = "Ожидаемый экономический эффект составляет 1,5 млн рублей."
        self.assertIn("financial_role_mismatch:1.5:cost", issues(wrong, source))
        self.assertNotIn("financial_role_mismatch:1.5:cost", issues(correct, source))

    def test_revenue_is_not_an_expense(self):
        source = "Выручка предприятия составила 1,5 млрд рублей."
        self.assertIn("financial_role_mismatch:1.5:cost",
                      issues("Затраты завода составили 1,5 млрд рублей.", source))
        self.assertNotIn("financial_role_mismatch:1.5:cost",
                         issues("Выручка завода составила 1,5 млрд рублей.", source))

    def test_hypothetical_budget_not_actual_investment(self):
        source = ("Рассмотрим условный расчёт. Если грузовик перевозит 20 тонн "
                  "сырья по 1 000 рублей за тонну, 50 рейсов — это 1 млн "
                  "рублей. Это не оценка фактических потерь предприятия.")
        self.assertIn("conditional_amount_presented_as_spending:1",
                      issues("Внедрение системы обошлось в 1 млн рублей.", source))
        self.assertFalse(any(x.startswith("conditional_amount_presented_as_spending")
                             for x in issues("Приведён условный расчёт на 1 млн рублей.",
                                             source)))

    def test_real_investment_not_treated_as_hypothetical(self):
        source = ("Компания инвестировала 1 млн рублей в обновление цеха. "
                  "Проект завершили и передали заказчику.")
        self.assertFalse(any(x.startswith("conditional_amount_presented_as_spending")
                             for x in issues("Компания потратила 1 млн рублей на ремонт.",
                                             source)))

    def test_explicit_spend_not_rejected_next_to_conditional_example(self):
        source = ("Рассмотрим условный расчёт. Если подрядчик потратит "
                  "1 млн рублей, срок окупаемости изменится. "
                  "Предприятие фактически инвестировало 1 млн рублей "
                  "в модернизацию производственного цеха.")
        summary = "Предприятие вложило 1 млн рублей в модернизацию."
        self.assertFalse(any(x.startswith("conditional_amount_presented_as_spending")
                             for x in issues(summary, source)))

    def test_broad_alcohol_volume_not_strong_only(self):
        source = (
            "В январе-октябре производство алкоголя (без учета пива) "
            "сократилось на 7,7%, до 128,72 млн декалитров. "
            "Розлив спиртных напитков крепче 9% сократился на 6,8%, "
            "до 84,89 млн декалитров."
        )
        self.assertIn("quantity_scope_mismatch:128.72", issues(
            "Выпуск крепкого алкоголя снизился до 128,72 млн декалитров.", source))
        self.assertNotIn("quantity_scope_mismatch:128.72", issues(
            "Производство алкоголя снизилось до 128,72 млн декалитров.", source))

    def test_finished_last_year_not_current_year(self):
        source = (
            "В 2025 году компания завершила модернизацию птицеводческого "
            "комплекса. Сейчас реконструируют другой корпус."
        )
        wrong = "В текущем году предприятие завершило модернизацию комплекса."
        correct = "В 2025 году предприятие завершило модернизацию комплекса."
        self.assertIn("completed_event_moved_to_publication_year",
                      issues(wrong, source, published="2026-11-20"))
        self.assertNotIn("completed_event_moved_to_publication_year",
                         issues(correct, source, published="2026-11-20"))

    def test_current_year_completion_is_valid(self):
        source = "В 2026 году завод завершил модернизацию оборудования."
        self.assertNotIn("completed_event_moved_to_publication_year",
                         issues("В текущем году завод завершил модернизацию.",
                                source, published="2026-11-20"))

    def test_unknown_metric_is_not_invented(self):
        source = "Организация сообщила о росте показателя на 3,2%."
        summary = "По данным компании показатель вырос на 3,2%."
        self.assertFalse(any(x.startswith("metric_binding")
                             for x in issues(summary, source)))

    def test_quality_axes_separate_relevance_and_accuracy(self):
        source = "Компания запустила новую линию производства оборудования."
        docs = material(source)
        raw = {"main_event": "Запуск линии", "is_news_event": True,
               "summary": "Компания запустила линию производства оборудования.",
               "uncertainties": [], "requires_review": False}
        check = validate_response(attach_evidence(raw, docs), docs)
        self.assertEqual(check["quality_axes"]["industrial_relevance"],
                         "not_assessed_by_summarizer")
        self.assertEqual(check["quality_axes"]["factual_accuracy"],
                         "not_established")
        self.assertEqual(check["quality_axes"]["editorial_transfer"],
                         "requires_human_approval")

    def test_integration_review_not_publication(self):
        source = (
            "По результатам января-октября на железную дорогу погружено "
            "273 млн тонн, из которых 155,3 млн тонн отправлено на экспорт. "
            "В годовом сопоставлении результат вырос на 1,5% и на 4,7% соответственно."
        )
        docs = material(source)
        summary = ("На железную дорогу погружено 273 млн тонн угля. "
                   "Общий объем погрузки вырос на 4,7%, а экспорт — на 1,5%. "
                   "Экспорт составил 155,3 млн тонн.")
        raw = {"main_event": "Данные по погрузке угля", "is_news_event": True,
               "summary": summary, "uncertainties": [], "requires_review": False}
        result = validate_response(attach_evidence(raw, docs), docs)
        self.assertFalse(result["passed"])
        self.assertTrue(any(x.startswith("metric_binding_mismatch")
                            for x in result["issues"]))
        self.assertEqual(result["relation_audit"]["verification_level"],
                         "heuristic_review_only")


if __name__ == "__main__":
    unittest.main()
