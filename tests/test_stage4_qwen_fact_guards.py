"""Real-report regression fixtures for Qwen v2.1: no Ollama or PostgreSQL."""
from __future__ import annotations

import unittest

from processing.qwen_fact_guards import (
    _explicit_physical_units, equivalent_explicit_units, focused_fact_issues,
)
from processing.qwen_abstractive_summary import attach_evidence, validate_response


def materials(text: str, title="Промышленное событие"):
    return [{"id": "original:1", "title": title, "text": text, "is_primary": True}]


def result(summary, source, title="Промышленное событие"):
    docs = materials(source, title)
    raw = {
        "main_event": "Промышленное событие",
        "is_news_event": True,
        "summary": summary,
        "uncertainties": [],
        "requires_review": False,
    }
    return validate_response(attach_evidence(raw, docs), docs)


class SourceFactsTests(unittest.TestCase):
    def test_tonne_abbreviation_is_explicitly_equivalent(self):
        source = ("На заводе запущена линия фасовки удобрений. "
                  "Линия сможет фасовать около 70 т в час. "
                  "Оборудование уже установлено на предприятии.")
        text = ("Предприятие запустило линию фасовки удобрений. "
                "Её производительность составляет около 70 тонн в час. "
                "Оборудование установлено на заводе.")
        self.assertTrue(equivalent_explicit_units("70", text, source))
        check = result(text, source)
        self.assertNotIn("number_unit_mismatch:70", check["issues"])
        self.assertNotIn("claim_2_unit_not_in_evidence", check["issues"])

    def test_metre_abbreviation_is_explicitly_equivalent(self):
        source = ("На предприятии представлен новый испытательный комплекс. "
                  "Максимальная дальность оборудования достигает 1000 м. "
                  "Испытания оборудования начались на заводе.")
        text = ("Завод представил новый испытательный комплекс. "
                "Дальность работы оборудования составляет до 1000 метров. "
                "Выполнение программы испытаний уже началось.")
        self.assertTrue(equivalent_explicit_units("1000", text, source))
        check = result(text, source)
        self.assertNotIn("number_unit_mismatch:1000", check["issues"])
        self.assertNotIn("claim_2_unit_not_in_evidence", check["issues"])

    def test_other_unit_is_not_misclassified_as_same(self):
        source = "Испытательная трасса имеет длину 1000 м."
        text = "Испытательная трасса имеет длину 1000 километров."
        self.assertFalse(equivalent_explicit_units("1000", text, source))

    def test_units_with_different_physical_meaning_stay_blocked(self):
        source = ("На промышленной площадке открыли новый комплекс испытаний. "
                  "Установка рассчитана на нагрузку в 70 кг. "
                  "Компания представила оборудование местным предприятиям.")
        text = ("Компания открыла новый испытательный комплекс. "
                "Оборудование рассчитано на нагрузку 70 тонн. "
                "Установка представлена местным предприятиям.")
        check = result(text, source)
        self.assertFalse(check["passed"])
        self.assertIn("number_unit_mismatch:70", check["issues"])

    def test_planned_jobs_cannot_be_reported_as_created(self):
        source = ("На заводе запустили первую очередь выпуска труб. "
                  "В проект инвестировано 1 млрд рублей, будет создано 103 рабочих места. "
                  "Предприятие реконструировало цех и построило ветку длиной 820 метров.")
        summary = ("Предприятие запустило первую очередь трубного завода. "
                   "Инвестиции позволили создать 103 рабочих места. "
                   "Компания реконструировала цех и построила новую ветку.")
        check = result(summary, source)
        self.assertIn("planned_jobs_reported_as_created", check["issues"])

    def test_future_job_commitment_remains_future(self):
        source = "Будет создано 103 рабочих места после запуска второй очереди."
        summary = "После запуска второй очереди планируется создать 103 рабочих места."
        self.assertNotIn("planned_jobs_reported_as_created",
                         focused_fact_issues(summary, materials(source)))

    def test_navigation_exception_cannot_be_silently_dropped(self):
        source = ("В Московском бассейне завершилась навигация. "
                  "Движение судов прекращается на всех участках, "
                  "только работа Рыбинского шлюза продлена до 25 ноября 2026 года. "
                  "Начался межнавигационный ремонт гидротехнических сооружений.")
        summary = ("В Московском бассейне завершилась навигация. "
                   "Движение судов прекращается на всех участках. "
                   "Начался ремонт гидротехнических сооружений.")
        check = result(summary, source)
        self.assertIn("source_exception_omitted", check["issues"])

    def test_navigation_with_exception_mentioned_not_rejected(self):
        source = ("Навигация завершилась, только работа Рыбинского шлюза продлена до "
                  "25 ноября 2026 года. Движение судов прекращается на всех участках.")
        summary = ("Движение судов прекращается на всех участках, "
                   "кроме Рыбинского шлюза.")
        self.assertNotIn("source_exception_omitted",
                         focused_fact_issues(summary, materials(source)))

    def test_wrong_polymer_attribute_rejected(self):
        source = ("Концерн выполнил контракт на поставку снайперских винтовок. "
                  "Винтовка имеет телескопический приклад и рукоятку управления "
                  "из ударопрочного полимера. Оборудование отгружено заказчику.")
        summary = ("Предприятие выполнило контракт на поставку винтовок. "
                   "Изделие оснащено эргономичным полимерным прикладом. "
                   "Поставка завершилась в срок.")
        check = result(summary, source)
        self.assertIn("unsupported_material_binding:polymer_stock", check["issues"])

    def test_source_supported_polymer_stock_not_rejected(self):
        source = "Приклад выполнен из ударопрочного полимера."
        summary = "Винтовка оснащена полимерным прикладом."
        self.assertNotIn("unsupported_material_binding:polymer_stock",
                         focused_fact_issues(summary, materials(source)))

    def test_unmentioned_calendar_year_is_not_whitelisted(self):
        source = ("Концерн выполнил обязательства по поставке винтовок. "
                  "По сравнению с 2025 годом объёмы выпуска выросли. "
                  "Изделия отгружены основному заказчику.")
        summary = ("Концерн отгрузил изделия по контракту. "
                   "В 2026 году выпуск вырос по сравнению с 2025 годом. "
                   "Предприятие завершило исполнение контракта.")
        check = result(summary, source)
        self.assertIn("unsupported_number:2026", check["issues"])


if __name__ == "__main__":
    unittest.main()
