"""CPU-only tests for conservative TextRank -> Gemma paraphrase (no Ollama)."""
import unittest

from processing.extractive_rewriter import (
    REWRITE_FORMAT, check_paraphrase, rewrite_extractive, split_extractive_sentences,
)


class RewriteTests(unittest.TestCase):
    def test_split_preserves_abbreviation_and_order(self):
        text = "На заводе им. Кирова запущена линия. Мощность составляет 40 изделий."
        self.assertEqual(
            split_extractive_sentences(text),
            ["На заводе им. Кирова запущена линия.",
             "Мощность составляет 40 изделий."],
        )

    def test_valid_rewrite_is_same_number_of_sentences_and_not_shortened(self):
        src = (
            "На предприятии «КАМАЗ» завершили реконструкцию цеха. "
            "В новом помещении установили 40 станков."
        )
        answers = {
            "sentences": [
                "Предприятие «КАМАЗ» завершило работы по реконструкции цеха.",
                "В обновлённом помещении разместили 40 станков.",
            ]
        }
        calls = []

        def fake(messages, schema, **kwargs):
            calls.append((messages, schema, kwargs))
            return answers

        result = rewrite_extractive(src, request=fake)
        self.assertTrue(result.rewritten, result.warnings)
        self.assertTrue(result.review_required)
        self.assertFalse(result.publication_performed)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], REWRITE_FORMAT)
        self.assertIn("КАМАЗ", result.text)

    def test_new_company_or_number_causes_fallback(self):
        original = (
            "Завод «Север» запустил новую линию обработки металла. "
            "Предприятие установило 40 станков для выполнения заказов."
        )
        def fake(messages, schema, **kwargs):
            return {"sentences": [
                "Компания «Сфера-Проект-113» запустила новую линию обработки металла.",
                "Предприятие установило 50 станков для выполнения заказов.",
            ]}
        result = rewrite_extractive(original, request=fake)
        self.assertFalse(result.rewritten)
        self.assertEqual(result.text, result.extractive)
        self.assertIn("Сфера-Проект-113", result.model_candidate)
        self.assertTrue(any("changed_numbers" in warning or "changed_quoted" in warning
                            for warning in result.warnings))

    def test_reject_shortening_and_drop(self):
        source = "Компания начала выпуск приборов для измерения температуры."
        result = rewrite_extractive(
            source, request=lambda *a, **kw: {"sentences": ["Выпустили приборы."]}
        )
        self.assertFalse(result.rewritten)
        self.assertTrue(any("length" in w for w in result.warnings))
        result = rewrite_extractive(
            source, request=lambda *a, **kw: {"sentences": []}
        )
        self.assertEqual(result.warnings, ["sentence_count_changed"])

    def test_new_location_mid_sentence_is_not_accepted(self):
        original = "На заводе КАМАЗа модернизировали контроль печей."
        rewritten = "В Платово на заводе КАМАЗа модернизировали контроль печей."
        self.assertTrue(any(w.startswith("new_capitalized_token:")
                            for w in check_paraphrase(original, rewritten)))

    def test_lost_plan_guard(self):
        source = "Компания планирует выпустить новую серию трансформаторов."
        other = "Компания уже выпустила новую серию трансформаторов."
        self.assertIn("lost_uncertainty_or_plan_marker", check_paraphrase(source, other))

    def test_technical_opportunities_is_not_uncertainty(self):
        # Regression: "возможности" is not a hedge, it means capability.
        source = (
            "Разработка открывает широкие возможности для предприятий отрасли."
        )
        paraphrase = (
            "Эта разработка создаёт новые перспективы для предприятий отрасли."
        )
        self.assertNotIn(
            "lost_uncertainty_or_plan_marker",
            check_paraphrase(source, paraphrase),
        )

    def test_missing_final_period_causes_safe_fallback_and_keeps_candidate(self):
        src = "Государственная Дума приняла новый закон об аттракционах."
        raw = "Государственная Дума утвердила новый закон об аттракционах,"
        result = rewrite_extractive(
            src, request=lambda *args, **kwargs: {"sentences": [raw]}
        )
        self.assertFalse(result.rewritten)
        self.assertIn("sentence_1:sentence_missing_terminal_punctuation",
                      result.warnings)
        self.assertEqual(result.text, src)
        self.assertEqual(result.model_candidate, raw)

    def test_negation_flip_falls_back(self):
        src = "Компания не выпустила новую линейку оборудования."
        bad = "Компания уже выпустила новую линейку оборудования."
        self.assertIn("changed_negation", check_paraphrase(src, bad))

    def test_plan_cannot_become_completed_action(self):
        src = "Компания планирует запустить новую линию оборудования."
        bad = "Компания запустила новую линию оборудования."
        warnings = check_paraphrase(src, bad)
        self.assertIn("lost_plan_marker", warnings)
        self.assertIn("introduced_completed_event", warnings)

    def test_number_context_change_is_rejected(self):
        src = "Предприятие установило 40 станков для новых заказов."
        bad = "Предприятие установило 40 предприятий для новых заказов."
        self.assertIn("changed_number_context", check_paraphrase(src, bad))

    def test_gemma5_rusal_co2_direction_reversal_is_rejected(self):
        src = ("При производстве одного килограмма оксида скандия "
               "будет поглощаться примерно 1,8 тонны CO2.")
        hallucination = ("В процессе производства одного килограмма "
                         "оксида скандия будет выбрасываться примерно "
                         "1,8 тонны углекислого газа.")
        self.assertIn(
            "changed_direction:co2_absorption_emission",
            check_paraphrase(src, hallucination),
        )

    def test_gemma5_opposite_growth_direction_is_rejected(self):
        src = "Компания увеличила производственные мощности на 40 процентов."
        other = "Компания сократила производственные мощности на 40 процентов."
        self.assertIn(
            "changed_direction:increase_decrease",
            check_paraphrase(src, other),
        )

    def test_no_false_positive_for_absorption_rewording(self):
        src = "При производстве поглощается 1,8 тонны углекислого газа."
        other = "В процессе производства улавливается 1,8 тонны углекислого газа."
        self.assertNotIn(
            "changed_direction:co2_absorption_emission",
            check_paraphrase(src, other),
        )

    def test_thousands_abbreviation_is_not_a_sentence_boundary(self):
        src = ("Предприятие расширило выпуск до 580 тыс. изделий в год. "
               "В 2026 году открыли дополнительную линию.")
        self.assertEqual(
            split_extractive_sentences(src),
            ["Предприятие расширило выпуск до 580 тыс. изделий в год.",
             "В 2026 году открыли дополнительную линию."],
        )

    def test_gemma10_rusal_future_investment_not_completed(self):
        source = "Общий объем инвестиций в проект составит около 500 миллионов рублей."
        candidate = "Общий объем инвестиций в проект составил около 500 миллионов рублей."
        self.assertIn(
            "changed_tense_future_to_past:investment_or_quantity",
            check_paraphrase(source, candidate),
        )
        result = rewrite_extractive(source, request=lambda *a, **kw: {
            "sentences": [candidate],
        })
        self.assertFalse(result.rewritten)
        self.assertEqual(result.text, source)

    def test_completed_investment_cannot_become_future(self):
        source = "Общий объем инвестиций в проект составил около 500 миллионов рублей."
        candidate = "Общий объем инвестиций в проект составит около 500 миллионов рублей."
        self.assertIn(
            "changed_tense_past_to_future:investment_or_quantity",
            check_paraphrase(source, candidate),
        )

    def test_same_tense_is_allowed(self):
        source = "Общий объем инвестиций в проект составит около 500 миллионов рублей."
        candidate = "Инвестиции в проект составят около 500 миллионов рублей."
        self.assertNotIn(
            "changed_tense_future_to_past:investment_or_quantity",
            check_paraphrase(source, candidate),
        )

    def test_gemma10_reject_unsubstantiated_comparison_target(self):
        source = "Стоимость отечественного изделия в 1,5 раза ниже."
        candidate = "Стоимость отечественного изделия в 1,5 раза ниже, чем у зарубежных аналогов."
        self.assertIn(
            "introduced_comparison_target",
            check_paraphrase(source, candidate),
        )

    def test_malformed_model_output_falls_back(self):
        text = "Компания построила новый завод."
        for reply in ({"summary": text}, {"sentences": ["OK", "EXTRA"]}):
            result = rewrite_extractive(text, request=lambda *a, **kw: reply)
            self.assertFalse(result.rewritten)
        result = rewrite_extractive(
            text, request=lambda *a, **kw: (_ for _ in ()).throw(
                RuntimeError("not found")
            )
        )
        self.assertFalse(result.rewritten)
        self.assertTrue(result.warnings[0].startswith("model_error:"))


if __name__ == "__main__":
    unittest.main()
