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
