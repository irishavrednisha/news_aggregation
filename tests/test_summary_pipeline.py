"""Offline draft-only workflow contract; no database, network or Ollama."""
import unittest

from processing.summary_pipeline import rewrite_summary_draft, tagged_summary_status


SOURCE = (
    "На заводе «Север» запустили новую линию обработки металла. "
    "На предприятии установили 40 станков для выполнения заказов."
)


class SummaryPipelineTests(unittest.TestCase):
    def test_textrank_without_model_needs_no_gemma_review(self):
        result = rewrite_summary_draft(
            SOURCE, backend="textrank",
            request=lambda *args, **kwargs: self.fail("Gemma should not run"),
        )
        self.assertEqual(result.text, SOURCE)
        self.assertFalse(result.review_required)

    def test_valid_gemma_rewrite_is_never_ready_without_review(self):
        def fake(*args, **kwargs):
            return {"sentences": [
                "Завод «Север» запустил новую линию для обработки металла.",
                "На предприятии разместили 40 станков для выполнения заказов.",
            ]}
        result = rewrite_summary_draft(SOURCE, request=fake, rewrite_mode="aligned")
        self.assertTrue(result.rewritten, result.warnings)
        self.assertTrue(result.review_required)
        self.assertEqual(
            tagged_summary_status(["металлургия"], review_pending=result.review_required),
            "review_required",
        )

    def test_invalid_model_output_keeps_only_original_textrank(self):
        result = rewrite_summary_draft(
            SOURCE, rewrite_mode="aligned", request=lambda *args, **kwargs: {
                "sentences": [
                    "Завод «Север» запустил линию обработки металла.",
                    "На предприятии разместили 50 станков для выполнения заказов.",
                ]
            },
        )
        self.assertFalse(result.rewritten)
        self.assertTrue(result.review_required)
        self.assertEqual(result.text, SOURCE)
        self.assertTrue(result.warnings)

    def test_default_whole_mode_can_merge_sentences_with_review(self):
        calls = []
        def fake(messages, schema, **kwargs):
            calls.append((messages, schema))
            return {"summary": (
                "На заводе «Север» начала работу новая линия для обработки "
                "металла, а на предприятии разместили 40 станков для "
                "выполнения заказов."
            )}
        result = rewrite_summary_draft(SOURCE, request=fake)
        self.assertTrue(result.rewritten, result.warnings)
        self.assertTrue(result.review_required)
        self.assertEqual(
            tagged_summary_status(["машиностроение"], review_pending=True),
            "review_required",
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("summary", calls[0][1]["properties"])
        self.assertNotIn("sentences", calls[0][1]["properties"])

    def test_whole_mode_fallback_keeps_review_gate(self):
        result = rewrite_summary_draft(
            SOURCE, request=lambda *a, **kw: {
                "summary": ("На заводе «Север» начала работу новая линия "
                            "для обработки металла, а на предприятии "
                            "разместили 50 станков для выполнения заказов.")
            }
        )
        self.assertFalse(result.rewritten)
        self.assertEqual(result.text, SOURCE)
        self.assertTrue(result.review_required)
        self.assertIn("changed_numbers_or_dates", result.warnings)

    def test_invalid_rewrite_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            rewrite_summary_draft(SOURCE, rewrite_mode="other")

    def test_model_failure_returns_safe_extraction(self):
        def fail(*args, **kwargs):
            raise RuntimeError("Ollama offline")
        result = rewrite_summary_draft(SOURCE, request=fail)
        self.assertEqual(result.text, SOURCE)
        self.assertTrue(result.review_required)
        self.assertIn("model_error:", result.warnings[0])

    def test_reject_unrecognized_backend_and_cloud_model(self):
        with self.assertRaises(ValueError):
            rewrite_summary_draft(SOURCE, backend="something")
        with self.assertRaises(ValueError):
            rewrite_summary_draft(SOURCE, model="gemma3:cloud")
        with self.assertRaises(ValueError):
            rewrite_summary_draft(SOURCE, model="qwen3.5:4b")

    def test_tagging_preserves_review_even_after_tags_assigned(self):
        self.assertEqual(
            tagged_summary_status(["машиностроение"], review_pending=True),
            "review_required",
        )
        self.assertEqual(
            tagged_summary_status(["машиностроение"], review_pending=False),
            "ready",
        )
        self.assertEqual(
            tagged_summary_status([], review_pending=True),
            "not_for_publication",
        )


if __name__ == "__main__":
    unittest.main()
