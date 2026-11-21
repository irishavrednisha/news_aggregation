"""Offline contract/regression tests for bounded Qwen -> Gemma, no Ollama or DB."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from processing import qwen_gemma_final as f
from scripts import stage4_final_acceptance as runner

S1 = "Завод «Маяк» планирует открыть линию в Туле в декабре 2026 года."
S2 = "Новая линия рассчитана на выпуск 20 деталей в смену."
S3 = "Перед запуском предприятие проведёт испытания оборудования."
DRAFT = S1 + " " + S2 + " " + S3
SOURCE = [{
    "id": "news-1",
    "source": "Пример",
    "url": "https://example.org/industry/one",
    "title": "Завод «Маяк» планирует открыть линию",
    "text": DRAFT + " Это предварительный план развития предприятия. "
            "Промышленное оборудование находится на стадии подготовки к испытаниям.",
    "published_at": "2026-11-20",
    "is_primary": True,
}]
QUOTES = [S1, S2, S3]


def review(verdict="supported", quote_override=None):
    return {
        "claims": [{
            "claim_id": i,
            "verdict": verdict if i == 1 else "supported",
            "reason": "Сверено с оригиналом" if verdict == "supported" else "Изменён статус",
            "error_type": "" if verdict == "supported" else "temporal_status",
            "source_id": "news-1",
            "evidence_quote": (
                quote_override if i == 1 and quote_override is not None else line),
            "suggested_correction": "",
        } for i, line in enumerate(QUOTES, 1)],
        "overall_comment": "Проверено",
    }


class GemmaFake:
    def __init__(self, edits=None, reviewer=None, fail_step=None):
        self.results = [reviewer if reviewer is not None else review(),
                        edits if edits is not None else {"summary": DRAFT}]
        self.calls = []
        self.fail_step = fail_step

    def __call__(self, messages, schema, *, model, options):
        self.calls.append((model, schema))
        if self.fail_step == len(self.calls):
            raise TimeoutError("Ollama offline")
        if model != f.GEMMA_MODEL or len(self.calls) > 2:
            raise AssertionError("Unexpected/unbounded model invocation")
        return self.results[len(self.calls) - 1], {"eval_count": 10}


def fake_qwen(**changes):
    value = {
        "qwen_summary": DRAFT,
        "main_event": "План нового производства",
        "claims": [
            {"claim": line, "source_ids": ["news-1"], "evidence_quotes": [line]}
            for line in QUOTES
        ],
        "validation": {"passed": True, "issues": [], "flags": []},
        "status": "summary_draft",
        "attempts": [{"number": 1}],
    }
    value.update(changes)
    return value


class Stage4FinalTests(unittest.TestCase):
    def test_guard_changes_in_numbers_status_negation_and_conditions(self):
        wrong = DRAFT.replace("20 деталей", "40 деталей").replace(
            "планирует открыть", "открыл"
        )
        found = f._edit_differences(DRAFT, wrong)
        self.assertIn("edit_changed_numbers_or_units", found)
        self.assertIn("edit_changed_planned", found)
        self.assertIn("edit_changed_completed", found)
        self.assertIn("edit_changed_conditional", f._edit_differences(
            "Если закон примут, завод начнёт работу.", "Завод начнёт работу."))
        self.assertIn("edit_changed_negation", f._edit_differences(
            "Завод не завершил испытания.", "Завод завершил испытания."))

    def test_verbatim_evidence_required_and_all_claims_covered(self):
        bad = f._review_check(review(quote_override="Это не текст источника."), DRAFT, SOURCE)
        self.assertIn("gemma_review_unverifiable_quote", bad)
        incomplete = review()
        incomplete["claims"].pop()
        self.assertIn("gemma_review_missing_claims",
                      f._review_check(incomplete, DRAFT, SOURCE))

    def test_clean_edit_is_selected_but_never_published(self):
        model = GemmaFake()
        old = json.dumps(SOURCE, ensure_ascii=False, sort_keys=True)
        with patch.object(f, "summarize_cluster", return_value=fake_qwen()), patch.object(
                f, "_validate_variant", return_value={"passed": True, "issues": []}):
            result = f.summarize_final("event-1", SOURCE, request=model)
        self.assertEqual(result["status"], "ready_for_moderation")
        self.assertEqual(result["selected_version"], "gemma")
        self.assertEqual(result["final_summary"], DRAFT)
        self.assertEqual(result["model_requests"]["qwen"], 1)
        self.assertEqual(result["model_requests"]["gemma_fact_review"], 1)
        self.assertEqual(result["model_requests"]["gemma_edit"], 1)
        self.assertFalse(result["auto_publication_allowed"])
        self.assertEqual(json.dumps(SOURCE, ensure_ascii=False, sort_keys=True), old)

    def test_corrupt_gemma_edit_falls_back_to_verified_qwen(self):
        bad_summary = DRAFT.replace("20 деталей", "40 деталей")
        with patch.object(f, "summarize_cluster", return_value=fake_qwen()), patch.object(
                f, "_validate_variant", return_value={"passed": True, "issues": []}):
            result = f.summarize_final("event-1", SOURCE,
                                       request=GemmaFake(edits={"summary": bad_summary}))
        self.assertEqual(result["selected_version"], "qwen")
        self.assertEqual(result["final_summary"], DRAFT)
        self.assertIn("edit_changed_numbers_or_units", result["validation_issues"])

    def test_review_disagreement_never_greenlights_draft(self):
        model = GemmaFake(reviewer=review(verdict="contradicted"))
        with patch.object(f, "summarize_cluster", return_value=fake_qwen()), patch.object(
                f, "_validate_variant", return_value={"passed": True, "issues": []}):
            result = f.summarize_final("event-1", SOURCE, request=model)
        self.assertEqual(result["status"], "needs_manual_review")
        self.assertIn("gemma_review_contradicted_claim_1",
                      result["validation_issues"])

    def test_qwen_guard_cannot_be_silenced_by_gemma(self):
        model = GemmaFake()
        invalid = fake_qwen(validation={"passed": False,
                                         "issues": ["planned_event_changed_to_completed"]})
        with patch.object(f, "summarize_cluster", return_value=invalid), patch.object(
                f, "_validate_variant", return_value={"passed": True, "issues": []}):
            result = f.summarize_final("event-1", SOURCE, request=model)
        self.assertEqual(result["status"], "needs_manual_review")
        self.assertIn("qwen_guard_rejected_candidate", result["review_reasons"])

    def test_ollama_failure_is_processing_error_with_qwen_saved(self):
        with patch.object(f, "summarize_cluster", return_value=fake_qwen()):
            result = f.summarize_final("event-1", SOURCE,
                                       request=GemmaFake(fail_step=1))
        self.assertEqual(result["status"], "processing_error")
        self.assertEqual(result["qwen_summary"], DRAFT)
        self.assertFalse(result["final_summary"])

    def test_empty_source_does_not_invoke_model(self):
        result = f.summarize_final("event-1", [])
        self.assertEqual(result["status"], "not_for_publication")
        self.assertEqual(result["model_requests"]["qwen"], 0)

    def test_local_endpoint_enforced(self):
        with self.assertRaises(ValueError):
            f.SequentialOllama("https://example.com:11434")

    def test_resume_keeps_completed_event_and_rejects_input_change(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            inp = folder / "manifest.json"
            out = folder / "out"
            data = {"stage": 4, "clusters": [
                {"cluster_id": "event-1", "source_materials": SOURCE},
            ]}
            inp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            item = {
                "cluster_id": "event-1", "source_sha256": f.source_digest(SOURCE),
                "status": "needs_manual_review", "selected_version": "none",
                "model_versions": {"pipeline": f.VERSION},
                "model_requests": {"qwen": 1}, "validation_issues": [],
                "processing_time": 1.0,
            }
            with patch.object(runner, "summarize_final", return_value=item) as mocked:
                one = runner.generate(inp, out, run_tests=False)
                two = runner.generate(inp, out, run_tests=False)
            self.assertEqual(one["processed"], 1)
            self.assertEqual(two["processed"], 1)
            self.assertEqual(mocked.call_count, 1)
            payload = json.loads((out / "stage4_final_results.json").read_text(
                encoding="utf-8"))
            self.assertEqual(payload["clusters"][0]["source_sha256"],
                             f.source_digest(SOURCE))
            data["clusters"][0]["source_materials"][0]["text"] += " Новый факт."
            inp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.generate(inp, out, run_tests=False)

    def test_retry_json_failure_only_once(self):
        bad_json = fake_qwen(
            qwen_summary="", validation={"passed": False,
                                         "issues": ["generation_error:JSONDecodeError"]},
        )
        counter = [bad_json, fake_qwen()]
        with patch.object(f, "summarize_cluster", side_effect=counter) as spy, patch.object(
                f, "_validate_variant", return_value={"passed": True, "issues": []}):
            result = f.summarize_final("event-1", SOURCE, request=GemmaFake())
        self.assertEqual(spy.call_count, 2)
        self.assertEqual(result["model_requests"]["qwen"], 2)
        self.assertEqual(result["status"], "ready_for_moderation")


if __name__ == "__main__":
    unittest.main()
