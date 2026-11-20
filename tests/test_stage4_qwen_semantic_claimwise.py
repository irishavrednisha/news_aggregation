"""Offline regressions for advisory *claim-wise* Gemma review.

No Ollama, PostgreSQL, GPU, or MAX. Fakes only verify control flow,
schema recovery, citation provenance, and no publication permission.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from processing.qwen_abstractive_summary import canonical_materials, source_digest
from processing.qwen_semantic_claimwise import (
    _valid_fact, _valid_language, review_cluster_claimwise,
)
from scripts.stage4_semantic_claimwise_audit import run


def row(summary=None, source=None):
    docs = canonical_materials([{
        "id": "doc", "source": "Проверочный источник",
        "url": "https://example.org/event",
        "title": "Завод выпустил оборудование",
        "text": source or ("Завод выпустил оборудование для шахт. "
                           "На предприятии завершились испытания установок. "
                           "Устройства поставлены заказчикам."),
        "published_at": "2026-11-20", "is_primary": True,
    }])
    return {
        "cluster_id": 7, "source_materials": docs,
        "source_sha256": source_digest(docs),
        "qwen_summary": (summary if summary is not None else
                         "Завод выпустил оборудование для шахт. "
                         "Предприятие завершило испытания установок. "
                         "Устройства поставлены заказчикам."),
        "main_event": "Выпуск оборудования",
    }


class FakeOneByOne:
    def __init__(self, bad_once=False, verdict="supported", language=False):
        self.calls = 0
        self.bad_once = bad_once
        self.verdict = verdict
        self.language = language

    def __call__(self, messages, schema, *, model, options):
        assert model == "gemma3:4b"
        self.calls += 1
        user = json.loads(messages[1]["content"])
        if "claim" in user:
            if self.bad_once:
                self.bad_once = False
                return {"verdict": "supported", "reason": "",
                        "quote_indices": ["zero"]}, {}
            return {"verdict": self.verdict, "reason": "Проверены субъект и действие",
                    "quote_indices": [0]}, {}
        return {"needs_edit": self.language,
                "notes": ["Исправить выражение «Завод выпустил оборудование»"] if self.language else []}, {}


class ClaimWiseTests(unittest.TestCase):
    def test_audit_has_separate_calls_per_claim_and_language(self):
        fake = FakeOneByOne()
        reviewed = review_cluster_claimwise(row(), request=fake)
        self.assertEqual(fake.calls, 4)
        self.assertEqual(reviewed["model_calls"], 4)
        # An extractive/near-verbatim summary can independently require review.
        self.assertIn(reviewed["status"],
                      ("review_required", "advisory_no_issues_detected"))
        self.assertFalse(reviewed["publication_allowed"])
        self.assertFalse(reviewed["semantic_accuracy_certified"])
        self.assertEqual(len(reviewed["answer"]["claims"]), 3)
        self.assertEqual(reviewed["answer"]["language"]["needs_edit"], False)

    def test_bad_fact_schema_retried_once(self):
        fake = FakeOneByOne(bad_once=True)
        reviewed = review_cluster_claimwise(row(), request=fake)
        self.assertEqual(fake.calls, 5)
        # An extractive/near-verbatim summary can independently require review.
        self.assertIn(reviewed["status"],
                      ("review_required", "advisory_no_issues_detected"))
        self.assertIn("invalid_response",
                      reviewed["claim_diagnostics"][0]["attempts"][0])
        self.assertEqual(len(reviewed["claim_diagnostics"][0]["attempts"]), 2)

    def test_bad_fact_schema_twice_is_review_required_and_traceable(self):
        class AlwaysBad(FakeOneByOne):
            def __call__(self, messages, schema, *, model, options):
                if "claim" in json.loads(messages[1]["content"]):
                    self.calls += 1
                    return {"verdict": "supported", "reason": "",
                            "quote_indices": []}, {}
                return super().__call__(messages, schema, model=model, options=options)
        reviewed = review_cluster_claimwise(row(), request=AlwaysBad())
        self.assertEqual(reviewed["status"], "review_required")
        self.assertIn("claim_1_empty_fact_reason", reviewed["issues"])
        self.assertFalse(reviewed["publication_allowed"])

    def test_fake_model_supported_cannot_override_real_deterministic_block(self):
        source = ("Электротрактор оснащён инвертором с напряжением 220 В. "
                  "Это специальное оборудование для удалённых участков. "
                  "Поставщики планируют расширить линейку.")
        summary = ("Мощность мобильной станции составляет 220 В. "
                   "Оборудование предназначено для удалённых участков. "
                   "Компания планирует расширить линейку.")
        reviewed = review_cluster_claimwise(
            row(summary=summary, source=source), request=FakeOneByOne())
        self.assertIn("deterministic_review_required", reviewed["issues"])
        self.assertIn("dimension_mismatch:power_vs_voltage",
                      reviewed["mechanical_issues"])
        self.assertEqual(reviewed["status"], "review_required")

    def test_model_contradiction_is_advisory_but_rejected(self):
        reviewed = review_cluster_claimwise(
            row(), request=FakeOneByOne(verdict="contradicted"))
        self.assertIn("claim_1_semantic_contradicted", reviewed["issues"])
        self.assertFalse(reviewed["semantic_accuracy_certified"])

    def test_language_requires_specific_note(self):
        self.assertEqual(_valid_language({"needs_edit": True, "notes": []}),
                         "language_flag_without_example")
        self.assertIsNone(_valid_language({
            "needs_edit": True, "notes": ["Пропущено подлежащее"]}))
        sample = "Завод выпустил оборудование для шахт."
        self.assertEqual(_valid_language({
            "needs_edit": True, "notes": ["Некоторые фразы можно упростить."]
        }, sample), "language_flag_without_source_excerpt")
        self.assertIsNone(_valid_language({
            "needs_edit": True,
            "notes": ["Неверно использовано «Завод выпустил оборудование»."]
        }, sample))
        fake = FakeOneByOne(language=True)
        reviewed = review_cluster_claimwise(row(), request=fake)
        self.assertIn("model_flagged_russian_language", reviewed["language_flags"])
        self.assertNotIn("model_flagged_russian_language", reviewed["issues"])

    def test_invalid_model_types_fail_closed(self):
        self.assertEqual(_valid_fact({"verdict": [], "reason": "text",
                                     "quote_indices": [0]}, 1),
                         "invalid_fact_verdict")
        self.assertEqual(_valid_fact({"verdict": "supported", "reason": "text",
                                     "quote_indices": ["zero"]}, 1),
                         "invalid_fact_quote_indices")

    def test_original_qwen_uncertainty_is_not_discarded(self):
        original = row()
        original["uncertainties"] = ["В источнике нет подтверждения итогов проекта"]
        reviewed = review_cluster_claimwise(
            original, request=FakeOneByOne())
        self.assertEqual(reviewed["status"], "review_required")
        self.assertIn("model_reported_uncertainties",
                      reviewed["mechanical_issues"])
        self.assertIn("deterministic_review_required", reviewed["issues"])

    def test_original_qwen_request_for_review_is_not_discarded(self):
        original = row()
        original["attempts"] = [{"response": {
            "summary": original["qwen_summary"], "requires_review": True,
        }}]
        reviewed = review_cluster_claimwise(
            original, request=FakeOneByOne())
        self.assertEqual(reviewed["status"], "review_required")
        self.assertIn("model_requested_review", reviewed["mechanical_issues"])

    def test_malformed_original_uncertainties_fails_closed(self):
        original = row()
        original["uncertainties"] = "not a list"
        fake = FakeOneByOne()
        reviewed = review_cluster_claimwise(original, request=fake)
        self.assertIn("original_uncertainties_invalid", reviewed["issues"])
        self.assertEqual(fake.calls, 0)

    def test_no_summary_is_not_a_model_call(self):
        fake = FakeOneByOne()
        reviewed = review_cluster_claimwise(row(summary=""), request=fake)
        self.assertEqual(fake.calls, 0)
        self.assertIn("no_summary", reviewed["issues"])

    def test_one_run_resume_does_not_reuse_old_batch_or_repeat_good_results(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, target = root/"src.json", root/"audit.json"
            source.write_text(json.dumps({"stage":4,"clusters":[row()]},
                                         ensure_ascii=False), encoding="utf-8")
            fake = FakeOneByOne()
            first = run(source, target, request=fake)
            self.assertTrue(first["complete"])
            self.assertTrue(first["schema_all_valid"])
            self.assertEqual(first["metrics"]["valid_model_claim_verdicts"], 3)
            self.assertEqual(first["metrics"]["model_calls"], 4)
            self.assertEqual(fake.calls, 4)
            second = run(source, target, request=fake, resume=True)
            self.assertEqual(fake.calls, 4)
            self.assertTrue(second["complete"])
            self.assertFalse(second["auto_publication"])

    def test_disconnect_stops_and_saves_checkpoint(self):
        class Down:
            def __call__(self, *args, **kwargs):
                raise ConnectionError("offline")
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, target = root/"src.json", root/"audit.json"
            source.write_text(json.dumps({"stage":4,"clusters":[row(), dict(row(), cluster_id=8)]},
                                         ensure_ascii=False),encoding="utf-8")
            report = run(source, target, request=Down())
            self.assertFalse(report["complete"])
            self.assertEqual(report["stopped_reason"], "local_reviewer_unavailable")
            self.assertTrue(target.is_file())
            self.assertEqual(report["metrics"]["processed"], 1)

    def test_multi_claim_report_is_not_capped_at_six(self):
        part = "Завод выпустил оборудование для шахт."
        big = row(summary=" ".join([part]*7), source=part)
        fake = FakeOneByOne()
        result = review_cluster_claimwise(big, request=fake)
        self.assertNotIn("invalid_claim_count", result["issues"])
        self.assertEqual(len(result["answer"]["claims"]), 7)


if __name__ == "__main__":
    unittest.main()
