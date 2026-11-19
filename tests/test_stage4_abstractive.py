"""No Ollama or database needed: mocked real contract and failure modes."""
from __future__ import annotations

import unittest
from processing.abstractive_summary import (
    FACT_FORMAT, SUMMARY_FORMAT, VERDICT_FORMAT, source_digest,
    summarize_event_cluster,
)
from processing.abstractive_validator import validate_summary
from processing.stage4_evidence import shared_boilerplate, clean_working_materials, extract_grounded_facts
from scripts.run_stage4_summaries import assemble, metrics, representative

SOURCE = (
    "АО «Север» сообщило о запуске новой линии обработки металла на предприятии "
    "в Туле. В сообщении указано, что линия оснащена 40 станками и предназначена "
    "для выпуска деталей для промышленного оборудования. Производительность "
    "комплекса составляет 120 изделий в месяц. Работы по установке линии "
    "завершились в ноябре 2026 года."
)
MATERIAL = [{"id": 11, "title": "АО «Север» запустило линию в Туле",
             "text": SOURCE, "url": "https://example.org/one", "is_primary": True,
             "published_at": "2026-11-01"}]
DRAFT = (
    "В Туле предприятие «Север» ввело в эксплуатацию линию металлообработки. "
    "Комплекс с 40 станками будет выпускать детали для промышленного оборудования. "
    "Заявленная производительность оборудования составляет 120 изделий в месяц."
)
FACTS = {"facts": [
    {"kind": "organization", "value": "АО «Север»", "quote": "АО «Север» сообщило",
     "source_id": "11"},
    {"kind": "event", "value": "запуске новой линии обработки металла",
     "quote": "сообщило о запуске новой линии обработки металла", "source_id": "11"},
    {"kind": "number", "value": "40 станками",
     "quote": "линия оснащена 40 станками", "source_id": "11"},
    {"kind": "number", "value": "120 изделий в месяц",
     "quote": "составляет 120 изделий в месяц", "source_id": "11"},
]}
EVIDENCES = [
    "АО «Север» сообщило о запуске новой линии обработки металла на предприятии в Туле",
    "линия оснащена 40 станками и предназначена для выпуска деталей для промышленного оборудования",
    "Производительность комплекса составляет 120 изделий в месяц",
]


def stub(summary=DRAFT):
    calls = []
    def request(messages, schema, *, model):
        calls.append((messages, schema, model))
        if schema is FACT_FORMAT:
            return FACTS
        if schema is SUMMARY_FORMAT:
            return {"summary": summary}
        if schema is VERDICT_FORMAT:
            from processing.abstractive_validator import sentences
            return {"claims": [
                {"claim": part, "verdict": "supported", "evidence": EVIDENCES[i]}
                for i, part in enumerate(sentences(summary))
            ]}
        raise AssertionError("Unexpected request schema")
    return request, calls


class ValidatorTests(unittest.TestCase):
    def test_valid_draft(self):
        result = validate_summary(DRAFT, MATERIAL)
        self.assertTrue(result.passed, result.issues)
        self.assertEqual(result.sentence_count, 3)

    def test_unsupported_numbers_and_unit(self):
        self.assertIn("unsupported_number:50",
                      validate_summary(DRAFT.replace("40 станками", "50 станками"),
                                       MATERIAL).issues)
        issues = validate_summary(
            DRAFT.replace("40 станками", "40 тонн"), MATERIAL).issues
        self.assertIn("number_unit_mismatch:40", issues)

    def test_invented_technical_object(self):
        candidate = DRAFT.replace("линию металлообработки", "ядерный реактор")
        issues = validate_summary(candidate, MATERIAL).issues
        self.assertTrue(any(x.startswith("unsupported_technical_object:реактор") for x in issues))

    def test_copied_headline(self):
        copied = "АО «Север» запустило линию в Туле. " + DRAFT
        self.assertIn("headline_copied_verbatim", validate_summary(copied, MATERIAL).issues)

    def test_made_up_organization(self):
        issues = validate_summary(DRAFT.replace("«Север»", "«Искра»"), MATERIAL).issues
        self.assertTrue(any(x.startswith("unsupported_quoted_name:") for x in issues))

    def test_invented_cause(self):
        issues = validate_summary(DRAFT.replace("Комплекс с", "Благодаря субсидиям комплекс с"),
                                  MATERIAL).issues
        self.assertIn("unsupported_explicit_cause", issues)

    def test_planned_not_completed(self):
        original = [{"title": "Предприятие планирует запуск",
                     "text": "Компания планирует создать линию на заводе. "
                             "Предприятие намерено установить оборудование." * 3}]
        result = validate_summary(
            "Завод запустил линию. Предприятие начало работу. "
            "Оборудование используется в производстве.", original)
        self.assertIn("planned_event_changed_to_completed", result.issues)

    def test_repetition(self):
        repeated = "Предприятие «Север» ввело линию. " * 3
        self.assertIn("repeated_sentences",
                      validate_summary(repeated, MATERIAL).issues)


class PipelineTests(unittest.TestCase):
    def test_happy_path_is_draft_not_approved(self):
        request, calls = stub()
        result = summarize_event_cluster(19, MATERIAL, "TextRank original.",
                                         request=request, max_retries=0)
        self.assertEqual(result.status, "summary_draft", result.validation)
        self.assertEqual(len(calls), 2)  # summary + semantic review; facts are anchored locally
        self.assertGreaterEqual(len(result.facts), 3)
        self.assertTrue(all(f["quote"] in SOURCE or f["quote"] in MATERIAL[0]["title"]
                            for f in result.facts))
        self.assertEqual(result.gemma_summary, DRAFT)
        self.assertEqual(result.source_materials[0]["text"], SOURCE)
        self.assertTrue(result.validation["passed"])

    def test_retries_exhaust_then_manual_review(self):
        request, calls = stub(DRAFT.replace("40 станками", "50 станками"))
        result = summarize_event_cluster(19, MATERIAL, "TextRank original.",
                                         request=request, max_retries=2)
        self.assertEqual(result.status, "summary_review")
        self.assertEqual(len(result.attempts), 3)
        self.assertTrue(all(not x["validation"]["passed"] for x in result.attempts))
        self.assertIn("unsupported_number:50", result.validation["issues"])
        self.assertEqual(len(calls), 3)  # three attempts; no lossy LLM fact extraction

    def test_insufficient_material_does_not_call_model(self):
        request, calls = stub()
        result = summarize_event_cluster(
            1, [{"id": 1, "title": "Вопрос", "text": "мало"}], "",
            request=request)
        self.assertEqual(result.status, "summary_review")
        self.assertEqual(result.validation["issues"], ["insufficient_source_material"])
        self.assertFalse(calls)

    def test_deterministic_facts_are_verifiable_source_spans(self):
        facts, errors = extract_grounded_facts(MATERIAL)
        self.assertFalse(errors, errors)
        self.assertGreaterEqual(len(facts), 3)
        for fact in facts:
            self.assertIn(fact["quote"], SOURCE + MATERIAL[0]["title"])
            self.assertIn(fact["value"], fact["quote"])

    def test_no_invented_event_in_advertorial(self):
        articles = [{"id": 9, "title": "Обзор промышленного оборудования",
                     "text": "В статье обсуждаются возможности модернизации. " * 8}]
        facts, errors = extract_grounded_facts(articles)
        self.assertEqual(facts, [])
        self.assertIn("no_supported_news_event", errors)

    def test_cleaned_working_sources_leave_originals_unchanged(self):
        raw = dict(MATERIAL[0], text=SOURCE + " Услуга чужого сайта.")
        shared = {"" : {"услуга чужого сайта."}}
        cleaned, notes = clean_working_materials([raw], shared)
        # Clip boundaries are conservative (>=450 characters).
        self.assertEqual(raw["text"], SOURCE + " Услуга чужого сайта.")
        self.assertNotEqual(raw, cleaned[0]) if notes else self.assertEqual(raw, cleaned[0])

    def test_fingerprint_stable_and_invalidated_when_source_changes(self):
        before = source_digest(MATERIAL)
        self.assertEqual(before, source_digest([dict(MATERIAL[0])]))
        changed = [dict(MATERIAL[0], text=SOURCE + " Новая информация.")]
        self.assertNotEqual(before, source_digest(changed))

    def test_refuse_cloud_and_oversized_model(self):
        for model in ("gemma3:cloud", "gemma3:12b", "qwen3.5:4b"):
            with self.subTest(model=model), self.assertRaises(ValueError):
                summarize_event_cluster(1, MATERIAL, "", model=model)

    def test_verdict_wrong_or_fake_quote_blocks_draft(self):
        request, calls = stub()
        def fake(messages, schema, *, model):
            if schema is VERDICT_FORMAT:
                return {"claims": [{"claim": part, "verdict": "supported",
                                    "evidence": "выдуманная цитата"}
                                   for part in DRAFT.split(". ")][:3]}
            return request(messages, schema, model=model)
        result = summarize_event_cluster(19, MATERIAL, "", request=fake, max_retries=0)
        self.assertEqual(result.status, "summary_review")
        self.assertTrue(any("semantic_claim_" in x for x in result.validation["issues"]))

    def test_report_never_claims_manual_accuracy(self):
        request, _ = stub()
        row = summarize_event_cluster(19, MATERIAL, "", request=request,
                                      max_retries=0).to_dict()
        report = assemble([row], min_clusters=100, model="gemma3:4b",
                          started_utc="2026-11-19T00:00:00Z", persisted=False)
        self.assertFalse(report["metrics"]["minimum_sample_met"])
        self.assertIsNone(report["metrics"]["manual_factual_accuracy"])
        self.assertEqual(report["representative_before_after"][0]["status"],
                         "summary_draft")
        self.assertEqual(len(report["clusters"]), 1)


if __name__ == "__main__":
    unittest.main()
