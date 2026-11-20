"""Offline evidence retrieval and v1 migration tests; no model or database."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from processing.qwen_abstractive_summary import (
    attach_evidence, canonical_materials, source_digest, validate_response,
)
from processing.qwen_evidence import build_claims, source_passages
from scripts.compare_stage4_qwen import run
from scripts.revalidate_stage4_qwen import check_report


def materials():
    return [{
        "id": "m1", "title": "Завод начал испытания новых электродвигателей",
        "url": "https://example.org/1", "source": "A", "is_primary": True,
        "text": (
            "Компания «ТестМаш» начала испытания нового электродвигателя на заводе в Туле. "
            "Стенд рассчитан на нагрузку 20 кВт и будет проверяться до декабря 2026 года. "
            "Серийное производство предприятие пока не запускало. "
            "После испытаний будет принято решение о дальнейшей программе."
        ),
    }, {
        "id": "m2", "title": "Испытания электродвигателя на заводе",
        "url": "https://example.org/2", "source": "B", "is_primary": False,
        "text": (
            "Компания «ТестМаш» сообщила о начале испытаний двигателя в Туле. "
            "Проверки на стенде мощностью 20 кВт продолжатся до конца 2026 года. "
            "Решение о запуске серийного производства ещё не принято."
        ),
    }]


def base_response(summary, **kwargs):
    return {"main_event": "Испытания", "is_news_event": True,
            "summary": summary, "requires_review": False, "uncertainties": [],
            **kwargs}


class EvidenceTests(unittest.TestCase):
    def test_claims_follow_summary_sentences_not_original_sentences(self):
        summary = ("Компания «ТестМаш» испытывает новый электродвигатель в Туле. "
                   "Проверка оборудования рассчитана на нагрузку 20 кВт.")
        docs = canonical_materials(materials())
        claims = build_claims(summary, docs)
        self.assertEqual(len(claims), 2)
        self.assertEqual([x["claim"] for x in claims],
                         ["Компания «ТестМаш» испытывает новый электродвигатель в Туле.",
                          "Проверка оборудования рассчитана на нагрузку 20 кВт."])
        for claim in claims:
            self.assertTrue(claim["source_ids"])
            self.assertEqual(len(claim["source_ids"]), len(claim["evidence_quotes"]))
            for sid, quote in zip(claim["source_ids"], claim["evidence_quotes"]):
                doc = next(x for x in docs if x["id"] == sid)
                self.assertIn(quote, doc["title"] + "\n" + doc["text"])
                self.assertLessEqual(len(quote), 450)

    def test_llm_cannot_substitute_quotes_or_claims(self):
        docs = canonical_materials(materials())
        summary = "Компания «ТестМаш» испытывает электродвигатель в Туле."
        raw = base_response(summary)
        raw["claims"] = [{"claim": "Всё отлично", "source_ids": ["made_up"],
                          "evidence_quotes": ["выдуманная цитата"]}]
        answer = attach_evidence(raw, docs)
        self.assertNotEqual(answer["claims"], raw["claims"])
        self.assertEqual(answer["claims"][0]["claim"], summary)
        self.assertNotIn("выдуманная цитата", str(answer))

    def test_missing_numbers_are_never_invented_as_quotes(self):
        docs = canonical_materials(materials())
        text = ("Компания «ТестМаш» объявила о выпуске 900 двигателей. "
                "Поставки этих изделий уже завершены.")
        rebuilt = attach_evidence(base_response(text), docs)
        result = validate_response(rebuilt, docs)
        self.assertFalse(result["passed"])
        self.assertTrue(any(x.startswith("unsupported_number:900") or
                            "number_not_in_evidence" in x
                            for x in result["issues"]))

    def test_no_unrelated_article_citations(self):
        docs = canonical_materials(materials())
        text = "Строительный холдинг «ДругойЗавод» построил завод по производству цемента."
        claims = build_claims(text, docs)
        check = validate_response(attach_evidence(base_response(text), docs), docs)
        self.assertFalse(check["passed"])
        self.assertTrue(any(x.startswith(("unsupported_quoted_name:",
                                          "claim_1_quoted_name_not_in_evidence",
                                          "claim_1_evidence_structure"))
                            for x in check["issues"]))
        self.assertEqual(claims[0]["claim"], text)

    def test_large_document_quoted_passages_are_exact(self):
        docs = canonical_materials(materials())
        docs[0]["text"] = ("На промышленной площадке идёт ремонт оборудования. " * 80
                           + "На заводе готовят испытания установки мощностью 400 кВт.")
        passages = source_passages(docs)
        self.assertTrue(passages)
        for item in passages:
            self.assertLessEqual(len(item["quote"]), 450)
            full = next(x for x in docs if x["id"] == item["source_id"])
            self.assertTrue(item["quote"] in full["title"] or
                            item["quote"] in full["text"])

    def test_old_report_rechecked_without_rewriting_original_summary(self):
        doc = canonical_materials(materials())[0]
        summary = ("Компания «ТестМаш» приступила к испытаниям электродвигателя в Туле. "
                   "Мощность испытательного стенда составляет 20 кВт. "
                   "Серийное производство пока не запущено.")
        old = {
            "stage": 4, "created_utc": "2026-11-20T00:00:00Z",
            "clusters": [{
                "cluster_id": 900001, "source_materials": [doc],
                "source_sha256": source_digest([doc]),
                "qwen_summary": summary, "main_event": "Испытания",
                "uncertainties": [],
                "attempts": [{"response": base_response(summary, claims=[])}],
            }],
        }
        original = copy.deepcopy(old)
        converted = check_report(old)
        row = converted["clusters"][0]
        self.assertEqual(row["qwen_summary"], summary)
        self.assertEqual(len(row["claims_v2"]), 3)
        self.assertNotIn("claim_count_mismatch", row["validation_v2"]["issues"])
        self.assertEqual(old, original)
        self.assertTrue(converted["editorial_approval_required"])

    def test_runner_stops_after_first_connection_error_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            input_path = base / "source.json"
            output_path = base / "out.json"
            entries = []
            for i in range(3):
                doc = dict(materials()[0])
                doc["id"] = "source:" + str(i)
                entries.append({"cluster_id": 900000 + i,
                                "source_materials": [doc]})
            input_path.write_text(json.dumps({"stage": 4, "clusters": entries},
                                             ensure_ascii=False), encoding="utf-8")
            calls = []
            def offline(*args, **kwargs):
                calls.append(1)
                raise ConnectionRefusedError("Ollama down")
            report = run(input_path, output_path, request=offline,
                         limit=3, measure_gpu=False)
            self.assertEqual(len(calls), 1)
            self.assertEqual(report["stopped_reason"], "local_ollama_unreachable")
            self.assertFalse(report["complete_requested_run"])
            self.assertEqual(len(report["clusters"]), 1)
            self.assertTrue(output_path.is_file())


if __name__ == "__main__":
    unittest.main()
