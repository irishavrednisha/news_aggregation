"""Offline regression tests: no Ollama, SQL, network or MAX."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from processing import qwen_abstractive_summary as q
from scripts.compare_stage4_qwen import run, match_gemma, evaluate


TITLE = "Тульский завод готовит новую линию выпуска деталей"
TEXT = (
    "Завод «Уралмаш» планирует запустить новую линию деталей в Туле "
    "в декабре 2026 года. Новое оборудование рассчитано на выпуск 20 "
    "деталей в смену. Монтаж линии начнется в октябре, затем проведут "
    "испытания оборудования. Предприятие объясняет, что речь идет о плане "
    "развития производственной площадки, а не о завершенном запуске. "
)
P1 = "Завод «Уралмаш» планирует запуск линии деталей в декабре 2026 года."
P2 = "Оборудование должно выпускать 20 деталей за смену."
P3 = "Испытания линии проведут после её монтажа."
SUMMARY = " ".join([P1, P2, P3])
QUOTES = [
    "Завод «Уралмаш» планирует запустить новую линию деталей в Туле в декабре 2026 года.",
    "Новое оборудование рассчитано на выпуск 20 деталей в смену.",
    "Монтаж линии начнется в октябре, затем проведут испытания оборудования.",
]


def source(sid="source_1", title=TITLE, text=TEXT):
    return {"id": sid, "title": title, "text": text,
            "url": f"https://example.org/{sid}", "source": "Тестовый источник",
            "is_primary": sid == "source_1", "published_at": "2026-11-20"}


def answer(parts=(P1, P2, P3), quotes=QUOTES, ids=None, **overrides):
    ids = ids or ["source_1"] * len(parts)
    obj = {"main_event": "План запуска производственной линии",
           "is_news_event": True, "summary": " ".join(parts),
           "claims": [
               {"claim": sentence, "source_ids": [sid],
                "evidence_quotes": [quote]}
               for sentence, sid, quote in zip(parts, ids, quotes)],
           "uncertainties": [], "requires_review": False}
    obj.update(overrides)
    return obj


class Fake:
    def __init__(self, result=None, exc=None):
        self.calls = 0
        self.result = result if result is not None else answer()
        self.exc = exc

    def __call__(self, messages, schema, *, model, options):
        self.calls += 1
        if self.exc:
            raise self.exc
        assert model == q.MODEL
        assert "claims" not in schema["properties"]
        return self.result, {"eval_count": 100, "eval_duration": 2_000_000_000,
                             "tokens_per_second": 50}


class QwenStage4Tests(unittest.TestCase):
    def test_source_pair_count_includes_missing_gemma_summary(self):
        rows = [
            {"status": "summary_draft", "baseline_match": "matched",
             "gemma_summary": "Завод выпускает детали.", "qwen_summary": "Завод выпускает детали.",
             "validation": {"issues": []}, "attempts": []},
            {"status": "summary_review", "baseline_match": "baseline_summary_missing",
             "gemma_summary": "", "qwen_summary": "Выпущены новые детали.",
             "validation": {"issues": []}, "attempts": []},
            {"status": "summary_review", "baseline_match": "baseline_source_mismatch",
             "gemma_summary": "", "qwen_summary": "Сводка другого события.",
             "validation": {"issues": []}, "attempts": []},
        ]
        report = evaluate(rows)
        self.assertEqual(report["processed"], 3)
        self.assertEqual(report["paired_same_source"], 2)
        self.assertEqual(report["paired_with_gemma_summary"], 1)
        self.assertEqual(report["gemma_generated_in_paired"], 1)

    def test_exact_source_digest_is_stable(self):
        self.assertEqual(q.source_digest([source()]), q.source_digest([source()]))
        self.assertNotEqual(q.source_digest([source()]),
                            q.source_digest([source(text=TEXT+" Новая фраза.")]))
        self.assertEqual(len(q.source_digest([source()])), 64)

    def test_numbers_units_change_is_rejected(self):
        bad = answer(parts=(P1, P2.replace("20", "40"), P3))
        issues = q.validate_response(bad, [q.canonical_materials([source()])[0]])["issues"]
        self.assertIn("unsupported_number:40", issues)

    def test_claim_number_must_occur_in_its_own_quote(self):
        changed = "Оборудование должно выпускать 20 деталей и 50 узлов за смену."
        bad = answer(parts=(P1, changed, P3))
        original = q.canonical_materials([source()])
        issues = q.validate_response(bad, original)["issues"]
        self.assertIn("claim_2_number_not_in_evidence", issues)

    def test_plan_changed_to_result(self):
        sentence = "Завод «Уралмаш» запустил новую линию деталей в декабре 2026 года."
        problems = q._claim_issues(
            {"claim": sentence, "source_ids": ["source_1"],
             "evidence_quotes": [QUOTES[0]]}, sentence,
            {"source_1": q.canonical_materials([source()])[0]}, 1)
        self.assertIn("claim_1_plan_changed_to_result", problems)

    def test_invented_cause_is_review(self):
        sentence = "Благодаря субсидии предприятие выпустит 20 деталей."
        problems = q._claim_issues(
            {"claim": sentence, "source_ids": ["source_1"],
             "evidence_quotes": [QUOTES[1]]}, sentence,
            {"source_1": q.canonical_materials([source()])[0]}, 1)
        self.assertIn("claim_1_unsupported_cause", problems)

    def test_actor_acronym_mismatch(self):
        sentence = "РЖД готовит линию деталей в Туле."
        problems = q._claim_issues(
            {"claim": sentence, "source_ids": ["source_1"],
             "evidence_quotes": [QUOTES[0]]}, sentence,
            {"source_1": q.canonical_materials([source()])[0]}, 1)
        self.assertIn("claim_1_actor_acronym_not_in_evidence", problems)

    def test_false_negation_detected(self):
        sentence = "Завод не планирует запускать линию деталей в Туле."
        problems = q._claim_issues(
            {"claim": sentence, "source_ids": ["source_1"],
             "evidence_quotes": [QUOTES[0]]}, sentence,
            {"source_1": q.canonical_materials([source()])[0]}, 1)
        self.assertIn("claim_1_unsupported_negation", problems)

    def test_growth_decline_mismatch(self):
        sentence = "Выпуск сократился до 20 тонн."
        text = "Выпуск вырос до 20 тонн благодаря модернизации предприятия."
        problems = q._claim_issues(
            {"claim": sentence, "source_ids": ["source_1"],
             "evidence_quotes": [text]}, sentence,
            {"source_1": source(text=text)}, 1)
        self.assertIn("claim_1_growth_changed_to_decline", problems)

    def test_invalid_exact_quote_detected(self):
        broken = answer(quotes=[QUOTES[0].replace("Туле", "Москве"),
                                QUOTES[1], QUOTES[2]])
        problems = q.validate_response(broken, q.canonical_materials([source()]))["issues"]
        self.assertIn("claim_1_quote_not_in_source", problems)

    def test_long_article_preserves_end_and_number_context(self):
        text = TEXT + " ".join(["На площадке обсуждают проект модернизации."] * 280)
        text += " В конце статьи приведен итог: предприятие планирует поставить 70 станков."
        context, coverage = q.prepare_context([source(text=text)], max_chars=3100)
        self.assertIn("70 станков", context[0]["text"])
        self.assertIn("source_1", coverage["truncated_ids"])
        self.assertEqual(len(context), 1)
        self.assertLess(len(context[0]["text"]), len(text))

    def test_short_damaged_source_requires_review_before_model(self):
        fake = Fake()
        out = q.summarize_cluster(1, [source(text="битый текст")], request=fake)
        self.assertEqual(out["status"], "summary_review")
        self.assertIn("insufficient_source_material", out["validation"]["issues"])
        self.assertEqual(fake.calls, 0)

    def test_no_headline_verb_does_not_block_generation(self):
        article = source(title="Крупный промышленный объект в Туле")
        fake = Fake()
        out = q.summarize_cluster(2, [article], request=fake)
        self.assertEqual(fake.calls, 1)
        self.assertEqual(len(out["attempts"]), 1)
        self.assertNotIn("no_supported_news_event", out["validation"]["issues"])

    def test_two_documents_allow_verifiable_evidence(self):
        second = source("source_2", text="Завод «Уралмаш» подтвердил испытания оборудования после монтажа. "
                        + TEXT)
        sources = q.canonical_materials([source(), second])
        claim = {"claim": P3, "source_ids": ["source_1", "source_2"],
                 "evidence_quotes": [QUOTES[2],
                                     "Завод «Уралмаш» подтвердил испытания оборудования после монтажа."]}
        problems = q._claim_issues(claim, P3,
                                   {m["id"]: m for m in sources}, 3)
        self.assertFalse(any("quote_not_in_source" in x for x in problems))

    def test_two_sources_conflicting_number_requires_review(self):
        left = source("source_1", title="Завод сообщил о выпуске новых деталей",
                      text="Завод сообщил о выпуске 20 деталей в месяц. "*5)
        right = source("source_2", title="Завод сообщил о выпуске новых деталей",
                       text="Завод сообщил о выпуске 40 деталей в месяц. "*5)
        flagged = q.cross_source_issues(q.canonical_materials([left, right]))
        self.assertIn("potential_cross_source_numeric_conflict", flagged)

    def test_ollama_unavailable_fails_closed(self):
        fake = Fake(exc=ConnectionError("Local Ollama is offline"))
        out = q.summarize_cluster(3, [source()], request=fake)
        self.assertEqual(out["status"], "summary_review")
        self.assertTrue(out["validation"]["issues"][0].startswith("generation_error"))
        self.assertEqual(fake.calls, 1)

    def test_malformed_json_from_client_fails_closed(self):
        fake = Fake(exc=json.JSONDecodeError("Invalid", "{", 1))
        out = q.summarize_cluster(3, [source()], request=fake)
        self.assertEqual(out["status"], "summary_review")
        self.assertIn("JSONDecodeError", out["validation"]["issues"][0])

    def test_invalid_response_schema(self):
        invalid = {"summary": SUMMARY}
        out = q.summarize_cluster(4, [source()], request=Fake(invalid), max_retries=0)
        self.assertEqual(out["status"], "summary_review")
        self.assertIn("schema_fields_mismatch", out["validation"]["issues"])

    def test_model_false_review_flag_is_not_publication_permission(self):
        out = q.summarize_cluster(5, [source()], request=Fake())
        self.assertTrue(out["requires_human_approval"])
        self.assertFalse(out["auto_publication_allowed"])
        self.assertNotEqual(out["status"], "published")

    def test_model_no_event_does_not_become_automatic_rejection(self):
        no_event = answer(is_news_event=False, summary="", claims=[],
                          uncertainties=["Недостаточно сведений"], requires_review=True)
        out = q.summarize_cluster(6, [source()], request=Fake(no_event))
        self.assertEqual(out["status"], "summary_review")
        self.assertIn("model_classified_no_news_event", out["validation"]["issues"])

    def test_duplicate_ids_are_rejected(self):
        out = q.summarize_cluster(7, [source(), source()], request=Fake())
        self.assertIn("duplicate_source_ids", out["validation"]["issues"])

    def test_nonlocal_ollama_endpoint_is_forbidden(self):
        for url in ("https://example.org:11434", "http://api.ollama.ai",
                    "http://user:password@localhost:11434"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                q.LocalOllama(url)
        q.LocalOllama("http://127.0.0.1:11434")

    def test_resume_uses_prior_snapshot_and_does_not_touch_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_json, output = root/"old.json", root/"new.json"
            obj = {"stage": 4, "clusters": [{
                "cluster_id": 101, "source_materials": [source()],
                "gemma_summary": "Готовится новая линия.", "status": "summary_review",
            }]}
            raw = json.dumps(obj, ensure_ascii=False)
            input_json.write_text(raw, encoding="utf-8")
            fake = Fake()
            first = run(input_json, output, request=fake, measure_gpu=False, limit=1)
            self.assertEqual(fake.calls, 1)
            review = root/"new_human_review.csv"
            self.assertTrue(review.exists())
            original_labels = review.read_bytes() + b"\n# preserved"
            review.write_bytes(original_labels)
            second = run(input_json, output, request=fake, measure_gpu=False,
                         limit=1, resume=True)
            self.assertEqual(fake.calls, 1)
            self.assertEqual(review.read_bytes(), original_labels)
            self.assertEqual(input_json.read_text(encoding="utf-8"), raw)
            self.assertEqual(first["clusters"][0]["source_sha256"],
                             second["clusters"][0]["source_sha256"])

    def test_comparison_rejects_different_baseline_source(self):
        current = {"cluster_id": 9, "source_materials": [source()]}
        baseline = {"9": {"cluster_id": 9, "source_materials": [
            source(text=TEXT+" иное")], "gemma_summary": "Текст"}}
        self.assertEqual(match_gemma(current, baseline)[1],
                         "baseline_source_mismatch")


    def test_claim_local_ordinary_organization_mismatch(self):
        sentence = "Компания Росатом планирует новую линию в Туле."
        problems = q._claim_issues(
            {"claim": sentence, "source_ids": ["source_1"],
             "evidence_quotes": [QUOTES[0]]}, sentence,
            {"source_1": q.canonical_materials([source()])[0]}, 1)
        self.assertIn("claim_1_proper_name_not_in_evidence", problems)

    def test_unit_mismatch_in_selected_claim_quote(self):
        evidence = "Завод поставил 20 кг сырья на новый производственный участок."
        sentence = "Завод поставил 20 тонн сырья на участок."
        problems = q._claim_issues(
            {"claim": sentence, "source_ids": ["source_1"],
             "evidence_quotes": [evidence]}, sentence,
            {"source_1": source(text=evidence)}, 1)
        self.assertIn("claim_1_unit_not_in_evidence", problems)

    def test_experimental_test_is_not_serial_production(self):
        evidence = "Завод завершил испытания опытного станка на своей площадке."
        sentence = "Завод начал серийное производство нового станка."
        problems = q._claim_issues(
            {"claim": sentence, "source_ids": ["source_1"],
             "evidence_quotes": [evidence]}, sentence,
            {"source_1": source(text=evidence)}, 1)
        self.assertIn("claim_1_unsupported_serial_production", problems)

    def test_explicit_model_review_is_not_suppressed(self):
        out = q.summarize_cluster(11, [source()],
                                  request=Fake(answer(requires_review=True)))
        self.assertIn("model_requested_review", out["validation"]["issues"])
        self.assertEqual(out["status"], "summary_review")


    def test_resume_retries_failed_ollama_without_replacing_completed_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inp, output = root/"articles.json", root/"comparison.json"
            inp.write_text(json.dumps({"stage": 4, "clusters": [{
                "cluster_id": 55, "source_materials": [source()],
                "gemma_summary": "Тестовая Gemma сводка"}]},
                ensure_ascii=False), encoding="utf-8")
            failed = Fake(exc=ConnectionError("Ollama offline"))
            run(inp, output, request=failed, limit=1, measure_gpu=False)
            self.assertEqual(failed.calls, 1)
            self.assertFalse(json.loads(output.read_text(encoding="utf-8"))
                             ["clusters"][0]["qwen_summary"])
            recovered = Fake()
            report = run(inp, output, request=recovered, limit=1,
                         measure_gpu=False, resume=True)
            self.assertEqual(recovered.calls, 1)
            self.assertTrue(report["clusters"][0]["qwen_summary"])


if __name__ == "__main__":
    unittest.main()
