"""LLM prototype contract tests without Ollama/model download/DB/MAX."""
import unittest

from processing.local_news_llm import (
    TAGS, FACT_FORMAT, SUMMARY_FORMAT, FACT_SYSTEM, SUMMARY_SYSTEM,
    extract_messages, summary_messages,
    source_payload, validate_evidence,
    validate_fact_decision, validate_summary, analyze_news,
)

TITLE = "Завод выпустил новый силовой трансформатор"
BODY = (
    "Электротехническое предприятие изготовило новую серию силовых "
    "трансформаторов для распределительных сетей. "
    "Компания сообщила, что на испытательном стенде оборудование "
    "успешно прошло проверку изоляции. "
    "Производственная мощность линии составляет 40 изделий в месяц."
)
EVIDENCE = [
    "Электротехническое предприятие изготовило новую серию силовых "
    "трансформаторов для распределительных сетей.",
    "Компания сообщила, что на испытательном стенде оборудование "
    "успешно прошло проверку изоляции.",
]


class LocalLLMTests(unittest.TestCase):
    def test_categories_are_fixed_and_local_api_does_not_require_model_at_import(self):
        self.assertEqual(len(TAGS), 18)
        self.assertEqual(len(set(TAGS)), 18)
        self.assertEqual(FACT_FORMAT["properties"]["tags"]["items"]["enum"], list(TAGS))

    def test_evidence_must_appear_in_source(self):
        source = source_payload(TITLE, BODY)
        self.assertEqual(validate_evidence(source, EVIDENCE), EVIDENCE)
        with self.assertRaisesRegex(ValueError, "исказила цитату"):
            validate_evidence(source, ["Компания построила станцию на Марсе."])

    def test_invalid_tag_refused(self):
        with self.assertRaisesRegex(ValueError, "несуществующий"):
            validate_fact_decision({
                "relevant": True,
                "tags": ["космическая промышленность"],
                "evidence": EVIDENCE,
                "uncertain": False,
            }, source_payload(TITLE, BODY))

    def test_numerical_hallucinations_raise_review_warning(self):
        self.assertEqual(validate_summary(
            "Выпущены 150 новых трансформаторов.", EVIDENCE
        ), ["numbers_not_in_evidence:150"])
        self.assertEqual(validate_summary(
            "Предприятие изготовило серию трансформаторов.", EVIDENCE
        ), [])

    def test_abstractive_two_stage_processing(self):
        responses = [
            {"relevant": True,
             "tags": ["электротехническое направление"],
             "evidence": EVIDENCE,
             "uncertain": False},
            {"summary": ("Предприятие выпустило линейку трансформаторов "
                         "для распределительных сетей. "
                         "Оборудование выдержало проверку изоляции.")},
        ]
        calls = []

        def fake_request(messages, schema, **kwargs):
            calls.append((messages, schema, kwargs))
            return responses[len(calls)-1]

        decision = analyze_news(TITLE, BODY, request=fake_request)
        self.assertTrue(decision.relevant)
        self.assertEqual(decision.tags, ["электротехническое направление"])
        self.assertIn("Оборудование выдержало", decision.summary)
        self.assertEqual(decision.evidence, EVIDENCE)
        self.assertTrue(decision.review_required)
        self.assertEqual(decision.warnings, [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][1], SUMMARY_FORMAT)

    def test_focus_summary_orders_tags_and_verifies_with_original(self):
        edited = "Завод начал выпуск новой серии трансформаторов для сетей."
        messages = extract_messages(TITLE, BODY, focus_summary=edited)
        self.assertIn(edited, messages[1]["content"])
        self.assertIn(BODY, messages[1]["content"])
        self.assertIn("исходной статьи", messages[0]["content"].lower())
        self.assertLess(
            messages[1]["content"].index(edited),
            messages[1]["content"].index(BODY),
        )
        calls = []
        def fake(msg, schema, **kw):
            calls.append((msg, schema))
            return {
                "relevant": True,
                "tags": ["электротехническое направление"],
                "evidence": [EVIDENCE[0]],
                "uncertain": False,
            }
        result = analyze_news(
            TITLE, BODY, request=fake, tags_only=True,
            focus_summary=edited,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result.tags, ["электротехническое направление"])
        self.assertTrue(result.review_required)
        self.assertEqual(result.evidence, [EVIDENCE[0]])
        self.assertEqual(result.summary, "")

    def test_focus_summary_cannot_supply_source_evidence(self):
        invented = "Завод якобы построил 900 новых реакторов на Марсе."
        def fake(*args, **kwargs):
            return {
                "relevant": True,
                "tags": ["электротехническое направление"],
                "evidence": [invented],
                "uncertain": False,
            }
        with self.assertRaisesRegex(ValueError, "исказила цитату"):
            analyze_news(
                TITLE, BODY, request=fake, tags_only=True,
                focus_summary=invented,
            )

    def test_focus_summary_negative_still_requires_human_review(self):
        def fake(*args, **kwargs):
            return {
                "relevant": False, "tags": [], "evidence": [],
                "uncertain": False,
            }
        result = analyze_news(
            TITLE, BODY, request=fake, tags_only=True,
            focus_summary="Корпоративный праздник на заводе.",
        )
        self.assertFalse(result.relevant)
        self.assertEqual(result.tags, [])
        self.assertTrue(result.review_required)

    def test_focus_summary_empty_is_invalid(self):
        with self.assertRaises(ValueError):
            extract_messages(TITLE, BODY, focus_summary=" ")

    def test_tags_only_runs_one_fact_request_without_summary(self):
        calls = []
        def fake(messages, schema, **kwargs):
            calls.append((messages, schema))
            return {
                "relevant": True,
                "tags": ["электротехническое направление"],
                "evidence": [EVIDENCE[0]],
                "uncertain": False,
            }
        result = analyze_news(TITLE, BODY, request=fake, tags_only=True)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], FACT_FORMAT)
        self.assertEqual(result.tags, ["электротехническое направление"])
        self.assertEqual(result.summary, "")
        self.assertEqual(result.evidence, [EVIDENCE[0]])
        self.assertTrue(result.review_required)
        self.assertFalse(result.warnings)

    def test_tags_only_preserves_uncertainty_flag(self):
        def fake(messages, schema, **kwargs):
            return {
                "relevant": True,
                "tags": ["электротехническое направление"],
                "evidence": EVIDENCE,
                "uncertain": True,
            }
        result = analyze_news(TITLE, BODY, request=fake, tags_only=True)
        self.assertTrue(result.review_required)
        self.assertIn("model_uncertain", result.warnings)

    def test_tags_only_irrelevant_news_no_tags(self):
        calls = []
        def fake(messages, schema, **kwargs):
            calls.append(1)
            return {
                "relevant": False, "tags": [],
                "evidence": [], "uncertain": False,
            }
        result = analyze_news(TITLE, BODY, request=fake, tags_only=True)
        self.assertFalse(result.relevant)
        self.assertFalse(result.tags)
        self.assertEqual(len(calls), 1)

    def test_flexible_summary_length_limit_is_550_characters(self):
        # 250–400 is a soft target; short summaries aren't padded.
        self.assertEqual(validate_summary("Завод обновил оборудование.", EVIDENCE), [])
        self.assertEqual(validate_summary("а" * 550, EVIDENCE), [])
        self.assertIn(
            "summary_empty_or_too_long",
            validate_summary("а" * 551, EVIDENCE),
        )

    def test_overlong_summary_gets_single_rewrite(self):
        replies = [
            {"relevant": True, "tags": ["электротехническое направление"],
             "evidence": EVIDENCE, "uncertain": False},
            {"summary": "Завод изготовил оборудование. " * 30},
            {"summary": "Компания выпустила трансформаторы для сетей. "
                        "Изоляция прошла испытания."},
        ]
        calls = []

        def fake_request(messages, schema, **kwargs):
            calls.append(messages)
            return replies[len(calls) - 1]

        result = analyze_news(TITLE, BODY, request=fake_request)
        self.assertEqual(len(calls), 3)
        self.assertLessEqual(len(result.summary), 550)
        self.assertNotIn("summary_empty_or_too_long", result.warnings)

    def test_still_overlong_after_rewrite_is_discarded(self):
        replies = [
            {"relevant": True, "tags": ["электротехническое направление"],
             "evidence": EVIDENCE, "uncertain": False},
            {"summary": "Завод изготовил оборудование. " * 30},
            {"summary": "Предприятие испытало новые агрегаты. " * 30},
        ]
        calls = []

        def fake_request(messages, schema, **kwargs):
            calls.append(messages)
            return replies[len(calls) - 1]

        result = analyze_news(TITLE, BODY, request=fake_request)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result.summary, "")
        self.assertIn("summary_discarded_too_long", result.warnings)
        self.assertTrue(result.review_required)

    def test_prompts_prefer_diverse_source_facts_without_padding(self):
        fact_prompt = extract_messages(TITLE, BODY)[1]["content"]
        summary_prompt = summary_messages(EVIDENCE)[1]["content"]
        self.assertIn("3–5", fact_prompt)
        self.assertIn("технические детали", summary_prompt)
        self.assertIn("«примерно»", FACT_SYSTEM)
        self.assertIn("«примерно»", SUMMARY_SYSTEM)
        self.assertIn("не растягивай", summary_prompt)
        self.assertIn("организации", FACT_SYSTEM)

    def test_exact_source_copy_is_flagged(self):
        quote = (
            "Предприятие освоило новую технологическую линию производства "
            "промышленных насосов и испытало все агрегаты на стенде."
        )
        self.assertIn(
            "summary_near_verbatim",
            validate_summary(quote, [quote]),
        )
        self.assertNotIn(
            "summary_near_verbatim",
            validate_summary(
                "Компания запустила выпуск насосов и проверила новое "
                "оборудование в ходе заводских испытаний.",
                [quote],
            ),
        )

    def test_near_verbatim_summary_is_rewritten_once(self):
        exact = EVIDENCE[0]
        replies = [
            {"relevant": True, "tags": ["электротехническое направление"],
             "evidence": EVIDENCE, "uncertain": False},
            {"summary": exact},
            {"summary": "Компания выпустила трансформаторы для электросетей. "
                        "Их изоляция прошла заводские испытания."},
        ]
        calls = []

        def fake_request(messages, schema, **kwargs):
            calls.append(messages)
            return replies[len(calls) - 1]

        result = analyze_news(TITLE, BODY, request=fake_request)
        self.assertEqual(len(calls), 3)
        self.assertFalse(result.warnings)
        self.assertIn("Компания выпустила", result.summary)
        self.assertTrue(result.review_required)

    def test_failed_rewrite_remains_warning_for_review(self):
        replies = [
            {"relevant": True, "tags": ["электротехническое направление"],
             "evidence": EVIDENCE, "uncertain": False},
            {"summary": EVIDENCE[0]},
            {"summary": EVIDENCE[0]},
        ]
        calls = []

        def fake_request(messages, schema, **kwargs):
            calls.append(messages)
            return replies[len(calls) - 1]

        result = analyze_news(TITLE, BODY, request=fake_request)
        self.assertEqual(len(calls), 3)
        self.assertIn("summary_near_verbatim", result.warnings)
        self.assertTrue(result.review_required)

    def test_unrelated_news_does_not_generate_summary(self):
        calls = []
        def fake_request(messages, schema, **kwargs):
            calls.append(messages)
            return {"relevant": False, "tags": [],
                    "evidence": [], "uncertain": False}
        decision = analyze_news("ОСК провела фестиваль", BODY, request=fake_request)
        self.assertFalse(decision.relevant)
        self.assertEqual(decision.tags, [])
        self.assertEqual(decision.summary, "")
        self.assertEqual(len(calls), 1)

    def test_malformed_quote_gets_one_exact_copy_retry(self):
        copied = "Компания создала новую несуществующую технологию производства."
        answers = [
            {"relevant": True, "tags": ["машиностроение"],
             "evidence": [copied], "uncertain": False},
            {"relevant": True, "tags": ["электротехническое направление"],
             "evidence": EVIDENCE, "uncertain": False},
            {"summary": "Компания выпустила трансформаторы для электросетей. "
                        "Оборудование прошло испытания."},
        ]
        calls = []
        def fake_request(messages, schema, **kwargs):
            calls.append(messages)
            return answers[len(calls) - 1]
        result = analyze_news(TITLE, BODY, request=fake_request)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result.tags, ["электротехническое направление"])
        self.assertEqual(result.evidence, EVIDENCE)
        self.assertEqual(result.warnings, [])
        self.assertTrue(result.review_required)

    def test_bad_quote_retry_does_not_silently_approve(self):
        wrong = "Компания создала новую несуществующую технологию производства."
        replies = [
            {"relevant": True, "tags": ["машиностроение"],
             "evidence": [wrong], "uncertain": False},
            {"relevant": True, "tags": ["машиностроение"],
             "evidence": [wrong], "uncertain": False},
        ]
        calls = []
        def fake_request(messages, schema, **kwargs):
            calls.append(messages)
            return replies[len(calls) - 1]
        with self.assertRaisesRegex(ValueError, "исказила цитату"):
            analyze_news(TITLE, BODY, request=fake_request)
        self.assertEqual(len(calls), 2)

    def test_quote_retry_cannot_silently_downgrade_to_unrelated(self):
        replies = [
            {"relevant": True, "tags": ["машиностроение"],
             "evidence": ["Несуществующая цитата о предприятии и станках."],
             "uncertain": False},
            {"relevant": False, "tags": [], "evidence": [], "uncertain": False},
        ]
        calls = []
        def fake_request(messages, schema, **kwargs):
            calls.append(messages)
            return replies[len(calls) - 1]
        with self.assertRaisesRegex(ValueError, "отказалась от релевантной"):
            analyze_news(TITLE, BODY, request=fake_request)
        self.assertEqual(len(calls), 2)

    def test_fail_closed_if_model_invents_evidence(self):
        def fake_request(messages, schema, **kwargs):
            return {"relevant": True, "tags": ["машиностроение"],
                    "evidence": ["Вдуманный завод открылся вчера в Париже."],
                    "uncertain": False}
        with self.assertRaisesRegex(ValueError, "исказила цитату"):
            analyze_news(TITLE, BODY, request=fake_request)


if __name__ == "__main__":
    unittest.main()
