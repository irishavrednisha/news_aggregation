"""CPU-only regression tests for LoRA diagnostics."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_news_lora import surviving_reply_tokens, rendered_token_ids, audit
from scripts.evaluate_news_lora import select_rows, summarize, f1, text_only_tokenizer, decode_json_response, grade_response


class DiagnosticsTests(unittest.TestCase):
    def test_text_only_tokenizer_skips_multimodal_image_decode(self):
        class FakePlainTokenizer:
            eos_token_id = 12
            def __call__(self, *, text, return_tensors, add_special_tokens):
                self.last_input = text
                assert return_tensors == "pt"
                assert add_special_tokens is False
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
            def decode(self, ids, *, skip_special_tokens):
                return "decoded"

        class FakeVisionProcessor:
            def __init__(self):
                self.tokenizer = FakePlainTokenizer()
            def __call__(self, *args, **kwargs):
                raise RuntimeError("Unsupported image file")

        processor = FakeVisionProcessor()
        text_tokenizer = text_only_tokenizer(processor)
        self.assertIs(text_tokenizer, processor.tokenizer)
        self.assertEqual(
            text_tokenizer(text="TEST NEWS", return_tensors="pt",
                           add_special_tokens=False)["input_ids"],
            [1, 2, 3],
        )
        self.assertEqual(text_tokenizer.last_input, "TEST NEWS")
        self.assertIs(text_only_tokenizer(text_tokenizer), text_tokenizer)

    def test_complete_fenced_json_is_parsed_but_missing_fields_fail(self):
        raw = '```json\n{"relevant":false,"tags":[],"evidence":[]}\n```'
        parsed, wrapped = decode_json_response(raw)
        self.assertTrue(wrapped)
        self.assertFalse(parsed["relevant"])
        result = grade_response(
            "fact_tagging",
            {"relevant": False, "tags": [], "evidence": []},
            raw,
            "Заголовок: Демонстрация\nСтатья: Тестовая новость.",
        )
        self.assertTrue(result["valid_json"])
        self.assertTrue(result["markdown_wrapped"])
        self.assertFalse(result["valid_contract"])
        self.assertIn("неверные типы", result["error"])

    def test_truncated_json_is_not_repaired(self):
        raw = '{"headline":"Длинный ответ","facts":[{"quote":"оборвал'
        result = grade_response("fact_tagging", {}, raw, "article")
        self.assertFalse(result["valid_json"])
        self.assertFalse(result["valid_contract"])
        with self.assertRaises(ValueError):
            decode_json_response('```json\n{"relevant":false}')

    def test_valid_contract_without_fence(self):
        raw = ('{"relevant":false,"tags":[],"evidence":[],'
               '"uncertain":false}')
        result = grade_response("fact_tagging", {"relevant": False, "tags": []},
                                raw, "Статья: пример")
        self.assertTrue(result["valid_json"])
        self.assertFalse(result["markdown_wrapped"])
        self.assertTrue(result["valid_contract"])
        self.assertTrue(result["relevant_correct"])

    def test_token_limit_diagnostic_in_summary(self):
        records = [
            {"task": "fact_tagging", "generation_hit_token_limit": True,
             "grade": {"valid_json": False, "valid_contract": False}},
            {"task": "fact_tagging", "generation_hit_token_limit": False,
             "grade": {"valid_json": True, "valid_contract": False,
                       "markdown_wrapped": True}},
        ]
        info = summarize(records)
        self.assertEqual(info["generation_hit_token_limit"], 1)
        self.assertEqual(info["markdown_wrapped_json"], 1)
        self.assertEqual(info["valid_json"], 1)
        self.assertEqual(info["valid_contract"], 0)

    def test_reply_survival(self):
        self.assertEqual(surviving_reply_tokens(1000, 1060, 1024), 24)
        self.assertEqual(surviving_reply_tokens(1100, 1160, 1024), 0)
        self.assertEqual(surviving_reply_tokens(500, 600, 1024), 100)

    def test_qwen_mapping_would_have_two_keys_not_two_tokens(self):
        class FakeQwenTokenizer:
            def apply_chat_template(self, messages, *, tokenize,
                                    add_generation_prompt, enable_thinking):
                self_tokenized = self
                self_tokenized.last_tokenize = tokenize
                prefix = "SYSTEM: " + messages[0]["content"]
                if len(messages) > 1:
                    prefix += " USER: " + messages[1]["content"]
                prefix += " ASSISTANT: "
                if add_generation_prompt:
                    return prefix
                return prefix + messages[-1]["content"] + " END"
            def __call__(self, text, *, add_special_tokens):
                # A real tokenization can return a BatchEncoding mapping.
                return {"input_ids": [ord(c) for c in text],
                        "attention_mask": [1] * len(text)}

        tok = FakeQwenTokenizer()
        messages = [
            {"role": "system", "content": "You are an industrial editor."},
            {"role": "user", "content": "Factory launched a new engine."},
            {"role": "assistant", "content": '{"relevant": true}'},
        ]
        ids = rendered_token_ids(tok, messages, add_generation_prompt=False)
        self.assertGreater(len(ids), 50)
        self.assertFalse(tok.last_tokenize)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "train.jsonl").write_text(
                json.dumps({"messages": messages, "task": "fact_tagging"})
                + "\n", encoding="utf-8",
            )
            result = audit(folder, tok, max_length=90)
        data = result["train"]["fact_tagging"]
        self.assertEqual(data["records"], 1)
        self.assertGreater(data["full_tokens_total"], 50)
        self.assertGreater(data["reply_tokens_total"], 0)
        self.assertEqual(data["possible_template_mismatch"], 0)
        self.assertEqual(data["aligned_records"], 1)

    def test_unaligned_chat_template_does_not_claim_missing_replies(self):
        class MismatchedTokenizer:
            def apply_chat_template(self, messages, *, tokenize,
                                    add_generation_prompt, enable_thinking):
                return "PREFIX-B" if add_generation_prompt else "PREFIX-A_RESPONSE"
            def __call__(self, text, *, add_special_tokens):
                return {"input_ids": [ord(c) for c in text]}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            messages = [{"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"}]
            (folder / "validation.jsonl").write_text(
                json.dumps({"messages": messages, "task": "fact_tagging"})
                + "\n", encoding="utf-8",
            )
            data = audit(folder, MismatchedTokenizer(), 1024)["validation"]["fact_tagging"]
        self.assertEqual(data["possible_template_mismatch"], 1)
        self.assertEqual(data["aligned_records"], 0)
        self.assertEqual(data["no_reply_tokens"], 0)

    def test_stratified_limit_reproducible(self):
        rows = [
            {"task": "fact_tagging", "story_id": "a"},
            {"task": "fact_tagging", "story_id": "b"},
            {"task": "abstractive_summary", "story_id": "a"},
            {"task": "abstractive_summary", "story_id": "c"},
            {"task": "fact_tagging", "story_id": "c"},
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "validation.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            selected = select_rows(path, 1)
        self.assertEqual(
            [(r["task"], r["story_id"]) for r in selected],
            [("fact_tagging", "a"), ("abstractive_summary", "a")],
        )

    def test_task_filter_keeps_only_requested_task(self):
        rows = [
            {"task": "fact_tagging", "story_id": "a"},
            {"task": "abstractive_summary", "story_id": "a"},
            {"task": "fact_tagging", "story_id": "b"},
            {"task": "abstractive_summary", "story_id": "b"},
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "validation.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            selected = select_rows(path, 0, "fact_tagging")
            self.assertEqual([r["story_id"] for r in selected], ["a", "b"])
            self.assertEqual([r["task"] for r in select_rows(path, 1, "abstractive_summary")],
                             ["abstractive_summary"])
            self.assertEqual(len(select_rows(path, 0)), 4)

    def test_summary_reports_failures_and_skips(self):
        rows = [
            {"task": "fact_tagging", "grade": {
                "valid_json": True, "valid_contract": True,
                "relevant_correct": True, "tags_exact": True,
            }},
            {"task": "abstractive_summary", "skip_reason": "too_long"},
        ]
        summary = summarize(rows)
        self.assertEqual(summary["records"], 2)
        self.assertEqual(summary["skipped_long_prompt"], 1)
        self.assertEqual(summary["relevance_correct"], 1)
        self.assertEqual(summary["summary_valid_contract"], 0)
        self.assertEqual(f1(0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
