"""Offline safeguards for short diagnostic validation creation."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_news_lora_short_validation import prepare


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize,
                            add_generation_prompt, enable_thinking):
        prefix = messages[0]["content"] + messages[1]["content"] + ":"
        return prefix if add_generation_prompt else prefix + messages[-1]["content"]

    def __call__(self, value, *, add_special_tokens):
        return {"input_ids": [ord(c) for c in value]}


class ShortValidationTests(unittest.TestCase):
    def test_keeps_originals_untouched_and_skips_long(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "original"
            data.mkdir()
            def rec(length):
                return {"task": "fact_tagging", "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "X" * length},
                    {"role": "assistant", "content": "OK"},
                ]}
            text = (json.dumps(rec(200)) + "\n" +
                    (json.dumps(rec(10)) + "\n") * 21)
            (data / "train.jsonl").write_text("TRAIN\n")
            (data / "validation.jsonl").write_text(text)
            (data / "test.jsonl").write_text("TEST\n")
            out = root / "short"
            report = prepare(data, out, FakeTokenizer(), 80)
            self.assertEqual(report["validation_kept"]["fact_tagging"], 21)
            self.assertEqual(report["validation_skipped"]["fact_tagging"], 1)
            self.assertEqual((data / "validation.jsonl").read_text(), text)
            self.assertEqual((data / "test.jsonl").read_text(), "TEST\n")
            self.assertFalse((out / "test.jsonl").exists())
            with self.assertRaises(FileExistsError):
                prepare(data, out, FakeTokenizer(), 80)


if __name__ == "__main__":
    unittest.main()
