"""Offline checks for TRL compatibility: no GPU/model download needed."""
import unittest

from scripts.train_news_lora import trl_api_compat, length_issues, small_validation_allowed


class TRLApiTests(unittest.TestCase):
    def test_trl_v1_api(self):
        class ModernTrainer:
            def __init__(self, *, model, processing_class):
                pass
        class ModernConfig:
            def __init__(self, *, max_length):
                pass
        self.assertEqual(trl_api_compat(ModernTrainer, ModernConfig),
                         ("processing_class", "max_length"))

    def test_legacy_unsloth_api(self):
        class LegacyTrainer:
            def __init__(self, *, model, tokenizer):
                pass
        class LegacyConfig:
            def __init__(self, *, max_seq_length):
                pass
        self.assertEqual(trl_api_compat(LegacyTrainer, LegacyConfig),
                         ("tokenizer", "max_seq_length"))

    def test_detect_truncated_validation(self):
        report = {
            "train": {"fact_tagging": {
                "full_truncated": 0, "possible_template_mismatch": 0,
                "no_reply_tokens": 0}},
            "validation": {"fact_tagging": {
                "full_truncated": 33, "possible_template_mismatch": 0,
                "no_reply_tokens": 29}},
        }
        self.assertEqual(length_issues(report), {
            "validation": {"truncated": 33, "unaligned": 0,
                           "no_reply_tokens": 29},
        })

    def test_tagging_only_small_validation_is_recognized(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            train = Path(folder) / "train.jsonl"
            validation = Path(folder) / "validation.jsonl"
            row = {"task": "fact_tagging", "messages": []}
            train.write_text(json.dumps(row) + "\n", encoding="utf-8")
            validation.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertTrue(small_validation_allowed(train, validation))
            validation.write_text(
                json.dumps({"task": "abstractive_summary"}) + "\n",
                encoding="utf-8",
            )
            self.assertFalse(small_validation_allowed(train, validation))

    def test_missing_compat_fails_early(self):
        class BadTrainer:
            def __init__(self, *, model):
                pass
        class ModernConfig:
            def __init__(self, *, max_length):
                pass
        with self.assertRaisesRegex(RuntimeError, "missing tokenizer"):
            trl_api_compat(BadTrainer, ModernConfig)


if __name__ == "__main__":
    unittest.main()
