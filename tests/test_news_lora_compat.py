"""Offline checks for TRL compatibility: no GPU/model download needed."""
import unittest

from scripts.train_news_lora import trl_api_compat


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
