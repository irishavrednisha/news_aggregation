"""Offline safety regressions; run: python -m unittest tests.test_moderation_gate"""
import unittest
from types import SimpleNamespace

from publishing.moderation_gate import (
    is_approved_post,
    max_publishing_enabled,
    split_post_snapshot,
)


class ModerationGateTests(unittest.TestCase):
    def test_max_is_disabled_by_default(self):
        self.assertFalse(max_publishing_enabled({}))
        self.assertFalse(max_publishing_enabled({"MAX_PUBLISHING_ENABLED": "false"}))
        self.assertFalse(max_publishing_enabled({"MAX_PUBLISHING_ENABLED": "unexpected"}))

    def test_only_explicit_opt_in_enables_max(self):
        self.assertTrue(max_publishing_enabled({"MAX_PUBLISHING_ENABLED": " TRUE "}))
        self.assertTrue(max_publishing_enabled({"MAX_PUBLISHING_ENABLED": "1"}))

    def test_ready_cluster_or_draft_post_is_never_approval(self):
        for status in ("ready", "draft", "new", "published", "sending", "uncertain", None):
            with self.subTest(status=status):
                post = SimpleNamespace(
                    status=status,
                    cluster_id=5,
                    post_text="Заголовок\n\nТекст и источник",
                )
                self.assertFalse(is_approved_post(post))

    def test_incomplete_approved_post_is_not_publishable(self):
        for value in ("", "Только заголовок", "Заголовок\n\n", "\n\nТекст"):
            with self.subTest(value=value):
                post = SimpleNamespace(status="approved", cluster_id=5, post_text=value)
                self.assertFalse(is_approved_post(post))

    def test_approved_snapshot_is_eligible(self):
        post = SimpleNamespace(
            status="approved",
            cluster_id=5,
            post_text="Заголовок\n\nСводка\n\nИсточник: https://example.org",
        )
        self.assertTrue(is_approved_post(post))
        self.assertEqual(
            split_post_snapshot(post.post_text),
            ("Заголовок", "Сводка\n\nИсточник: https://example.org"),
        )


if __name__ == "__main__":
    unittest.main()
