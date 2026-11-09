"""Safety and determinism checks for diagnostic mixed processing batches."""
import unittest

from scripts.select_processing_batch import create_batch


def make_row(i, *, source=None, tls=False):
    return {
        "source": source or ("A" if i % 2 else "B"),
        "title": f"Новость промышленности номер {i}",
        "url": f"https://example.org/news/{i}",
        "published_at": "2026-11-09",
        "text": "Здесь полный текст новости о производстве и оборудовании." * 7,
        "unverified_tls": tls,
    }


class MixedBatchTests(unittest.TestCase):
    def test_balances_without_loss_or_leak_and_keeps_tls(self):
        raw = [make_row(i, tls=i == 3) for i in range(16)]
        audit = {"as_of": "2026-11-09", "decisions": [
            {"url": row["url"], "accepted_by_primary_filter": i < 8,
             "filter_reason": "accepted" if i < 8 else "offtopic"}
            for i, row in enumerate(raw)
        ]}
        payload = create_batch(
            {"articles": raw}, audit, limit=10, accepted=6,
        )
        selected = payload["articles"]
        self.assertEqual(len(selected), 10)
        self.assertEqual(
            sum(r["primary_filter_accepted"] for r in selected), 6
        )
        self.assertEqual(
            sum(not r["primary_filter_accepted"] for r in selected), 4
        )
        self.assertEqual(
            len({r["url"] for r in selected}), len(selected)
        )
        self.assertEqual(payload["selection"]["selected"], 10)
        self.assertEqual(payload["gold_labels_provided"], False)
        self.assertFalse(payload["publishing_performed"])
        self.assertEqual(
            create_batch({"articles": raw}, audit, limit=10, accepted=6),
            payload,
        )
        full = create_batch({"articles": raw}, audit, limit=16, accepted=8)
        self.assertEqual(full["selection"]["unverified_tls"], 1)

    def test_cross_source_same_title_only_sampled_once(self):
        a, b = make_row(0), make_row(1)
        b["title"] = a["title"]
        payload = create_batch(
            {"articles": [a, b]},
            {"decisions": [
                {"url": a["url"], "accepted_by_primary_filter": True,
                 "filter_reason": "accepted"},
                {"url": b["url"], "accepted_by_primary_filter": False,
                 "filter_reason": "offtopic"},
            ]},
            limit=2, accepted=1,
        )
        self.assertEqual(len(payload["articles"]), 1)

    def test_requires_matching_url_sets_and_valid_filter_flags(self):
        a, b = make_row(0), make_row(1)
        for audit in (
            {"decisions": []},
            {"decisions": [{"url": a["url"], "accepted_by_primary_filter": "true"}]},
            {"decisions": [
                {"url": a["url"], "accepted_by_primary_filter": True},
                {"url": a["url"], "accepted_by_primary_filter": False},
            ]},
        ):
            with self.subTest(audit=audit):
                with self.assertRaises(ValueError):
                    create_batch({"articles": [a, b]}, audit)

    def test_no_implicit_training_and_early_mixed_prefix(self):
        raw = [make_row(i) for i in range(6)]
        audit = {"decisions": [
            {"url": r["url"], "accepted_by_primary_filter": i < 3,
             "filter_reason": "accepted" if i < 3 else "stale_article"}
            for i, r in enumerate(raw)
        ]}
        p = create_batch({"articles": raw}, audit, limit=6, accepted=3)
        self.assertEqual(
            [r["primary_filter_accepted"] for r in p["articles"]],
            [True, False, True, False, True, False],
        )
        self.assertFalse(p["training_performed"])


if __name__ == "__main__":
    unittest.main()
