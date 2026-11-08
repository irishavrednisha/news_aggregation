"""SFT split and safety checks. No model or training dependencies."""
import unittest

from scripts.build_news_sft import build_dataset, example_to_messages

TITLE = "Инженеры изготовили опытный электродвигатель"
BODY = (
    "На машиностроительном заводе инженеры изготовили опытный образец "
    "электродвигателя для станков. "
    "Новая установка прошла проверки на стенде предприятия."
)
EVIDENCE = [
    "На машиностроительном заводе инженеры изготовили опытный образец "
    "электродвигателя для станков."
]
BASE = {
    "story_id": "demo-story-001",
    "origin": "real",
    "split": "train",
    "review_status": "verified",
    "title": TITLE,
    "text": BODY,
    "relevant": True,
    "tags": ["машиностроение"],
    "evidence": EVIDENCE,
    "summary": "На заводе создали опытный электродвигатель для станков.",
}


class SFTDatasetTests(unittest.TestCase):
    def test_positive_creates_two_distinct_tasks(self):
        rows = example_to_messages(BASE)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["task"] for row in rows},
                         {"fact_tagging", "abstractive_summary"})
        self.assertEqual(rows[0]["messages"][0]["role"], "system")
        self.assertEqual(rows[1]["messages"][-1]["role"], "assistant")

    def test_negative_creates_only_classification(self):
        row = {
            **BASE, "relevant": False, "tags": [], "evidence": [],
            "summary": "",
        }
        messages = example_to_messages(row)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["task"], "fact_tagging")

    def test_synthetic_goes_only_train_and_only_if_enabled(self):
        synthetic = {**BASE, "origin": "synthetic"}
        self.assertEqual(len(build_dataset([synthetic])["splits"]["train"]), 0)
        self.assertEqual(
            len(build_dataset([synthetic], allow_synthetic=True)["splits"]["train"]), 2,
        )
        with self.assertRaisesRegex(ValueError, "Synthetic items must never"):
            build_dataset([{**synthetic, "split": "validation"}], allow_synthetic=True)

    def test_same_event_cannot_leak_across_splits(self):
        with self.assertRaisesRegex(ValueError, "Leakage"):
            build_dataset([
                BASE, {**BASE, "split": "test"}
            ], allow_synthetic=True)

    def test_wrong_numbers_are_rejected_from_summary(self):
        with self.assertRaisesRegex(ValueError, "numbers_not_in_evidence"):
            example_to_messages({
                **BASE,
                "summary": "Выпустили 250 новых двигателей.",
            })

    def test_draft_data_cannot_become_training_labels(self):
        draft = {**BASE, "review_status": "draft"}
        output = build_dataset([draft], allow_synthetic=True)
        self.assertEqual(output["splits"]["train"], [])
        self.assertEqual(output["skipped"]["not_verified"], 1)


if __name__ == "__main__":
    unittest.main()
