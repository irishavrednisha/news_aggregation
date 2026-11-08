"""Standard library tests: tagging-only SFT preparation safety checks."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_news_lora_tagging_only import prepare, label_type


def item(story: str, relevant: bool, tags: list[str],
         task: str = "fact_tagging") -> dict:
    return {
        "task": task, "story_id": story, "origin": "synthetic",
        "messages": [
            {"role": "system", "content": "editor"},
            {"role": "user", "content": "source"},
            {"role": "assistant", "content": json.dumps(
                {"relevant": relevant, "tags": tags, "evidence": [],
                 "uncertain": False})},
        ],
    }


def write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class TaggingOnlyDataTests(unittest.TestCase):
    def test_label_classification(self):
        self.assertEqual(label_type(item("a", False, [])), "negative")
        self.assertEqual(label_type(item("a", True, ["one"])), "single_tag")
        self.assertEqual(label_type(item("a", True, ["one", "two"])), "multi_tag")
        with self.assertRaises(ValueError):
            label_type(item("a", True, []))

    def test_balances_only_train_and_keeps_original_holdouts(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data = root / "source"
            data.mkdir()
            train = [item("n" + str(i), False, []) for i in range(50)]
            train += [item("s" + str(i), True, ["one"]) for i in range(100)]
            train += [item("m" + str(i), True, ["one", "two"]) for i in range(15)]
            train += [item("s" + str(i), True, ["one"], task="abstractive_summary")
                      for i in range(100)]
            val = [item("v" + str(i), False, []) for i in range(12)]
            val += [item("v12", True, ["one"], task="abstractive_summary")]
            write(data / "train.jsonl", train)
            write(data / "validation.jsonl", val)
            (data / "test.jsonl").write_text("DO NOT READ\n", encoding="utf-8")
            original_train = (data / "train.jsonl").read_bytes()
            original_val = (data / "validation.jsonl").read_bytes()
            dest = root / "tagging"
            info = prepare(data, dest, negative_repeats=2,
                           multi_repeats=3, seed=1)
            self.assertEqual(info["original_train"]["negative"], 50)
            self.assertEqual(info["original_train"]["multi_tag"], 15)
            self.assertEqual(info["oversampled_train"]["negative"], 100)
            self.assertEqual(info["oversampled_train"]["multi_tag"], 45)
            self.assertEqual(info["oversampled_train"]["single_tag"], 100)
            self.assertEqual(info["train_total"], 245)
            self.assertEqual(info["validation_total"], 12)
            out_train = [json.loads(line) for line in
                         (dest / "train.jsonl").read_text().splitlines()]
            out_val = [json.loads(line) for line in
                       (dest / "validation.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["task"] == "fact_tagging"
                                for row in out_train + out_val))
            self.assertEqual(len(out_val), 12)
            self.assertFalse((dest / "test.jsonl").exists())
            self.assertEqual((data / "train.jsonl").read_bytes(), original_train)
            self.assertEqual((data / "validation.jsonl").read_bytes(), original_val)
            self.assertEqual((data / "test.jsonl").read_text(), "DO NOT READ\n")
            with self.assertRaises(FileExistsError):
                prepare(data, dest)

    def test_overlap_is_rejected_without_creation(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data = root / "input"
            data.mkdir()
            train = [item("t" + str(i), False, []) for i in range(101)]
            validation = [item("t0", False, [])] + [
                item("v" + str(i), False, []) for i in range(11)
            ]
            write(data / "train.jsonl", train)
            write(data / "validation.jsonl", validation)
            dest = root / "new"
            with self.assertRaisesRegex(ValueError, "leakage"):
                prepare(data, dest)
            self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
