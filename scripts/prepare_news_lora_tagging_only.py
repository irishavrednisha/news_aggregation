"""Create an *experimental* classification-only SFT run with mild oversampling.

Train records are only copied from input TRAIN; VALIDATION is only copied from
input VALIDATION. Test is never read, created, or changed. No labels are
generated, inferred, changed or claimed human verified. Original files remain
unchanged; new output directory must not exist. Duplicates are train-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def label_type(row: dict) -> str:
    """Return a strict category from the *already labelled* assistant JSON."""
    if row.get("task") != "fact_tagging":
        raise ValueError("Expected a fact_tagging record")
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Messages missing")
    if messages[-1].get("role") != "assistant":
        raise ValueError("Missing final assistant answer")
    label = json.loads(messages[-1]["content"])
    relevant, tags = label.get("relevant"), label.get("tags")
    if type(relevant) is not bool or not isinstance(tags, list):
        raise ValueError("Invalid relevance/tag labels")
    if relevant and len(tags) == 1:
        return "single_tag"
    if relevant and len(tags) == 2 and len(set(tags)) == 2:
        return "multi_tag"
    if not relevant and not tags:
        return "negative"
    raise ValueError("Inconsistent relevance/tags in training label")


def prepare(data_dir: Path, output_dir: Path, *,
            negative_repeats: int = 2, multi_repeats: int = 3,
            seed: int = 3407) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_dir}")
    if output_dir.resolve() == data_dir.resolve():
        raise ValueError("Output directory must differ from input")
    if not 1 <= negative_repeats <= 4 or not 1 <= multi_repeats <= 4:
        raise ValueError("Repeat factors must be between 1 and 4")
    train_rows = read_rows(data_dir / "train.jsonl")
    validation_rows = read_rows(data_dir / "validation.jsonl")
    original = Counter()
    emitted = Counter()
    selected_train = []
    train_story_ids = set()
    for row in train_rows:
        if row.get("task") != "fact_tagging":
            continue
        kind = label_type(row)
        story = row.get("story_id")
        if not story or story in train_story_ids:
            raise ValueError("Duplicate/missing story_id in train fact_tagging")
        train_story_ids.add(story)
        original[kind] += 1
        copies = {"negative": negative_repeats, "multi_tag": multi_repeats,
                  "single_tag": 1}[kind]
        emitted[kind] += copies
        # No fabricated labels; duplicates apply only to the train set.
        for _ in range(copies):
            selected_train.append(row)
    selected_validation = []
    validation_stories = set()
    val_classes = Counter()
    for row in validation_rows:
        if row.get("task") != "fact_tagging":
            continue
        kind = label_type(row)
        story = row.get("story_id")
        if not story or story in validation_stories:
            raise ValueError("Duplicate/missing story_id in validation")
        if story in train_story_ids:
            raise ValueError("Train-validation leakage: " + story)
        validation_stories.add(story)
        val_classes[kind] += 1
        selected_validation.append(row)
    if len(selected_train) < 100 or len(selected_validation) < 10:
        raise ValueError("Insufficient data for tagging-only diagnostic run")
    random.Random(seed).shuffle(selected_train)
    # Validate everything before writing. Never overwrite source files.
    output_dir.mkdir(parents=True)
    for split, rows in (("train", selected_train),
                        ("validation", selected_validation)):
        with (output_dir / (split + ".jsonl")).open("x", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "purpose": "tagging-only experimental SFT; no summary training",
        "warning": (
            "Train oversamples identical synthetic labels: no new examples. "
            "Validation is short, length-biased and AI-reviewed; NOT gold. "
            "Do not compare eval_loss directly to previous mixed-task runs."
        ),
        "original_train": dict(original),
        "oversampled_train": dict(emitted),
        "train_total": len(selected_train),
        "validation_total": len(selected_validation),
        "validation_labels": dict(val_classes),
        "negative_repeats": negative_repeats,
        "multi_repeats": multi_repeats,
        "seed": seed,
        "source_files_untouched": True,
        "test_not_accessed": True,
    }
    (output_dir / "README.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path,
                   default=Path("training/news_sft_short_1024"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("training/news_sft_tagging_only"))
    p.add_argument("--negative-repeats", type=int, default=2)
    p.add_argument("--multi-repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=3407)
    args = p.parse_args()
    print(json.dumps(prepare(
        args.data_dir, args.output_dir,
        negative_repeats=args.negative_repeats,
        multi_repeats=args.multi_repeats,
        seed=args.seed,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
