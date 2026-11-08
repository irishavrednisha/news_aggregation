"""Build leakage-resistant SFT records for local Qwen news tagging + summaries.

Input JSON format:
 {"examples": [
   {"story_id":"unique-event-A", "origin":"synthetic"|"real",
    "split":"train"|"validation"|"test", "review_status":"verified",
    "title":"...", "text":"...", "relevant":true,
    "tags":["машиностроение"],
    "evidence":["Exact passage from source at least 15 chars"],
    "summary":"A short, factual paraphrase written or reviewed by an editor"}
 ]}
No model queries, no training, no databases, no MAX.
Synthetic items can only go to TRAIN and require --allow-synthetic.
Synthetic auto_checked items additionally need --allow-auto-checked-synthetic
to avoid silently turning model-generated answers into gold labels.
REAL validated test items are stored separately and NEVER put in train.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

from processing.local_news_llm import (
    FACT_SYSTEM, SUMMARY_SYSTEM, extract_messages, summary_messages,
    source_payload, validate_fact_decision, validate_summary,
)


def example_to_messages(item: dict) -> list[dict]:
    title, article = item["title"], item["text"]
    source = source_payload(title, article)
    relevant, tags, evidence, uncertain = validate_fact_decision({
        "relevant": item["relevant"],
        "tags": item["tags"],
        "evidence": item["evidence"],
        "uncertain": False,
    }, source)
    if uncertain:
        raise ValueError("Verified SFT record cannot be uncertain")
    if not relevant and item.get("summary"):
        raise ValueError("A negative example must not contain a summary")
    messages = []
    fact_msg = extract_messages(title, article) + [
        {"role": "assistant", "content": json.dumps({
            "relevant": relevant, "tags": tags,
            "evidence": evidence, "uncertain": False,
        }, ensure_ascii=False)}
    ]
    messages.append({"messages": fact_msg, "task": "fact_tagging"})
    if relevant:
        summary = item.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("Positive records need editor-reviewed abstractive summary")
        problems = validate_summary(summary, evidence)
        if problems:
            raise ValueError("Invalid training summary: " + "; ".join(problems))
        summary_msg = summary_messages(evidence) + [
            {"role": "assistant", "content": json.dumps({
                "summary": summary.strip()
            }, ensure_ascii=False)}
        ]
        messages.append({"messages": summary_msg, "task": "abstractive_summary"})
    return messages


def build_dataset(
    examples: list[dict], allow_synthetic: bool = False,
    allow_auto_checked_synthetic: bool = False,
) -> dict[str, list]:
    datasets = {"train": [], "validation": [], "test": []}
    seen = {}
    dropped = Counter()
    for index, item in enumerate(examples):
        origin = item.get("origin")
        split = item.get("split")
        status = item.get("review_status")
        story = str(item.get("story_id") or "").strip()
        if origin not in {"synthetic", "real"}:
            raise ValueError(f"row {index}: origin must be synthetic or real")
        if split not in datasets:
            raise ValueError(f"row {index}: set split=train/validation/test")
        if not story:
            raise ValueError(f"row {index}: missing story_id")
        # A story/event is one unit: related variants must never cross splits.
        if story in seen:
            if seen[story] != split:
                raise ValueError(f"Leakage: story_id {story} appears in different splits")
            raise ValueError(f"Duplicate story_id {story}")
        seen[story] = split
        if origin == "synthetic":
            if split != "train":
                raise ValueError("Synthetic items must never be used as validation/test")
            if not allow_synthetic:
                dropped["synthetic_not_enabled"] += 1
                continue
        accepted_status = (status == "verified") or (
            origin == "synthetic" and status == "auto_checked"
            and allow_synthetic and allow_auto_checked_synthetic
        )
        if not accepted_status:
            dropped["not_verified"] += 1
            continue
        messages = example_to_messages(item)
        for row in messages:
            datasets[split].append({
                "messages": row["messages"],
                "task": row["task"],
                "origin": origin,
                "story_id": story,
            })
    return {"splits": datasets, "skipped": dict(dropped)}


def main() -> None:
    p = argparse.ArgumentParser(description="Подготовить проверенную SFT-разметку для Qwen")
    p.add_argument("input", type=Path,
                   help="JSON: {examples:[...]} с validated source spans & summary")
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument(
        "--allow-auto-checked-synthetic", action="store_true",
        help="Явное разрешение SFT на автоматически проверенных искусственных"
             " новостях, только для TRAIN, без подтверждения качества",
    )
    p.add_argument("--output-dir", type=Path,
                   default=Path("training/news_sft"))
    args = p.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data.get("examples"), list):
        p.error("Ожидался объект JSON с examples")
    prepared = build_dataset(
        data["examples"], args.allow_synthetic,
        args.allow_auto_checked_synthetic,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in prepared["splits"].items():
        outfile = args.output_dir / f"{split}.jsonl"
        with outfile.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} учебных сообщений → {outfile}")
    print("Пропущено:", prepared["skipped"])
    print("NB: валидация/test с реальными и независимыми историями обязательна "
          "перед внедрением модели для публикации.")


if __name__ == "__main__":
    main()
