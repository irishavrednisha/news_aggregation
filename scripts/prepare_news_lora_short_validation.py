"""Create a separate, length-biased diagnostic validation; NEVER touch test."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

from scripts.audit_news_lora import rendered_token_ids


def prepare(data_dir: Path, out_dir: Path, tokenizer, max_length: int) -> dict:
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")
    if out_dir.resolve() == data_dir.resolve():
        raise ValueError("Output directory must differ from input directory")
    train_path = data_dir / "train.jsonl"
    validation_path = data_dir / "validation.jsonl"
    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Missing train.jsonl or validation.jsonl")
    kept, accepted, skipped = [], Counter(), Counter()
    with validation_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            tokens = rendered_token_ids(
                tokenizer, row["messages"], add_generation_prompt=False,
            )
            task = row.get("task", "unknown")
            if len(tokens) > max_length:
                skipped[task] += 1
            else:
                kept.append(line if line.endswith("\n") else line + "\n")
                accepted[task] += 1
    if len(kept) < 20 or accepted.get("fact_tagging", 0) == 0:
        raise ValueError("Not enough fully fitting validation examples")
    out_dir.mkdir(parents=True)
    shutil.copyfile(train_path, out_dir / "train.jsonl")
    (out_dir / "validation.jsonl").write_text(
        "".join(kept), encoding="utf-8"
    )
    report = {
        "warning": "AI-reviewed and length-biased, NOT independent gold.",
        "unchanged_original_validation_and_test": True,
        "max_length": max_length,
        "validation_kept": dict(accepted),
        "validation_skipped": dict(skipped),
    }
    (out_dir / "README.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("training/news_sft"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("training/news_sft_short_1024"))
    p.add_argument("--model", default="unsloth/Qwen3.5-2B")
    p.add_argument("--max-length", type=int, default=1024)
    args = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    print(json.dumps(prepare(args.data_dir, args.output_dir, tokenizer,
                             args.max_length), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
