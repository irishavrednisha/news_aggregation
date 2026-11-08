"""Check whether SFT text messages fit the training token budget.

Read-only: never trains, publishes, or modifies the input dataset.
The current text-column SFTTrainer loss includes the prompt as well as the
assistant reply. This tool quantifies truncation and surviving reply tokens.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def surviving_reply_tokens(prompt_tokens: int, full_tokens: int, limit: int) -> int:
    return max(0, min(full_tokens, limit) - prompt_tokens)


def rendered_token_ids(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> list[int]:
    """Match the training text-column path; never count BatchEncoding keys.

    For Qwen3.5, tokenize=True may yield a mapping with input_ids and
    attention_mask. len(mapping) is 2, NOT the sequence length. Instead
    render exactly as training does, then tokenize that rendered string.
    """
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise TypeError(
            "Chat template must render a non-empty string for text-column SFT"
        )
    encoding = tokenizer(rendered, add_special_tokens=False)
    token_ids = encoding["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Expected one rendered chat, got a batch")
        token_ids = token_ids[0]
    if not isinstance(token_ids, list) or not all(
        isinstance(token, int) for token in token_ids
    ):
        raise TypeError("Tokenizer did not return a flat list of token IDs")
    return token_ids


def audit(data_dir: Path, tokenizer, max_length: int) -> dict:
    report = {}
    for split in ("train", "validation", "test"):
        path = data_dir / (split + ".jsonl")
        if not path.is_file():
            continue
        by_task = defaultdict(lambda: {
            "records": 0, "full_truncated": 0, "no_reply_tokens": 0,
            "reply_tokens_total": 0, "reply_tokens_surviving": 0,
            "prompt_tokens_total": 0, "full_tokens_total": 0,
            "prompt_tokens_max": 0, "full_tokens_max": 0,
            "aligned_records": 0, "possible_template_mismatch": 0,
        })
        with path.open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                item = json.loads(line)
                messages = item["messages"]
                task = item.get("task", "unknown")
                full_ids = rendered_token_ids(
                    tokenizer, messages, add_generation_prompt=False,
                )
                prompt_ids = rendered_token_ids(
                    tokenizer, messages[:-1], add_generation_prompt=True,
                )
                prompt_len, full_len = len(prompt_ids), len(full_ids)
                metrics = by_task[task]
                metrics["records"] += 1
                metrics["full_truncated"] += int(full_len > max_length)
                metrics["prompt_tokens_total"] += prompt_len
                metrics["full_tokens_total"] += full_len
                metrics["prompt_tokens_max"] = max(
                    metrics["prompt_tokens_max"], prompt_len
                )
                metrics["full_tokens_max"] = max(
                    metrics["full_tokens_max"], full_len
                )
                if full_ids[:prompt_len] != prompt_ids:
                    metrics["possible_template_mismatch"] += 1
                    # Do not fabricate reply-token statistics when the prompt
                    # boundary cannot be located reliably.
                    continue
                count = surviving_reply_tokens(prompt_len, full_len, max_length)
                metrics["aligned_records"] += 1
                metrics["no_reply_tokens"] += int(count == 0)
                metrics["reply_tokens_total"] += max(0, full_len - prompt_len)
                metrics["reply_tokens_surviving"] += count
        report[split] = dict(by_task)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("training/news_sft"))
    parser.add_argument("--model", default="unsloth/Qwen3.5-2B")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    report = {
        "training_token_limit": args.max_length,
        "loss_warning": "Training text-column SFT loss includes prompt tokens.",
        "note": "Reply survival counts cover aligned records only; check possible_template_mismatch.",
        "splits": audit(args.data_dir, tokenizer, args.max_length),
    }
    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
