"""Compare Qwen3.5 base / LoRA checkpoint / final adapter on the same SFT prompts.

This is inference-only. It never trains, changes the collector or publishes.
Use validation for comparison; test needs explicit --allow-test. Save each
model's predictions to a separate JSONL for inspection of facts and summaries.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def select_rows(path: Path, limit_per_task: int) -> list[dict]:
    counts = Counter()
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get("task", "unknown")
            if limit_per_task > 0 and counts[task] >= limit_per_task:
                continue
            counts[task] += 1
            rows.append(row)
    return rows


def text_only_tokenizer(processor):
    """Return the underlying tokenizer, not Qwen3.5's multimodal processor.

    Positional processor(prompt) can be interpreted as an image argument and
    trigger torchvision.decode_image on a string instead of tokenizing text.
    """
    return getattr(processor, "tokenizer", processor)


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def decode_json_response(prediction: str) -> tuple[dict, bool]:
    """Parse complete JSON or one complete Markdown JSON fence.

    Never attempt to repair an incomplete/truncated model answer or silently
    convert an answer with a different schema into the requested schema.
    """
    import re

    cleaned = prediction.strip()
    wrapped = cleaned.startswith("```")
    if wrapped:
        matched = re.fullmatch(
            r"```(?:json|JSON)?[ \\t]*\\r?\\n([\\s\\S]*?)\\r?\\n?```",
            cleaned,
        )
        if matched is None:
            raise ValueError("Incomplete or malformed Markdown JSON fence")
        cleaned = matched.group(1).strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("Not a JSON object")
    return value, wrapped


def grade_response(task: str, reference: dict, prediction: str, user_prompt: str) -> dict:
    """Static, imperfect checks: never label summaries as factually verified."""
    from processing.local_news_llm import (
        validate_fact_decision, validate_summary,
    )
    try:
        parsed, markdown_wrapped = decode_json_response(prediction)
    except (ValueError, TypeError) as exc:
        return {
            "valid_json": False, "valid_contract": False,
            "error": str(exc), "parsed": None,
        }
    result = {
        "valid_json": True, "markdown_wrapped": markdown_wrapped,
        "valid_contract": False, "parsed": parsed,
    }
    if task == "fact_tagging":
        try:
            relevant, tags, evidence, uncertain = validate_fact_decision(
                parsed, user_prompt
            )
            result.update({
                "valid_contract": True, "relevant_correct":
                    relevant == reference.get("relevant"),
                "tags_exact": set(tags) == set(reference.get("tags", [])),
                "predicted_relevant": relevant,
                "predicted_tags": tags,
                "evidence_count": len(evidence),
                "uncertain": uncertain,
            })
        except (ValueError, TypeError) as exc:
            result.update({"valid_contract": False, "error": str(exc)})
    elif task == "abstractive_summary":
        try:
            quotes = json.loads(user_prompt.splitlines()[-1])
            summary = parsed.get("summary")
            warnings = validate_summary(summary, quotes)
            result.update({
                "valid_contract": len(warnings) == 0,
                "summary_chars": len(summary) if isinstance(summary, str) else 0,
                "warnings": warnings,
                "exact_reference_match": summary == reference.get("summary"),
            })
        except (ValueError, TypeError, IndexError) as exc:
            result.update({"valid_contract": False, "error": str(exc)})
    else:
        result.update({"valid_contract": False, "error": "Unknown task"})
    return result


def summarize(records: list[dict]) -> dict:
    count = len(records)
    ok_json = sum(r.get("grade", {}).get("valid_json", False) for r in records)
    valid = sum(r.get("grade", {}).get("valid_contract", False) for r in records)
    facts = [r for r in records if r["task"] == "fact_tagging"]
    summaries = [r for r in records if r["task"] == "abstractive_summary"]
    return {
        "records": count,
        "skipped_long_prompt": sum("skip_reason" in r for r in records),
        "valid_json": ok_json,
        "markdown_wrapped_json": sum(
            r.get("grade", {}).get("markdown_wrapped", False) for r in records
        ),
        "generation_hit_token_limit": sum(
            r.get("generation_hit_token_limit", False) for r in records
        ),
        "valid_contract": valid,
        "fact_count": len(facts),
        "relevance_correct": sum(
            r.get("grade", {}).get("relevant_correct", False) for r in facts
        ),
        "exact_tag_sets": sum(
            r.get("grade", {}).get("tags_exact", False) for r in facts
        ),
        "summary_count": len(summaries),
        "summary_valid_contract": sum(
            r.get("grade", {}).get("valid_contract", False) for r in summaries
        ),
        "warning": "AI-reviewed validation is diagnostic, not independent gold.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("training/news_sft"))
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--model", required=True,
                        help="Unsloth model id, checkpoint-50, or lora_adapter path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-task", type=int, default=8,
                        help="0 evaluates every record; positive number selects first N of each task")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("Test is held out. For a final audit pass --allow-test explicitly.")
    if args.max_new_tokens < 64 or args.max_seq_length < args.max_new_tokens + 128:
        parser.error("Sequence length must leave enough room for the input and output.")
    if args.model.startswith(("models/", "./", "../", "/")) and not Path(args.model).exists():
        parser.error("Local checkpoint/adapter not found: " + args.model)
    if args.output.exists():
        parser.error("Output already exists. Choose a fresh filename; never overwrite a comparison.")
    rows = select_rows(args.data_dir / (args.split + ".jsonl"), args.limit_per_task)
    if not rows:
        parser.error("No examples selected")

    from unsloth import FastLanguageModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    only_text_tokenizer = text_only_tokenizer(tokenizer)
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as destination:
        for index, item in enumerate(rows, 1):
            messages = item["messages"]
            task = item.get("task", "unknown")
            prompt = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            tokens = only_text_tokenizer(text=prompt, return_tensors="pt", add_special_tokens=False)
            prompt_len = tokens["input_ids"].shape[1]
            row = {
                "task": task, "story_id": item.get("story_id"),
                "model": args.model, "prompt_tokens": prompt_len,
                "reference": json.loads(messages[-1]["content"]),
            }
            if prompt_len + args.max_new_tokens > args.max_seq_length:
                row["skip_reason"] = "prompt_exceeds_context_budget"
            else:
                inputs = {k: v.to("cuda") for k, v in tokens.items()}
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens,
                        do_sample=False, use_cache=True,
                        pad_token_id=only_text_tokenizer.eos_token_id,
                    )
                completion = only_text_tokenizer.decode(
                    generated[0, prompt_len:], skip_special_tokens=True
                ).strip()
                row["prediction"] = completion
                row["completion_tokens"] = generated.shape[1] - prompt_len
                row["generation_hit_token_limit"] = (
                    row["completion_tokens"] >= args.max_new_tokens
                )
                row["grade"] = grade_response(
                    task, row["reference"], completion, messages[-2]["content"]
                )
                del inputs, generated
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            destination.flush()
            results.append(row)
            print(f"[{index}/{len(rows)}] {task}: "
                  f"{row.get('skip_reason') or row.get('grade', {}).get('valid_contract')}",
                  flush=True)
    metrics = summarize(results)
    metrics["model"] = args.model
    metrics["split"] = args.split
    metrics["output"] = str(args.output)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
