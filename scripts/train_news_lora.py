"""Train local Qwen3.5 text behavior on APPROVED news SFT examples.

This is optional, GPU-only and requires an EXPLICIT --run flag.
Use a separate WSL2/Linux virtualenv: Python, torch CUDA, unsloth,
transformers>=5, datasets, trl. Do not install training deps into the
news collector's running .venv.

Input: training/news_sft/train.jsonl & validation.jsonl from build_news_sft.py
No database or MAX API use. Synthetic examples are train-only, verified
real examples are required to evaluate generalization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from inspect import signature


VRAM_GB = {
    "unsloth/Qwen3.5-2B": 5,
    "unsloth/Qwen3.5-4B": 10,
    "unsloth/Qwen3.5-9B": 22,
}


def trl_api_compat(trainer_type, config_type) -> tuple[str, str]:
    """Identify API names used by installed TRL or Unsloth's patched TRL."""
    trainer_params = signature(trainer_type).parameters
    config_params = signature(config_type).parameters
    if "processing_class" in trainer_params:
        tokenizer_arg = "processing_class"
    elif "tokenizer" in trainer_params:
        tokenizer_arg = "tokenizer"
    else:
        raise RuntimeError("Unsupported TRL SFTTrainer: missing tokenizer/processing_class")
    if "max_length" in config_params:
        length_arg = "max_length"
    elif "max_seq_length" in config_params:
        length_arg = "max_seq_length"
    else:
        raise RuntimeError("Unsupported TRL SFTConfig: missing max_length/max_seq_length")
    return tokenizer_arg, length_arg


def length_issues(report: dict) -> dict[str, dict[str, int]]:
    """Detect examples that silently lose training/evaluation labels."""
    result = {}
    for split in ("train", "validation"):
        tasks = report.get(split, {})
        truncated = sum(s["full_truncated"] for s in tasks.values())
        unaligned = sum(s["possible_template_mismatch"] for s in tasks.values())
        no_reply = sum(s["no_reply_tokens"] for s in tasks.values())
        if truncated or unaligned or no_reply:
            result[split] = {
                "truncated": truncated,
                "no_reply_tokens": no_reply,
                "unaligned": unaligned,
            }
    return result


def count_records(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def small_validation_allowed(train_path: Path, val_path: Path) -> bool:
    """Only allow a small validation for an explicit tagging-only experiment."""
    tasks = set()
    for path in (train_path, val_path):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    task = json.loads(line).get("task")
                    if task != "fact_tagging":
                        return False
                    tasks.add(task)
    return tasks == {"fact_tagging"}


def main() -> None:
    p = argparse.ArgumentParser(description="Явно запустить bf16 LoRA для автономных новостей")
    p.add_argument("--data-dir", type=Path, default=Path("training/news_sft"))
    p.add_argument("--model", choices=tuple(VRAM_GB),
                   default="unsloth/Qwen3.5-2B")
    p.add_argument("--output-dir", type=Path, default=Path("models/news_qwen_lora"))
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--assistant-loss-only", action="store_true",
                   help="Opt in to prompt/completion loss only on assistant JSON")
    p.add_argument("--allow-small-validation", action="store_true",
                   help="Only for tagging-only diagnostic validation with >=10 real examples")
    p.add_argument("--run", action="store_true",
                   help="Без него обучение НЕ запускается")
    args = p.parse_args()
    train_path = args.data_dir / "train.jsonl"
    val_path = args.data_dir / "validation.jsonl"
    if not train_path.is_file() or not val_path.is_file():
        p.error("Сначала запусти python -m scripts.build_news_sft ...")
    train_count, val_count = count_records(train_path), count_records(val_path)
    print(f"Train messages: {train_count}; validation messages: {val_count}")
    print("NOTICE: validation is AI-reviewed, not independently human-verified gold.")
    if not args.run:
        print("DRY RUN: обучение не запускалось. Для запуска добавь --run.")
        return
    if train_count < 100:
        p.error("Нужно >=100 качественных train сообщений")
    if val_count < 20:
        if not (
            args.allow_small_validation
            and args.assistant_loss_only
            and val_count >= 10
            and small_validation_allowed(train_path, val_path)
        ):
            p.error(
                "Нужно >=20 real validation сообщений; разрешение на 10–19 "
                "только для явно указанного --allow-small-validation + "
                "--assistant-loss-only и tagging-only dataset"
            )
        print(
            "WARNING: Only", val_count,
            "length-biased, AI-reviewed tagging validation records! "
            "Metric is highly uncertain and is not a gold benchmark."
        )
    if args.max_steps < 1 or args.max_seq_length < 512:
        p.error("Некорректное число шагов или длина контекста")

    # Unsloth must patch Transformers before its first import.
    from unsloth import FastLanguageModel
    # Stop before heavyweight model loading if labels will be truncated.
    from transformers import AutoTokenizer
    from scripts.audit_news_lora import audit as audit_lengths
    checking_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    errors = length_issues(
        audit_lengths(args.data_dir, checking_tokenizer, args.max_seq_length)
    )
    del checking_tokenizer
    if errors:
        p.error(
            "SFT length preflight failed: " + json.dumps(errors, ensure_ascii=False)
            + ". Use a separate length-compatible validation for diagnostics or "
              "a sufficiently large GPU-compatible context. Original data unchanged."
        )

    # Unsloth was imported above, before Transformers.
    import torch
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig
    if not torch.cuda.is_available():
        p.error("Обучение bf16 LoRA требует CUDA GPU, на CPU не запускаем")
    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print("GPU:", torch.cuda.get_device_name(0),
          f"{gpu_total:.1f} GiB VRAM")
    if gpu_total + 0.5 < VRAM_GB[args.model]:
        p.error(f"Для {args.model} ориентир Unsloth: {VRAM_GB[args.model]} ГБ VRAM; "
                "выбери модель поменьше")
    # Desktop GPU memory is shared with the display server. A model that
    # fits in total VRAM can still fail if Ollama, games or browsers use it.
    free_bytes, _ = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024**3)
    print(f"Currently free GPU memory: {free_gb:.1f} GiB")
    if free_gb < VRAM_GB[args.model]:
        p.error("Недостаточно свободной GPU-памяти: закрой Ollama/игры "
                "или уменьши длину контекста/модель, затем повтори")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=False,  # Qwen3.5: do not default to QLoRA
        load_in_16bit=True,
        full_finetuning=False,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=args.max_seq_length,
    )

    ds = load_dataset("json", data_files={
        "train": str(train_path), "validation": str(val_path),
    })
    if args.assistant_loss_only:
        def to_prompt_completion(batch):
            messages = batch["messages"]
            prefix = tokenizer.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            complete = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )
            if not complete.startswith(prefix):
                raise ValueError("Prompt is not prefix of rendered assistant chat")
            return {"prompt": prefix, "completion": complete[len(prefix):]}
        ds = ds.map(to_prompt_completion)
        ds = ds.remove_columns([
            column for column in ds["train"].column_names
            if column not in ("prompt", "completion")
        ])
    else:
        def to_text(batch):
            # Legacy full-dialogue loss includes prompt tokens too.
            messages = batch["messages"]
            return {"text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
                enable_thinking=False,
            )}
        ds = ds.map(to_text)
        ds = ds.remove_columns([
            column for column in ds["train"].column_names if column != "text"
        ])

    # TRL 1.x uses processing_class/max_length; some Unsloth builds still
    # patch in tokenizer/max_seq_length. Detect the actual installed API.
    tokenizer_arg, length_arg = trl_api_compat(SFTTrainer, SFTConfig)
    sft_config_kwargs = {length_arg: args.max_seq_length}
    if args.assistant_loss_only:
        if "completion_only_loss" not in signature(SFTConfig).parameters:
            p.error("Installed TRL has no completion_only_loss")
        sft_config_kwargs["completion_only_loss"] = True
    trainer = SFTTrainer(
        model=model, **{tokenizer_arg: tokenizer},
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        args=SFTConfig(
            output_dir=str(args.output_dir),
            **sft_config_kwargs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            warmup_steps=min(20, max(1, args.max_steps // 10)),
            max_steps=args.max_steps,
            logging_steps=min(10, args.max_steps),
            eval_strategy="steps",
            eval_steps=min(50, args.max_steps),
            save_strategy="steps",
            save_steps=min(50, args.max_steps),
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="adamw_8bit",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            seed=3407,
            report_to="none",
            dataset_num_proc=1,
        ),
    )
    trainer.train()
    print("Best checkpoint by diagnostic validation loss:",
          trainer.state.best_model_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "lora_adapter"))
    print("LoRA saved:", args.output_dir / "lora_adapter")
    print("ВАЖНО: этот адаптер ещё нужно протестировать и отдельно "
          "экспортировать в локальный Ollama/GGUF. Он НЕ включён в MAX.")


if __name__ == "__main__":
    main()
