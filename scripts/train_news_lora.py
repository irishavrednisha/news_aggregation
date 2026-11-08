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


VRAM_GB = {
    "unsloth/Qwen3.5-2B": 5,
    "unsloth/Qwen3.5-4B": 10,
    "unsloth/Qwen3.5-9B": 22,
}


def count_records(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def main() -> None:
    p = argparse.ArgumentParser(description="Явно запустить bf16 LoRA для автономных новостей")
    p.add_argument("--data-dir", type=Path, default=Path("training/news_sft"))
    p.add_argument("--model", choices=tuple(VRAM_GB),
                   default="unsloth/Qwen3.5-2B")
    p.add_argument("--output-dir", type=Path, default=Path("models/news_qwen_lora"))
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--run", action="store_true",
                   help="Без него обучение НЕ запускается")
    args = p.parse_args()
    train_path = args.data_dir / "train.jsonl"
    val_path = args.data_dir / "validation.jsonl"
    if not train_path.is_file() or not val_path.is_file():
        p.error("Сначала запусти python -m scripts.build_news_sft ...")
    train_count, val_count = count_records(train_path), count_records(val_path)
    print(f"Train messages: {train_count}; validated real validation: {val_count}")
    if not args.run:
        print("DRY RUN: обучение не запускалось. Для запуска добавь --run.")
        return
    if train_count < 100 or val_count < 20:
        p.error("Нужно >=100 качественных train и >=20 независимых real validation сообщений")
    if args.max_steps < 1 or args.max_seq_length < 512:
        p.error("Некорректное число шагов или длина контекста")

    # Import Unsloth first, per supported local fine-tuning setup.
    from unsloth import FastLanguageModel
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
    def to_text(batch):
        # Same chat layout as inference; no fabricated chain-of-thought.
        messages = batch["messages"]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )}
    ds = ds.map(to_text)
    ds = ds.remove_columns([k for k in ds["train"].column_names if k != "text"])

    # Unsloth's version-pinned training recipes use max_seq_length and
    # tokenizer. If a future TRL major version changes these APIs,
    # follow Unsloth's matching versioned documentation.
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        args=SFTConfig(
            output_dir=str(args.output_dir),
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            warmup_steps=20,
            max_steps=args.max_steps,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=50,
            save_total_limit=2,
            optim="adamw_8bit",
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            seed=3407,
            report_to="none",
            dataset_num_proc=1,
        ),
    )
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(args.output_dir / "lora_adapter"))
    print("LoRA saved:", args.output_dir / "lora_adapter")
    print("ВАЖНО: этот адаптер ещё нужно протестировать и отдельно "
          "экспортировать в локальный Ollama/GGUF. Он НЕ включён в MAX.")


if __name__ == "__main__":
    main()
