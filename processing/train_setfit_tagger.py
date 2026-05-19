import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

import pandas as pd

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments


DATASET_PATH = ROOT_DIR / "data" / "tagging_train.csv"
MODEL_OUTPUT_DIR = ROOT_DIR / "models" / "setfit_industry_tagger"

BASE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_training_dataset() -> Dataset:
    df = pd.read_csv(DATASET_PATH)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("В CSV должны быть колонки: text, label")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)

    return Dataset.from_pandas(df[["text", "label"]])


def main():
    dataset = load_training_dataset()

    labels = sorted(set(dataset["label"]))

    print("Количество примеров:", len(dataset))
    print("Теги:", labels)

    model = SetFitModel.from_pretrained(
        BASE_MODEL_NAME,
        labels=labels
    )

    args = TrainingArguments(
        batch_size=8,
        num_epochs=1,
        num_iterations=20,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    trainer.train()

    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(MODEL_OUTPUT_DIR))

    print(f"Модель сохранена в: {MODEL_OUTPUT_DIR}")


if __name__ == "__main__":
    main()