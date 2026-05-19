from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
SETFIT_MODEL_DIR = ROOT_DIR / "models" / "setfit_industry_tagger"

_setfit_model_cache = None


def load_setfit_model():
    """
    Загружает SetFit-модель из локальной папки.

    Если модель ещё не обучена и папки нет, возвращает None.
    Благодаря этому основной pipeline не падает.
    """
    global _setfit_model_cache

    if _setfit_model_cache is not None:
        return _setfit_model_cache

    if not SETFIT_MODEL_DIR.exists():
        print(f"SetFit-модель не найдена: {SETFIT_MODEL_DIR}")
        _setfit_model_cache = None
        return None

    try:
        from setfit import SetFitModel

        _setfit_model_cache = SetFitModel.from_pretrained(str(SETFIT_MODEL_DIR))
        print(f"SetFit-модель загружена: {SETFIT_MODEL_DIR}")

        return _setfit_model_cache

    except Exception as error:
        print(f"Ошибка загрузки SetFit-модели: {error}")
        _setfit_model_cache = None
        return None


def get_setfit_scores(text: str) -> dict[str, float]:
    """
    Возвращает вероятности SetFit по тегам.

    Пример результата:
    {
        "авиация": 0.82,
        "судостроение": 0.04,
        ...
    }
    """
    if not text:
        return {}

    model = load_setfit_model()

    if model is None:
        return {}

    try:
        proba = model.predict_proba([text])[0]

        labels = list(model.labels)

        result = {}

        for label, score in zip(labels, proba):
            result[str(label)] = float(score)

        return result

    except Exception as error:
        print(f"Ошибка SetFit-предсказания: {error}")
        return {}


def get_setfit_best_tag(text: str) -> tuple[str | None, float]:
    scores = get_setfit_scores(text)

    if not scores:
        return None, 0.0

    best_tag = max(scores, key=scores.get)
    best_score = scores[best_tag]

    return best_tag, best_score