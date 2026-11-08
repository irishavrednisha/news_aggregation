# Qwen3.5-2B LoRA: диагностика после первого запуска

Первый запуск на Windows RTX 5060 Ti / WSL2 выполнил 250 шагов.
Сохранённый адаптер: models/news_qwen_lora/lora_adapter.
Значения eval_loss на ИИ-размеченной validation: 50=1.098, 100=1.274,
150=1.355, 200=1.373, 250=1.399. Меньше — лучше, но эти значения
сами по себе не доказывают качество тегов и фактическую точность сводок.

**До нового обучения и до выбора адаптера** проверьте, сколько целевых
ответов вообще оставалось после усечения последовательности до 1024 токенов.
Текущий скрипт обучения использует text-column и вычисляет loss по тексту
всего диалога, а не исключительно по финальному ответу assistant.
Если prompt уже длиннее 1024 токенов, ответ может быть вытеснен целиком.
Это может исказить результаты и требует отдельного исправления в будущем.

    cd ~/projects/news_aggregation
    git pull --ff-only origin fix/site-specific-news-collection-2026-11-06
    source ~/qwen-env/bin/activate
    python -m scripts.audit_news_lora --data-dir training/news_sft --max-length 1024 --output reports/token_audit_v2.json

После аудита сравните исходную модель и реально сохранившиеся адаптеры.
Старый запуск использовал save_total_limit=2 без load_best_model_at_end,
поэтому checkpoint-50, вероятно, удалён. Проверьте содержимое папки:

    find models/news_qwen_lora -maxdepth 2 -name adapter_config.json -print

Сравните на **одной и той же validation**, а НЕ test, например по 8
примеров двух задач. Каждый запуск загружает лишь одну модель в GPU:

    python -m scripts.evaluate_news_lora --model unsloth/Qwen3.5-2B --output reports/val_base.jsonl
    python -m scripts.evaluate_news_lora --model models/news_qwen_lora/checkpoint-200 --output reports/val_step200.jsonl
    python -m scripts.evaluate_news_lora --model models/news_qwen_lora/lora_adapter --output reports/val_step250.jsonl

В reports/val_*.metrics.json будут числа валидных JSON, решений о релевантности,
точных наборов тегов и формально допустимых сводок. В JSONL хранятся
предсказания и эталоны для проверки; фактичность summary этими
эвристиками НЕ подтверждается. Слишком длинные prompt будут явно
пропущены, НЕ обрезаны скрытно.

**Ограничения:** validation (67 сообщений) и test (60) размечены ИИ, а
не независимым человеком. Это диагностические проверки, не независимый
gold test. Не оценивать test при подборе параметров, не включать
автоматическую публикацию в MAX. Для полноценного сравнения нужно
проверять реальные события и точность доказательств отдельно.

Обновлённый train_news_lora.py с будущего запуска сохраняет лучшую по
validation loss контрольную точку даже при save_total_limit=2 и загружает
её перед сохранением lora_adapter. Это не меняет уже законченный запуск;
метрика AI validation по-прежнему не является независимой оценкой.

Исправление аудита: прежний скрипт считал число ключей словаря
BatchEncoding (2), а не длину input_ids. Результаты прежнего отчёта
с prompt_tokens_total=8000 и full_tokens_total=8000 при 4000 статьях
НЕВЕРНЫ. Новый скрипт рендерит text-column как при обучении, затем
токенизирует строку и проверяет совпадение префикса. Число
no_reply_tokens достоверно только для aligned_records; случаи
possible_template_mismatch отдельно выделяются и не включаются в него.

## 2026-11-08 — length audit and new safeguards

Original 1024-token SFT run: train=6700/6700 fit, validation=33/67
truncated (29 with zero answer tokens), test=25/60 truncated
(21 with zero answer tokens). Validation loss from the first run is
not a reliable indicator of real article tagging.

New training runs now fail their preflight if any train or validation
example is truncated, even if the data format is otherwise valid.
For technical experiments with 1024, generate a separate biased
short-validation dataset (original train, validation, test untouched):

    python -m scripts.prepare_news_lora_short_validation --data-dir training/news_sft --output-dir training/news_sft_short_1024 --max-length 1024

For experimental assistant-reply-only loss, add --assistant-loss-only
to train_news_lora.py (requires modern TRL prompt-completion support).
Before using it for a full run, perform a 10-step CUDA smoke test.
Short-validation metrics MUST NOT be compared directly against the old
67-example full-validation metrics. Test remains held out.

## 2026-11-08 — отдельный эксперимент с тегированием

Из 4000 исходных fact_tagging train: 2520 с одним тегом, 180 с двумя,
1300 нерелевантных. Это 67.5% положительных и только 6.7% двухтеговых
среди положительных. 18 кратких реальных validation имеют смещение
по длине; ИИ-разметка не является независимой проверкой.

Для предварительного исследования без модификации исходных файлов
подготовьте отдельную tagging-only выборку:

    python -m scripts.prepare_news_lora_tagging_only --data-dir training/news_sft_short_1024 --output-dir training/news_sft_tagging_only --negative-repeats 2 --multi-repeats 3

Систематическое повторение историй в train — только изменение веса
существующих размеченных примеров, НЕ создание новых уникальных событий
и НЕ устранение смещения синтетики. Проверочная validation не меняется,
тест не читается. Не считать такой эксперимент подтверждением улучшения.

Tagging-only validation состоит из 18 историй, поэтому запуск
train_news_lora.py требует явные флаги --allow-small-validation и
--assistant-loss-only. Отдельная модель предназначена только для
тегирования. Для сводок пока использовать базовую модель с проверкой
источников — прошлый мультизадачный LoRA 250 выдумывал предприятия.

## 2026-11-08 — Готовая Gemma 3 4B только для переписывания TextRank

Новый автономный вариант не использует LoRA, не сокращает исходную статью
ещё раз и не вмешивается в работающий publisher. TextRank выбирает
предложения, Gemma 3 4B переформулирует каждое из них отдельно. Количество
предложений и порядок должны сохраняться, длина отдельного предложения
обычно должна оставаться в пределах 85–145% исходного. Проверяются числа,
кавычки, некоторые новые названия и маркеры планов/неопределённости.
Если программная проверка не пройдена, возвращается исходный TextRank.
Если прошла — это только черновик: проверка не доказывает
фактической эквивалентности, авто-публикация отключена.

В Windows PowerShell:

    ollama pull gemma3:4b
    cd D:\Projects\news_aggregation
    git pull --ff-only origin fix/site-specific-news-collection-2026-11-06
    .\.venv\Scripts\python.exe check_extractive_rewrite.py tagging_fresh_25.json --limit 3 --output collector_test_results/gemma3_rewrite_preview.json

Сравните поля `extractive`, `text`, `rewritten`, `warnings`.
Скрипт только переписывает, существующие теги не заменяет.
Независимое смысловое подтверждение и интеграция с публикацией — отдельные задачи.

2026-11-08: результаты Gemma на трёх новостях: 1 черновик принят,
2 остались TextRank. Причина ложного блокирования слова «возможности»
исправлена: потенциальные возможности разработки не являются
неопределённостью результата. При этом модель не должна завершать
предложение запятой вместо точки. Все ответные фразы модели теперь
сохраняются в `model_candidate` даже при отказе проверки, чтобы
отличать реальную фактическую ошибку от ложного запрета. Поле
`model_candidate` — НЕ проверенная сводка и НЕ для публикации.
