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
