# Этап 4: Qwen3.5 9B — абстрактивная суммаризация с проверяемыми цитатами

## Что реализовано

Новая реализация `processing/qwen_abstractive_summary.py` работает **отдельно**
от Gemma 3 4B. Она не вызывает публикацию в MAX, не обращается к PostgreSQL,
не меняет исходные новости, утверждённые администратором тексты и `master`.
Основной конвейер не переключён на Qwen автоматически.

- Выбранная локальная модель: `qwen3.5:9b-q4_K_M` из Ollama.
  Вызовы идут через `http://127.0.0.1:11434/api/chat`, только по loopback.
- Модель сама определяет событие по тексту, **без обязательного
  регулярного выражения `no_supported_news_event`**.
- Полный текст передаётся, если он входит в ограничение контекста.
  Для длинных статей остаются начало, релевантные фрагменты из середины,
  предложения с числами и конец. Исходники остаются целыми в JSON-отчёте.
- При нескольких публикациях сохраняются ID и URL источников; подозрение
  на числовое/статусное противоречие переводит результат на проверку.
- Генерация идёт через JSON Schema: `main_event`, `is_news_event`,
  `summary`, `claims` (отдельное утверждение на предложение, source_ids,
  evidence_quotes), `uncertainties`, `requires_review`.
- Проверка требует **точного** присутствия цитат в указанных оригиналах,
  сохранения чисел, единиц, названий и отдельных признаков смысла:
  планы/результат, рост/снижение, отрицание, причина, организация.
- `summary_draft` означает только отсутствие обнаруженных механических
  проблем. **Это не одобрение редактора, не доказательство достоверности
  и не основание для автопубликации.** Все кандидаты требуют ручной проверки.

Самопроверка Qwen не является независимой верификацией. Механические
проверки также не способны гарантированно доказать, что организация,
действие и число относятся к одному событию. Неподтверждённые связи
обязательно оцениваются человеком. TextRank и Gemma сохранены без изменений.

## Настройка Windows 11 / PowerShell

```powershell
cd D:\Projects\news_aggregation
git fetch origin
git switch --track origin/feat/stage4-qwen35-9b-auditable-2026-11-20

ollama list
ollama ps

.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_stage4_qwen.py" -v
```

Если локальная ветка уже создана: `git switch feat/stage4-qwen35-9b-auditable-2026-11-20`,
затем `git pull --ff-only`. Активация .venv через Activate.ps1 не требуется.

Важно: **файл старых 91 реальных материалов должен существовать локально**.
Он не загружается с сайтов повторно. По документации прежнего теста он
называется `collector_test_results\stage4_independent_holdout\gemma_holdout.json`.
При другом расположении замени путь в команде.

## Сначала три статьи (smoke test)

```powershell
.\.venv\Scripts\python.exe scripts\compare_stage4_qwen.py `
  --input-json collector_test_results\stage4_independent_holdout\gemma_holdout.json `
  --limit 3 `
  --output collector_test_results\stage4_qwen_smoke3.json
```

Скрипт пишет отдельные:
- `stage4_qwen_smoke3.json` — источники, Qwen/Gemma, цитаты,
  валидация, ошибки, время, метрики Ollama и снимки GPU;
- `stage4_qwen_smoke3.csv` — сравнение одинаковых исходников;
- `stage4_qwen_smoke3.md` — удобная ручная проверка;
- `stage4_qwen_smoke3_human_review.csv` — пустые графы оценки редактора
  (файл **не перезаписывается** при повторном запуске).

Если локальная модель не отвечает, JSON с причиной ошибки всё равно
сохраняется; не нужно запускать тест повторно вслепую.

## Диагностическое сравнение на 91 сохранённой статье

```powershell
.\.venv\Scripts\python.exe scripts\compare_stage4_qwen.py `
  --input-json collector_test_results\stage4_independent_holdout\gemma_holdout.json `
  --model qwen3.5:9b-q4_K_M `
  --limit 91 `
  --num-ctx 4096 `
  --num-predict 1400 `
  --max-input-chars 6000 `
  --output collector_test_results\stage4_qwen_vs_gemma_91.json
```

После прерывания повтори ту же команду с `--resume`: готовые результаты
переиспользуются при совпадении исходного SHA256, модели, параметров
и входного файла. Никаких записей в PostgreSQL.

Для более медленной работы/ошибок нехватки памяти сначала уменьши
`--max-input-chars` до 4500 либо `--num-predict` до 1100.
**Не увеличивай `num_ctx` до проверки реального потребления VRAM**.
Скорость в токенах в секунду берётся из `eval_count / eval_duration`
локального Ollama, время запроса — фактически измеренное.

Если имеется **только** исходный `holdout_manifest.json`, передай
`--input-json <путь к holdout_manifest.json>` и отдельно
`--gemma-json <путь к gemma_holdout.json>`.
При несовпадении хешей двух версий источников сравнение не считается
парным. Если Gemma-сводки нет — результат Qwen остаётся доступным
для оценки без выдуманного показателя сравнения.

**Эти 91 материала уже использовались для диагностики Gemma:**
полученные здесь цифры не являются новой независимой оценкой.

## Новый holdout после исправлений

Собери отдельный набор с исключением **обоих** предыдущих этапов,
не используя старые статьи как новый тест:

```powershell
.\.venv\Scripts\python.exe scripts\stage4_independent_holdout.py `
  --exclude collector_test_results\stage4_abstractive_v5_100.json `
  --exclude collector_test_results\stage4_independent_holdout\gemma_holdout.json `
  --out-dir collector_test_results\stage4_qwen_fresh_holdout `
  --mode both --target 100 --minimum 40 --max-age-days 30 --collect-only
```

В манифесте должна быть проверена независимость URL, текстов, событий
и источников; скрипт обеспечивает первоначальное исключение совпадений
URL/заголовков/текстов, но не заменяет ручную проверку идентичности событий.

Для парного сравнения обе модели запускаются на одном **замороженном**
манифесте:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage4_summaries.py `
  --input-json collector_test_results\stage4_qwen_fresh_holdout\holdout_manifest.json `
  --limit 100 --min-clusters 40 `
  --output collector_test_results\stage4_qwen_fresh_holdout\gemma_fresh.json

.\.venv\Scripts\python.exe scripts\compare_stage4_qwen.py `
  --input-json collector_test_results\stage4_qwen_fresh_holdout\holdout_manifest.json `
  --gemma-json collector_test_results\stage4_qwen_fresh_holdout\gemma_fresh.json `
  --limit 100 `
  --output collector_test_results\stage4_qwen_fresh_holdout\qwen_vs_gemma.json
```

Не утверждай качество на выборке менее 40 уникальных событий.
Отдельно требуется тест с **реальными многодокументными кластерами**:
в прежних 91 материалах таких не было.

## Ручная редакционная оценка

В `*_human_review.csv` для каждой сводки оцени (pass/fail/na):
`faithful_facts`, `natural_russian`, `main_event_covered`,
`quotes_represent_claims`, `source_quality`; в `notes` выпиши
проблемный факт и точную цитату. Только на проверенной разметке можно
сравнивать фактическую точность и естественность языка, в том числе
хорошие ошибочно отклонённые сводки. Результат автоматики в метрике
`qwen_mechanical_drafts` — **не точность модели**.

Для анализа отправь `stage4_qwen_vs_gemma_91.json`, одноимённый
`.csv` и заполненный `_human_review.csv` (при возможности также
снимок `ollama ps` во время генерации). Не публикуй отчёты с
полными текстами, если на них действуют ограничения распространения.
