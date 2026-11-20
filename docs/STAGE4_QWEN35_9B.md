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
- Qwen генерирует JSON Schema **только с пятью полями**: `main_event`,
  `is_news_event`, `summary`, `uncertainties`, `requires_review`.
  Это сокращает вывод модели и уменьшает риск незаконченного JSON.
- Дополнительный детерминированный индекс выбирает дословные фрагменты
  исходников для каждого предложения сводки. Он строит `claims`, `source_ids`
  и `evidence_quotes` **после** генерации. Цитаты не дописывает Qwen.
  Совпадение цитаты и близость слов не доказывают верности смысловой связи.
- Проверка требует **точного** присутствия цитат в указанных оригиналах,
  сохранения чисел, единиц, названий и отдельных признаков смысла:
  планы/результат, рост/снижение, отрицание, причина, организация.
- `summary_draft` означает только отсутствие обнаруженных механических
  проблем. **Это не одобрение редактора, не доказательство достоверности
  и не основание для автопубликации.** Все кандидаты требуют ручной проверки.

### Результаты локальной диагностики 20 ноября 2026 года

Устаревшая схема v1: из 91 материала Qwen сгенерировала 58 сводок, а
27 записей получили отказ подключения Ollama; механически принятых
результатов не было. Пересчёт доказательств v2 **без изменения текстов**
дал 18 формально чистых и 40 требующих проверки сводок.
Smoke v2 на трёх статьях: 3 генерации, 1 механически принята,
2 отклонены. Это не независимая оценка фактов.

При проверке оригиналов выявлены **два опасных пропуска валидатора**:
Qwen называла будущие 103 рабочих места уже созданными и пропускала
исключение для Рыбинского шлюза, хотя сводка объявляла прекращение
движения «на всех участках». Оба текста проходили механическую проверку.
В другой сводке материал рукоятки винтовки был ошибочно приписан прикладу.
В то же время «70 т» / «70 тонн» и «1000 м» / «1000 метров» вызывали
ложное отклонение. Эти случаи теперь защищены точечными правилами в
\`processing/qwen_fact_guards.py\` (v2.1), без изменения Gemma.

**Порог качества не достигнут и не измерен:** ни один механический статус
не является ручной проверкой причинно-следственной связи, главного
события, грамматики или полноты.

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

.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_stage4_qwen*.py" -v
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

При отказе подключения к локальной модели скрипт сохраняет готовые строки,
останавливается после первой ошибки и возвращает код 4. Это предотвращает
появление множества одинаковых `WinError 10061`. После восстановления Ollama
нужно повторить ту же команду с `--resume`.

Проверить API Ollama в PowerShell:

```powershell
ollama ps
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

При отказе подключения запусти Ollama Desktop. Если приложение не запускает
API, проверь процесс/журнал Ollama; не начинай новый прогон до восстановления
локального сервера. По одному отказу невозможно определить причину остановки.


## Повторная проверка имеющихся 58 сводок без Ollama

Если уже получен старый отчёт `stage4_qwen_vs_gemma_91.json` (v1),
**не удаляй его**. Новая команда перестраивает доказательства из сохранённых
полных источников, не меняя сводок Qwen и не вызывая модель:

```powershell
.\.venv\Scripts\python.exe scripts\revalidate_stage4_qwen.py `
  --input-json collector_test_results\stage4_qwen_vs_gemma_91.json `
  --output collector_test_results\stage4_qwen_v2_revalidated_91.json
```

Результат содержит `claims_v2`, `validation_v2`, `status_v2`, а также
оригинальные записи. Это **ретроспективная диагностика**, а не новый прогон
Qwen или новый независимый тест. Сводки с неподтверждёнными числами и
смысловыми несоответствиями по-прежнему отправляются на проверку.
Передай файл `stage4_qwen_v2_revalidated_91.json` для анализа.

## Повторная проверка твоих файлов v2 уже с правилами v2.1 (без Ollama)

Сохрани исходные v2-отчёты. Новая команда перепроверит как старые 58
сводок, так и **три новых текста Qwen** в уже имеющихся JSON, не будет
перегенерировать их, изменять базу или перезаписывать исходники:

```powershell
.\.venv\Scripts\python.exe scripts\revalidate_stage4_qwen.py `
  --input-json collector_test_results\stage4_qwen_v2_revalidated_91.json `
  --output collector_test_results\stage4_qwen_v21_revalidated_91.json
```

```powershell
.\.venv\Scripts\python.exe scripts\revalidate_stage4_qwen.py `
  --input-json collector_test_results\stage4_qwen_v2_smoke3.json `
  --output collector_test_results\stage4_qwen_v21_revalidated_smoke3.json
```

Для независимого нового вызова Ollama используй
`--output collector_test_results\stage4_qwen_v21_smoke3.json` и тот же
исходный `gemma_holdout.json`, чтобы не затереть старые результаты.

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

## Диагностика недобора материалов (20 ноября 2026)

Реальный `holdout_manifest(1).json` содержал всего 4 из необходимых
40 промышленных событий: 2 публикации НедраДВ, один анонс ярмарки
вакансий ТПП РФ (нерелевантен) и один потенциально рекламный кейс
видеоаналитики (пограничный). Все 4 были однодокументными.
Причины отклонений в старом порядке: без даты — 105, виденный URL — 96,
старые — 35. **105 — не доказанное число новых публикаций без даты**:
раньше свежесть проверялась до URL и текстового пересечения.

Теперь `scripts/stage4_independent_holdout.py` сначала проверяет историю
использованных URL, текстов и заголовков, затем дату. Он добавляет
`selection_stats.rejections_by_source`, `candidate_source_counts` и
`undated_unseen_examples` (только адрес, заголовок и параметры даты,
без полных текстов). Даты не угадываются. Этого должно хватить для
дальнейшего адресного исправления датирования в сборщиках.

**Не запускай Gemma или Qwen на четырёх статьях для оценки качества.**
Они не дают ни независимой статистики, ни теста многодокументности.
Сначала обнови ветку и выполни одну сборку с новым каталогом:

```powershell
.\.venv\Scripts\python.exe scripts\stage4_independent_holdout.py `
  --exclude collector_test_results\stage4_abstractive_v5_100.json `
  --exclude collector_test_results\stage4_qwen_v21_revalidated_91.json `
  --out-dir collector_test_results\stage4_qwen_fresh_diagnostic_v2 `
  --mode both --target 100 --minimum 40 `
  --max-age-days 30 --limit-per-source 15 --collect-only
```

Если программа вернёт код 2 и `minimum_sample_met=false`, это ожидаемо
при недоборе. Пришли новый `holdout_manifest.json`: прежде всего посмотрим,
какие конкретно источники поставляют статьи без даты.
Не используй `--allow-undated` для итоговой метрики качества.
Не меняй количество неуспешных случаев на выдуманные «новые статьи».
Не запускай `--resume` со старой папкой: он возьмёт старый манифест
без расширенной диагностики.

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

## Дополнительная проверка неизменности PostgreSQL

Включён отдельный интеграционный тест
`tests/test_stage4_qwen_postgres.py`: он вычисляет хеши полных строк
`news`, `clusters`, `posts` до и после прогона без Ollama. Тест
только читает БД и отказывается работать с именем базы, отличным
от `stage4_ci`. В GitHub Actions он пропускается, поскольку
подключения к локальной PostgreSQL нет.

Если **уже подготовлена отдельная тестовая копия БД** `stage4_ci`:

```powershell
$env:STAGE4_TEST_URL = "postgresql+psycopg2://USER:PASSWORD@127.0.0.1:5432/stage4_ci"
$env:STAGE4_TEST_WRITE_ACK = "ephemeral-only"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_stage4_qwen_postgres.py" -v
Remove-Item Env:STAGE4_TEST_URL
Remove-Item Env:STAGE4_TEST_WRITE_ACK
```

Замени USER и PASSWORD данными **только тестовой копии**. Не направляй
тест на рабочую БД и не передавай строку подключения вместе с отчётами.

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
