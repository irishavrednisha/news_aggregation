# Этап 2 из 7 — очистка, фильтр accepted/rejected и точные дубликаты

Ветка: `feat/stage2-filter-exact-dedupe-2026-11-18`.
Родитель: `fix/collector-isolation-tls-2026-11-18` @ `8574fb3`.
`master` не изменялся. MAX и Telegram не запускались.

## Разделение конвейера

1. **Raw collection:** `collectors.data_collector.collect_news(..., apply_filters=False, session=None)` собирает исходные статьи RSS/HTML без смысловых стоп-слов и без отказа по географии. Совсем пустые страницы/пустые HTML-заглушки отбрасываются самим сетевым сборщиком; они присутствуют в его диагностике, а не в таблице исходных текстов.
2. **Technical triage:** `processing.news_intake.technical_decision` проверяет заголовок, URL, служебные страницы, boilerplate, длину, источник и даты. Пустые/короткие тексты, RSS-превью, некорректные будущие даты — `rejected` с объяснением, но raw-снимки сохраняются. Старые полноценные статьи — `accepted` с причиной `stale_article` (свежесть решается отдельно).
3. **Semantic triage:** `processing.news_intake.semantic_decision` отклоняет явную рекламу, поздравления, вакансии, анонсы, дайджесты, уголовные и непрофильные сюжеты; промышленные события, проекты и содержательную отраслевую аналитику принимает. Для непрозрачных заголовков проверяется начало статьи, а не применяется безусловное отклонение. **Строгого RU/foreign фильтра нет.** Production возвращает только `accepted`/`rejected`; старый `processing/topic_filter.py` сохранён как legacy API.
4. **Exact dedupe:** `processing.exact_duplicates` удаляет трекинг URL, нормализует заголовок, проверяет идентичный текст и совпадение зеркального адреса. Параметры вроде `?obj=...` у НедраДВ НЕ удаляются. Один и тот же общий заголовок разных событий **не** является достаточным основанием для слияния.
5. **Persistence:** `database.intake_repository.save_intake_batch` сохраняет все извлечённые raw-статьи в `news_intake_records`. В `news` попадает только `accepted` без точной копии. Новый raw-журнал использует только `accepted` и `rejected`; копии имеют `rejected` + `duplicate_url/duplicate_text/...`, ссылки `duplicate_of_news_id` или `duplicate_of_intake_id`, исходные URL, тексты, даты и `seen_count`. Исторические `review`/`duplicate` в БД не переписываются ради сохранения аудита.
6. **Semantic clustering:** по-прежнему отдельный процесс `python main.py process`, читает только `news`. В ходе этапа 2 **не запускался**.

## Основания для решений

Семантические: `industrial_event`, `advertisement`, `congratulation`, `vacancy`, `digest`, `offtopic_event`, `consumer_only`, `industry_without_clear_event`, `uncertain_industry`.

Технические: `missing_title`, `invalid_url`, `navigation_page`, `empty_text`, `short_text`, `site_boilerplate`, `rss_preview_only`, `future_date`, `stale_article`.

Копии: `duplicate_url`, `duplicate_text`, `duplicate_mirror`, `duplicate_title_content`.

## Миграция: только добавление

`migrations/versions/20261118_02_news_intake.py`, зависит от `20261112_01`.
Добавляет **одну** таблицу `news_intake_records` и индексы. Не меняет `sources`, `news`, `clusters`, `posts` или ранние данные. Изображения продолжают обслуживаться старой таблицей `news_images` при `NEWS_IMAGES_ENABLED=true` (по умолчанию выключено).

Не пытайтесь запускать сбор на новой ветке до применения миграции — он **специально прекращает работу** и не переходит в старый теряющий записи режим. Сначала сделайте резервную копию PostgreSQL и проверку числа старых записей.

## PowerShell: локальное развёртывание и верификация

```powershell
cd D:\Projects\news_aggregation
git status --short
# Сохраните локальные изменения, если есть. Не переключайте грязное дерево.
git fetch origin
git switch -c feat/stage2-filter-exact-dedupe-2026-11-18 --track origin/feat/stage2-filter-exact-dedupe-2026-11-18
# Если ветка уже существует: git switch feat/stage2-filter-exact-dedupe-2026-11-18
git pull --ff-only

$env:MAX_PUBLISHING_ENABLED = "false"
$env:NEWS_IMAGES_ENABLED = "false"

.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_stage2_intake.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_stage1_stability.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_topic_filter.py" -v

# До миграции: только чтение вашей локальной БД, сохранить старые счётчики.
.\.venv\Scripts\python.exe scripts\check_stage2_intake_schema.py --allow-missing --output collector_test_results\stage2_before.json

# Резервная копия PostgreSQL 17: измените имя БД/пользователя по своему .env.
& "C:\Program Files\PostgreSQL\17\bin\pg_dump.exe" -U postgres -d news_aggregation -Fc -f "D:\Projects\news_aggregation\stage2_backup.dump"
Test-Path "D:\Projects\news_aggregation\stage2_backup.dump"

# Обязательный Alembic в уже существующем рабочем окружении:
.\.venv\Scripts\python.exe -m alembic current
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic current

# Немедленно после миграции, ДО любого нового сбора:
.\.venv\Scripts\python.exe scripts\check_stage2_intake_schema.py --baseline collector_test_results\stage2_before.json --output collector_test_results\stage2_after.json
```

Если у вас уже есть локальная ветка с таким именем, используйте `git switch` вместо `git switch -c`. Если отсутствует Alembic: `.\.venv\Scripts\python.exe -m pip install alembic`. Не устанавливайте тяжёлые ML-пакеты для этапа фильтрации.

## Аудит реальных новостей

```powershell
# Только чтение активных sources вашей PostgreSQL; не сохраняет статьи в неё.
.\.venv\Scripts\python.exe scripts\audit_stage2.py --live --from-db --target 250 --per-source 20 --require-coverage --output collector_test_results\stage2_real_250.json

# Альтернатива — использовать уже собранные в отдельном запуске raw-статьи.
.\.venv\Scripts\python.exe scripts\audit_stage2.py --input collector_test_results\independent_raw_news.json --target 250 --output collector_test_results\stage2_from_file.json
```

Выборка набирается из разных источников по кругу, а не из одного новостного сайта. `--require-coverage` возвращает ненулевой код, если извлечено меньше 200 доступных реальных текстов. Статус `independence_from_prior_samples` остаётся `not_verified`, пока не указан список ранее использованных URL через `--exclude-urls`; скрипт **не выдумывает** гарантию независимости данных.

В `tests/fixtures/stage2_gold_cases.json` — **50 вручную заданных, синтетических** граничных примеров с известными `expected_status` и `expected_reason`. Они не подменяют независимую реальную оценку. Реальные статьи автоматически не размечаются своей собственной моделью как gold standard.

Для подсчёта ошибок на **реальных** публикациях вручную подготовьте JSON:

```json
{"labels":[
  {"url":"https://example.org/news/real-article-1","expected_status":"accepted"},
  {"url":"https://example.org/news/real-article-2","expected_status":"rejected"},
  {"url":"https://example.org/news/real-article-3","expected_status":"rejected","expected_reason":"duplicate_text"}
]}
```

Затем добавьте `--labels collector_test_results\stage2_manual_labels.json`. В отчёте появятся `false_junk_pass`, `false_useful_reject`, `duplicate_tp/fp/fn`, список реальных ошибок и воспроизводимость при повторном прогоне. Если оценок нет, `gold_real_metrics=null`, а не сфабрикованный ноль ошибок.

После проверки миграции и принятия результатов аудита можно запускать **только сбор**:

```powershell
.\.venv\Scripts\python.exe main.py collect --limit 10
.\.venv\Scripts\python.exe main.py status
```

MAX и Telegram не запускать.

## Ограничения и критерий завершения

- Положительные/отрицательные ключевые слова — **консервативный эвристический фильтр**, а не обученная модель. `review` остаётся для ручного разбора.
- Вся историческая таблица `news` просматривается для совместимости со старой базой без fingerprints; на сотнях тысяч строк потребуется отдельная оптимизированная миграция/индекс.
- Сборщик всё ещё пропускает пустые/ошибочные HTML-ответы до сохранения raw snapshots; эти URL доступны только в диагностике сбора. `NEWS_MAX_AGE_DAYS` в старом коде действует независимо; для stage2 выбран безопасный `review` вместо удаления.
- Сбой уникального индекса из-за одновременного параллельного записи требует отдельной операции восстановления; до её реализации не запускайте два сборщика одновременно.
- GitHub Actions проверяют исходный код на Windows и Linux, но **не** содержат вашу локальную PostgreSQL и не подтверждают миграцию реальных пользовательских данных.
- Сам факт получения 200–300 статей не доказывает качества фильтрации. Этап нельзя признать законченным, пока не проверены полнота реальной выборки, независимая ручная разметка и сохранность локальной БД.

## Коррекция правил по аудиту реальных статей 18.11.2026

Первый независимый сетевой сбор дал 199 статей, из которых 10 были `accepted`, 187 — `review`, 2 — `rejected`. Это **не** показатель доли полезных публикаций: фильтр не распознавал обычные слова «комбинат ввели», «Росэл увеличил выпуск», «ОСК спустила», «ТМК изготовила». Пополнены глагольные и отраслевые признаки, но ограничены корпоративные PR/HR-события. Пересмотрен порядок проверки возраста: очевидная реклама/поздравление отклоняется независимо от даты; устаревшая промышленная новость всё равно идёт в `review`, а не в автоматическую публикацию. Отдельные проверки 29 заголовков взяты из реального отчёта, без объявления их статистически независимой разметкой.

Можно переоценить **тот же файл** без повторного обращения к сайтам и без записи в БД:

```powershell
.\.venv\Scripts\python.exe scripts\audit_stage2.py --input collector_test_results\stage2_real_250.json --date 2026-11-18 --target 250 --output collector_test_results\stage2_replayed.json
```

В `comparison_with_previous_model.transitions` будут старые и новые статусы; `changed` содержит конкретные заголовки и причины. Переход из `review` в `accepted` сам по себе не доказывает качество: источником истины остаётся независимая ручная оценка. Статус `review` сохраняется для действительно спорных случаев и устаревших публикаций.


## Проверка полного корпуса 231 статьи (18.11.2026)

По фактически загруженному `stage2_latest.json` (231 материал, 17 источников) первоначальный аудит дал `45 accepted / 168 review / 10 rejected / 8 duplicate`. Сравнение **полных нормализованных текстов всех статусов**, а не только `accepted`, выявило 13 групп одинаковых текстов и **21 избыточную копию**: 8 уже обнаруженных + 11 прежних `review` + 2 прежних `rejected`. Наиболее заметны четыре копии «ОСК усовершенствовала систему подготовки наставников», четыре копии о «Пролетарском заводе» и три копии выбора ракет ATMACA. Нельзя называть 8/21 полнотой промышленной дедупликации: это наблюдение одного корпуса, а не независимая gold-разметка.

**Исправление:** `audit_stage2.py` теперь обнаруживает точные копии в сыром корпусе независимо от стадии смысловой фильтрации, но не отбрасывает оригинальные записи. `intake_repository.py` хранит путь к оригиналу `review/rejected` в `duplicate_of_intake_id`, а к принятой `news` в `duplicate_of_news_id`. Событие со статусом `review` никогда не блокирует позднее поступившую действительно принятую новость. Raw-журнал сохраняет тексты, причины и все зеркала; повторные снимки учитываются `seen_count`. Окончательные цифры проверять только на CI-артефакте **текущего SHA**.

**Новая обязательная добавочная миграция:** `20261118_03_raw_duplicate_links.py` поверх `20261118_02`. Добавляет nullable FK `news_intake_records.duplicate_of_intake_id` и индекс, не меняет `news/clusters/posts/sources`. Без неё сбор специально останавливается до записи. Ни один CI-тест с SQLite и Github Actions **не заменяет** проверку локальной PostgreSQL 17 или самостоятельную резервную копию.

### PowerShell перед любым сбором на новой ветке

```powershell
cd D:\Projects\news_aggregation
git status --short
git fetch origin
git switch feat/stage2-filter-exact-dedupe-2026-11-18
git pull --ff-only

$env:MAX_PUBLISHING_ENABLED = "false"
$env:NEWS_IMAGES_ENABLED = "false"

.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_stage2_review_duplicates.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_stage2_followup.py" -v

# Только чтение БД до миграции, проверьте имя пользователя и БД по своему .env:
.\.venv\Scripts\python.exe -m alembic current
& "C:\Program Files\PostgreSQL\17\bin\pg_dump.exe" -U postgres -d news_aggregation -Fc -f "D:\Projects\news_aggregation\stage2_pre03_backup.dump"
Test-Path "D:\Projects\news_aggregation\stage2_pre03_backup.dump"

# Только после проверенной резервной копии:
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic current
.\.venv\Scripts\python.exe scripts\check_stage2_intake_schema.py --output collector_test_results\stage2_schema_after03.json

# Воспроизводимый read-only повторный аудит, без БД, MAX и Telegram:
.\.venv\Scripts\python.exe scripts\audit_stage2.py --input collector_test_results\stage2_live_250.json --date 2026-11-18 --target 250 --require-coverage --output collector_test_results\stage2_231_after_dedupe.json
```

Даже при успешной CI-проверке этап **не закрывается** до независимой разметки ошибок смысловой фильтрации, проверки локального PostgreSQL и исключения совпадений с ранее использованными обучающими URL.


## Обновление: два статуса вместо review (18.11.2026)

По просьбе владельца проекта **новые** решения фильтра и audit JSON используют **только** `accepted` / `rejected`. Отдельного `review` или `duplicate` как статуса нет. Смысл копий сохраняется в `reason=duplicate_*`, `stage=exact_duplicate`, `duplicate_of_url`; отчёт содержит `counts.exact_duplicates`. В БД сохраняются обе ссылки на оригинальные статьи и все raw-снимки. `news` пополняется только уникальными `accepted`, а в публикацию на этом этапе ничего не отправляется.

- Пустые/короткие и RSS-preview материалы отклоняются как технически неготовые, но не удаляются из raw-журнала.
- Очевидный мусор/HR/рекламные и непрофильные истории отклоняются с конкретной причиной.
- Промышленные статьи, проекты, аналитика и сомнительные по одному заголовку материалы с промышленными фактами в начале текста принимаются.
- Устаревшие содержательные статьи **не уничтожаются**: остаются `accepted` с причиной `stale_article` для отдельной проверки свежести перед публикацией.
- `review` и `duplicate` могут встречаться только в **исторических** старых отчётах/строках БД, которые не переписывались, и в полях сравнения `before_status`. Новое `status` содержит строго два значения.
- SQL-схема статуса — строковая; ради перехода к двум состояниям **новую миграцию не добавляем**. Однако обязательная добавочная миграция `20261118_03` для `duplicate_of_intake_id` по-прежнему нужна перед новым сбором.
- После обновления ветки повторный аудит выполняется без доступа к БД:
```powershell
.\.venv\Scripts\python.exe scripts\audit_stage2.py --input collector_test_results\stage2_live_250.json --date 2026-11-18 --target 250 --require-coverage --output collector_test_results\stage2_binary.json
```
Результаты первоначальных 231 записей и проценты `review`, приведённые в разделах выше, являются **историческими** — не интерпретируйте их как новый результат.
