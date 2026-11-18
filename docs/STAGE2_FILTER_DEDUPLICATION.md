# Этап 2 из 7 — очистка, tri-state фильтр и точные дубликаты

Ветка: `feat/stage2-filter-exact-dedupe-2026-11-18`.
Родитель: `fix/collector-isolation-tls-2026-11-18` @ `8574fb3`.
`master` не изменялся. MAX и Telegram не запускались.

## Разделение конвейера

1. **Raw collection:** `collectors.data_collector.collect_news(..., apply_filters=False, session=None)` собирает исходные статьи RSS/HTML без смысловых стоп-слов и без отказа по географии. Совсем пустые страницы/пустые HTML-заглушки отбрасываются самим сетевым сборщиком; они присутствуют в его диагностике, а не в таблице исходных текстов.
2. **Technical triage:** `processing.news_intake.technical_decision` проверяет заголовок, URL, сервисные страницы, boilerplate, длину, источник текста и даты. Пустой/короткий текст, RSS preview, слишком старые и будущие даты получают `review`, а не пропадают.
3. **Semantic triage:** `processing.news_intake.semantic_decision` отклоняет очевидную рекламу, поздравления, вакансии, дайджесты, ЕГЭ, уголовные и потребительские сюжеты без промышленного события. Промышленное событие принимается, неоднозначное — `review`. **Строгого RU/foreign фильтра нет.** Старый `processing/topic_filter.py` сохранён для совместимости предыдущих аудитов и тестов, production использует новую tri-state функцию.
4. **Exact dedupe:** `processing.exact_duplicates` удаляет трекинг URL, нормализует заголовок, проверяет идентичный текст и совпадение зеркального адреса. Параметры вроде `?obj=...` у НедраДВ НЕ удаляются. Один и тот же общий заголовок разных событий **не** является достаточным основанием для слияния.
5. **Persistence:** `database.intake_repository.save_intake_batch` добавляет все извлечённые статьи в `news_intake_records`. Только `accepted` и не дубликат попадает в прежнюю `news`. `review/rejected/duplicate` остаются в сыром журнале, включая URL, полный текст, источник, дату, причину и ссылку на оригинальный News ID. Повторный сбор идентичной записи увеличивает `seen_count`, новых News не создаёт.
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
  {"url":"https://example.org/news/real-article-3","expected_status":"duplicate"}
]}
```

Затем добавьте `--labels collector_test_results\stage2_manual_labels.json`. В отчёте появятся `false_junk_pass`, `false_useful_reject`, `useful_awaits_review`, `duplicate_tp/fp/fn`, список реальных ошибок и воспроизводимость при повторном прогоне. Если оценок нет, `gold_real_metrics=null`, а не сфабрикованный ноль ошибок.

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
