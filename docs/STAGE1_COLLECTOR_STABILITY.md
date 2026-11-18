# Этап 1 из 7 — независимый и диагностируемый сбор

Рабочая ветка: `fix/collector-isolation-tls-2026-11-18`
Основа: `feat/moderation-foundation-2026-11-12`, коммит `8ec87d8`.

## Контракт процессов

Пять отдельных ОС-процессов запускаются независимо одной точкой входа:

| Процесс | Команда | Зависимости и обмен |
| --- | --- | --- |
| Сбор и сохранение | `python main.py collect --limit 3` | Windows; без scikit-learn/torch, пишет только проверенные новости в существующую таблицу `news` |
| Модель и кластеризация | `python main.py process --hours 48` | Запускается отдельно в WSL с ML-библиотеками; читает `news`, пишет `clusters` и результаты обработки |
| Подготовка черновиков | `python main.py prepare --limit 30` | Требует существующую миграцию `20261112_01`, создаёт `posts` и `moderation_items`, не отправляет сообщения |
| Telegram-модерация | `python main.py telegram` | Отдельный Telegram polling; модератор принимает решения, MAX не вызывается |
| Публикация MAX | `python main.py max` | Принудительно заблокирована для этапа 1. `MAX_PUBLISHING_ENABLED=false` обязателен |

`python main.py` по умолчанию запускает **только сбор**. `run_scheduler.bat` запускает только регулярный сбор каждые 30 минут. Никакого импорта ML на этом пути нет. Сборщик не вызывает `create_tables` или `seed_sources` и не меняет настройки 25 источников в PostgreSQL.

Существующие статусы очереди:

- `news.cluster_id IS NULL`: новость ждёт обработки моделью.
- `clusters.status`: `new`, `ready`, `not_for_publication`; `summary_text` и `tags` дополняют состояние.
- `posts.status`: `draft`, `approved`, `sending`, `uncertain` и другие существующие переходы.
- `moderation_items.status`: `pending` и результат решения модератора.

Новой таблицы заданий не создаём: статус **ошибки конкретного сайта** хранится в локальном JSON отчёте, а не в БД. Отдельного persistent retry/dead-letter queue пока нет: это ограничение данного этапа. На уровне уникального `news.url` повторный сбор идемпотентен; отдельная статья с конфликтом БД не останавливает запись следующих.

## PowerShell: безопасное обновление

```powershell
cd D:\Projects\news_aggregation
git status --short
# Если есть несохранённые изменения — сохраните их до переключения ветки.
git fetch origin
git switch -c fix/collector-isolation-tls-2026-11-18 --track origin/fix/collector-isolation-tls-2026-11-18
# Если ветка уже есть: git switch fix/collector-isolation-tls-2026-11-18
git pull --ff-only

$env:MAX_PUBLISHING_ENABLED = "false"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_stage1_stability.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_collector_regressions.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_text_extraction_all_sources.py" -v

# Показывает реальные состояния таблиц, ничего не изменяет:
.\.venv\Scripts\python.exe main.py status

# ОДНА команда проверки ВСЕХ записей sources (включая неактивные):
.\.venv\Scripts\python.exe scripts\diagnose_all_sources.py --limit 2

# После чтения отчёта: реальный сбор и запись News, без ML:
.\.venv\Scripts\python.exe main.py collect --limit 2
```

Не запускайте `pip install -r requirments.txt` для Windows-сборщика: там есть scikit-learn и torch. Для уже рабочего окружения переустановка не требуется. В новом окружении есть `requirements-collector.txt`; если Code Integrity блокирует SQLAlchemy wheel, сохраните уже рабочую pure-Python установку. Активация PowerShell не требуется: используйте прямой путь `.venv\Scripts\python.exe`.

JSON отчёт: `collector_test_results/all_sources_diagnostics.json`. В нём для **каждой строки базы** есть ID, название, `is_active`, статус, число найденных кандидатов, извлечённых статей, полнота HTML-текста (эвристика 250 символов), заголовки, даты, количество кандидатов изображения, причины ошибок и пробы отвергнутых страниц. Статьи с ошибкой сервера/пустой оболочкой не считаются успешно собранными. **HTML длиной 250+ символов сам по себе не доказывает полноту**: отдельные примеры нужно проверить вручную.

```powershell
# Только сайт Ростеха, без записи в БД:
.\.venv\Scripts\python.exe scripts\diagnose_all_sources.py --source "Ростех" --limit 2
# Разрешить диагностический fallback лишь для заранее перечисленных
# официальных доменов и только после подтверждённой TLS-ошибки:
.\.venv\Scripts\python.exe scripts\diagnose_all_sources.py --source "Ростех" --allow-unverified-tls --limit 2
```

Диагностика с bypass помечает `tls_unverified_urls`. Её нельзя превращать в production-сбор или публикацию без ручной проверки. Отчёт не содержит скачанных фотографий: `image_candidates` означает только обнаруженные на странице ссылки. Оригинальный пайплайн получения и сохранения фото сохранён.

## TLS: без глобального отключения

1. Обычный Requests с `verify=True`.
2. При `SSLError` — попытка с системным хранилищем через опциональный `truststore`, без `inject_into_ssl()`.
3. При неуспехе — PEM CA bundle, назначенный **точному имени хоста** через `NEWS_SOURCE_CA_BUNDLES`. Старый `NEWS_CA_BUNDLE` ограничен доменами Газпрома, Ростеха и Роснефти.
4. Проверенный Chromium; у Роснефти сохраняется уже существующий официальный mirror `limited.rosneft.ru`.
5. Лишь при явном диагностическом opt-in и фактически обнаруженной ошибке сертификата — Chromium с ограниченным списком хостов: `rostec.ru`, `www.rostec.ru`, `rosneft.ru`, `www.rosneft.ru`, `limited.rosneft.ru`. Не допускаются URL с учётными данными и перенаправления на чужой хост.

Пример локального `.env` (не коммитить):

```dotenv
MAX_PUBLISHING_ENABLED=false
NEWS_IMAGES_ENABLED=false
NEWS_SOURCE_CA_BUNDLES={"www.rostec.ru":"C:/trusted-ca/rostec.pem","www.rosneft.ru":"C:/trusted-ca/rosneft.pem"}
```

Файлы CA должны быть взяты только из проверенного доверенного источника. Применение пользовательских CA не делает сайт достоверным автоматически. Для Газпрома альтернативный официальный RSS не выдумываем — если нет верифицированной ссылки, фиксируем недоступность.

## WSL: ML-этап отдельно

WSL должен использовать собственное Linux-окружение Python с рабочими ML пакетами и доступ к **той же** PostgreSQL. На WSL путь к репозиторию обычно `/mnt/d/Projects/news_aggregation`.

```bash
cd /mnt/d/Projects/news_aggregation
# Активируйте уже существующее рабочее WSL-окружение с PyTorch.
python main.py status
python main.py process --hours 48
```

Не запускайте Windows и WSL model workers одновременно. При соединении с PostgreSQL из WSL может потребоваться отдельное `DB_HOST` и настройка доступа в `pg_hba.conf` (не меняйте их без проверки текущей конфигурации). В этом этапе миграции и изменения схемы не выполнялись.

## Что означает прохождение CI

GitHub Actions без ML проверяет синтаксис, импорт главной точки входа, изоляцию ошибок, HTTPS-контракт, локальные фикстуры парсеров. Он **не** читает вашу локальную Windows/PostgreSQL и **не** подтверждает доступность 25 внешних сайтов из вашей сети. Для завершения эксплуатационной проверки обязателен JSON локального прогона; источники, не давшие полноценного текста, отмечайте как непроверенные или проблемные, а не как исправленные.
