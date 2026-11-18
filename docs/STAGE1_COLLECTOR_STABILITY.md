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
5. **По запросу оператора от 18 ноября** нормальный `main.py collect` включает Chromium-аналог «всё равно перейти» **только для Ростеха, Роснефти и Газпрома** и только после зафиксированной ошибки сертификата при verified-запросе. Список хостов жёстко ограничен: `rostec.ru`, `www.rostec.ru`, `rosneft.ru`, `www.rosneft.ru`, `limited.rosneft.ru`, `gazprom.ru`, `www.gazprom.ru`. Для браузерного обхода блокируются сторонние запросы, субресурсы и перенаправления; никакого `requests verify=False` нет.
6. Если нельзя принимать риск недостоверной страницы, задайте `NEWS_BROWSER_CERT_WARNING_FALLBACK=0`. При таком режиме недоверенные сертификаты остаются ошибкой, как раньше.

Пример локального `.env` (не коммитить):

```dotenv
MAX_PUBLISHING_ENABLED=false
NEWS_IMAGES_ENABLED=false
NEWS_SOURCE_CA_BUNDLES={"www.rostec.ru":"C:/trusted-ca/rostec.pem","www.rosneft.ru":"C:/trusted-ca/rosneft.pem"}
```

Файлы CA должны быть взяты только из проверенного доверенного источника. Применение пользовательских CA не делает сайт достоверным автоматически. Для Газпрома альтернативный официальный RSS не выдумываем — если нет верифицированной ссылки, фиксируем недоступность.

## Локальная проверка 18 ноября 2026 года

В присланном пользователем **read-only** JSON из локальной PostgreSQL ровно 25 записей; активны **17**, выключены **8**. У **21** источника найдены HTML-статьи (42 из 42 с датами и заголовками по эвристике); не ответили: **ИАПН (HTTP 403)**, **Ростех / Роснефть / Газпром (недоверенный TLS)**. `Газпром` отдельно имеет `is_active=false`.

Обратите внимание: во время обычного `main.py collect` используются только активные записи. Диагностика `--limit 2` проверяет **все 25**, но **не активирует** их.

После этого аудита:

- Если HTML-лента не дала ссылок, а у источника уже записан официальный `rss_url`, выполняется **одна ограниченная резервная попытка** получить RSS. У ИАПН в БД есть `https://iapn.ru/feed/`. Это не обещает доступность RSS: он тоже может дать HTTP 403.
- Повторная локальная проверка ИАПН 18 ноября 2026 года подтвердила **HTTP 403** и для HTML, и для RSS. Сертификат тут не причина; обход HTTPS не поможет.
- При чтении RSS сначала применяется обычная TLS-проверка, затем `truststore`, затем назначенный ровно данному хосту CA bundle.
- Диагностика явно показывает `rss_fallback_attempted`, `rss_fallback_succeeded`, `age_days`, `fresh_within_30_days` и возраст материалов. Старые статьи не маскируются под свежие.
- Исключены изображения из URL-директорий вроде `/images/logo/ru.png`, которые у «Норникеля» ошибочно считались кандидатом фотографии. Это не мешает загрузке настоящих фотографий.
- Режим `apply_filters=False` **не применяет тематический фильтр** и поэтому может показывать непрофильные и зарубежные новости. В обычном режиме фильтр работает. Краткий HTML-текст в диагностике виден как неполный, но не считается допустимым для production, пока не пройдёт фильтры.

Для диагностической проверки проблемных сайтов, без записи новостей:

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_all_sources.py --source "ИАПН" --limit 2
.\.venv\Scripts\python.exe scripts\diagnose_all_sources.py --source "Ростех" --limit 2
.\.venv\Scripts\python.exe scripts\diagnose_all_sources.py --source "Роснефть" --limit 2
.\.venv\Scripts\python.exe scripts\diagnose_all_sources.py --source "Газпром" --limit 2

# Только после оценки рисков и только для диагностического режима:
.\.venv\Scripts\python.exe scripts\diagnose_all_sources.py --source "Газпром" --allow-unverified-tls --limit 2
```

В диагностическом режиме **ничего не передаётся в БД**. В обычном `main.py collect` Chromium-обход (только для трёх источников) разрешён после ошибки CA: полученные новости проходят стандартные фильтры и могут быть сохранены в `news`, но MAX всё ещё заблокирован. В аудите сохраняются `tls_unverified_urls`, в обычном сборе выводится WARNING. Это осознанное снижение доверия к источнику: HTTPS больше не доказывает подлинность таких страниц, существует риск подмены содержимого. Для полной криптографической проверки используйте официально подтверждённый CA bundle вместо предупреждения браузера. ИАПН остаётся без обхода. Не активируйте источники автоматически и не изменяйте `sources.is_active` без явного решения.

```powershell
# По умолчанию браузерное предупреждение разрешено для трёх источников:
$env:NEWS_BROWSER_CERT_WARNING_FALLBACK = "1"
.\.venv\Scripts\python.exe main.py collect --limit 2

# Для выключения обхода сертификатов во всём рабочем сборе:
$env:NEWS_BROWSER_CERT_WARNING_FALLBACK = "0"

# В read-only диагностике обход включается только указанным флагом:
.\.venv\Scripts\python.exe scripts\diagnose_all_sources.py --source "Ростех" --source "Роснефть" --source "Газпром" --limit 2 --workers 3 --allow-unverified-tls --output collector_test_results/cert_warning_recheck.json
```

`Газпром` остаётся неактивным в таблице `sources` и не участвует в обычном сборе до отдельной активации.

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
