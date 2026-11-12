# Этап 1 — безопасная подготовка Telegram-модерации

Исходная ветка: fix/grounded-tagging-only-2026-11-09
Рабочая ветка: feat/moderation-foundation-2026-11-12

## Что изменено

- main.py больше не вызывает публикацию в MAX.
- publisher.py использует только Post со статусом approved; Cluster.status=ready не разрешает отправку.
- Отправка отключена по умолчанию через MAX_PUBLISHING_ENABLED=false.
- При будущей отправке используется сохранённый Post.post_text (формат: заголовок, пустая строка, полный текст), а не пересобранная сводка.
- Post атомарно резервируется: approved -> sending. После попытки статус становится uncertain, пока нет надёжного подтверждения от MAX. Автоматический повтор неизвестного результата запрещён.
- --dry-run показывает одобренные записи без браузера и без изменения БД.

Telegram-бот, история решений модераторов, изображения и миграции относятся к следующим этапам. На этом этапе сервис, создающий approved Post, пока не реализован. Не создавайте такие записи вручную и не включайте отправку.

## Резервная копия локальной PostgreSQL (до запуска)

Сделайте это на своём компьютере Windows, не в GitHub. PowerShell в каталоге проекта:

~~~powershell
cd D:\Projects\news_aggregation
git status
$backupDir = Join-Path $env:USERPROFILE "news_aggregation_private_backups"
New-Item -ItemType Directory -Force $backupDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$pgDump = "C:\Program Files\PostgreSQL\17\bin\pg_dump.exe"
& $pgDump -h localhost -p 5432 -U postgres -d news_aggregation -Fc -f (Join-Path $backupDir "db_$stamp.dump")
if ($LASTEXITCODE -ne 0) { throw "pg_dump failed" }
if (Test-Path .env) { Copy-Item .env (Join-Path $backupDir "env_$stamp.txt") }
~~~

Проверьте DB_NAME, DB_USER и DB_PORT в своём локальном .env; замените значения команды, если они отличаются. pg_dump запросит пароль. Резервные копии и копия .env конфиденциальны, храните их только в закрытой локальной папке.

## Переход в рабочую ветку

Если git status показывает незакоммиченные изменения, сначала сохраните их, чтобы не потерять работу.

~~~powershell
git fetch origin
git switch --track origin/feat/moderation-foundation-2026-11-12
~~~

Если ветка уже существует локально: git switch feat/moderation-foundation-2026-11-12.

## Настройка и тестирование

В локальном .env явно установите:

~~~dotenv
MAX_PUBLISHING_ENABLED=false
~~~

Тесты (без внешней сети и MAX):

~~~powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_moderation_gate.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_publisher_safety.py" -v
~~~

Безопасный просмотр очереди одобренных записей (без отправки и изменений БД):

~~~powershell
.\.venv\Scripts\python.exe -m publishing.publisher --dry-run
~~~

После резервной копии — обычный сбор и обработка:

~~~powershell
.\.venv\Scripts\python.exe main.py
~~~

main.py изменяет базу (сбор, кластеризация, сводки, теги), но больше НЕ отправляет в MAX. Это не read-only режим.

## Осталось для следующих этапов

- Telegram-одобрение с проверкой Telegram ID и журналом действий.
- Миграции и версии одобренных публикаций.
- Выбор, загрузка, хранение и правовая проверка одного изображения.
- Надёжное подтверждение доставки в MAX; до появления подтверждения uncertain не переводить автоматически в published.
- Независимая очередь модерации и отправки.

Не включайте MAX_PUBLISHING_ENABLED=true до реализации и проверки Telegram-модерации.
