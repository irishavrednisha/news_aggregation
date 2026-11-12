# Этап 2 — PostgreSQL для картинок и Telegram-модерации

Рабочая ветка: feat/moderation-foundation-2026-11-12

**На этом этапе НЕ создаётся Telegram-бот и НЕ запускается сбор картинок.**
Добавляются модели и 4 новые таблицы. Существующие таблицы не меняются.
Включать MAX_PUBLISHING_ENABLED=true пока нельзя.

## Новые таблицы

- news_images — до одного главного изображения на новость, путь к локальному файлу, URL, размер, права и авторство.
- moderation_items — очередь модерации для зафиксированной версии публикации, Telegram ID, решение о правах.
- moderation_actions — история всех действий модератора.
- publication_attempts — попытки отправки и внешние идентификаторы.

Примечание: даже если картинка отсутствует, новость позже можно будет отправить на модерацию без неё.

## Шаг 1. Забрать изменения

Откройте PowerShell в D:\Projects\news_aggregation:

~~~powershell
git status
git pull --ff-only
~~~

Если git status показывает незакоммиченные правки, сначала сохраните их.
Не используйте git reset --hard.

## Шаг 2. Проверить настройки, ничего не меняя

~~~powershell
.\.venv\Scripts\python.exe -c "from config import DB_HOST, DB_PORT, DB_NAME, DB_USER; print('БД:', DB_HOST, DB_PORT, DB_NAME, DB_USER)"
.\.venv\Scripts\python.exe scripts\check_stage2_schema.py
~~~

Сохраните вывод с количеством sources/news/clusters/posts — после миграции он должен остаться прежним.

Примечание для Windows: если при прямом запуске ранее появлялось
`ModuleNotFoundError: No module named 'database'`, выполните
`git pull --ff-only` — путь импорта в скрипте исправлен. Альтернативный
эквивалентный запуск из корня проекта:
`\.\.venv\Scripts\python.exe -m scripts.check_stage2_schema`.

## Шаг 3. Резервная копия PostgreSQL — обязательно!

Значения в примере должны соответствовать локальному .env. Пароль вводите в командной строке pg_dump только через штатный запрос, не в параметрах.

~~~powershell
$backupDir = Join-Path $env:USERPROFILE "news_aggregation_private_backups"
New-Item -ItemType Directory -Force $backupDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$pgDump = "C:\Program Files\PostgreSQL\17\bin\pg_dump.exe"
$pgRestore = "C:\Program Files\PostgreSQL\17\bin\pg_restore.exe"
$backup = Join-Path $backupDir "db_$stamp.dump"
& $pgDump -h localhost -p 5432 -U postgres -d news_aggregation -Fc -f $backup
if ($LASTEXITCODE -ne 0) { throw "Ошибка создания резервной копии" }
if (-not (Test-Path $backup)) { throw "Файл копии отсутствует" }
if ((Get-Item $backup).Length -le 0) { throw "Файл копии пустой" }
& $pgRestore -l $backup | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Не удалось прочитать резервную копию" }
Write-Host "Резервная копия проверена: $backup"
~~~

ВНИМАНИЕ: подставьте значения DB_HOST, DB_PORT, DB_USER и DB_NAME из шага 2, если они отличаются. Не кладите копию БД и .env в GitHub. Проверка списка архива не заменяет полноценный пробный restore, но выявляет многие ошибки файла.

## Шаг 4. Установить Alembic и проверить тесты

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-moderation-stage2.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_moderation_schema.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_moderation_gate.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_publisher_safety.py" -v
~~~

Все тесты должны быть OK. Они используют временную SQLite-базу и не обращаются к вашей PostgreSQL.

## Шаг 5. Применить миграцию ТОЛЬКО после успешной копии и тестов

Проверьте MAX_PUBLISHING_ENABLED=false в локальном .env.

~~~powershell
.\.venv\Scripts\python.exe -m alembic current
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic current
.\.venv\Scripts\python.exe scripts\check_stage2_schema.py
~~~

Ожидается версия 20261112_01 и новые таблицы со статусом OK.
Количество старых записей должно остаться прежним.

Если миграция сообщает, что таблиц нет, не делайте stamp и не создавайте
таблицы вручную: проверьте подключение к базе. Если Alembic обнаружит
ранее существующий конфликтующий revision или новые таблицы, остановитесь
и сохраните сообщение об ошибке.

Автоматический downgrade заблокирован, чтобы случайно не удалить новую
историю модерации.

## Что дальше

1. Подключить сохранение одной главной картинки из каждой статьи.
2. Подключить очередь Telegram-модерации к реальному боту.
3. Проверить контроль прав на изображение и подтверждение одобрения.
4. Только после этого отдельно тестировать публикацию в MAX.

Важно: текущий publisher всё ещё проверяет Post.status=approved, а не
новую таблицу moderation_items. Поэтому флаг MAX_PUBLISHING_ENABLED
оставляем false до подключения проверяемого одобрения в следующем этапе.
