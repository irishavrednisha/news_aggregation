# Этап 3/7: смысловая кластеризация событий

Ветка: feat/stage3-event-clustering-2026-11-18.
Основа: f673f89ef75c9a783f1d37a050a0ca43d4c77131 — последний зелёный коммит этапа 2 на Windows и Ubuntu. Четыре последующих коммита этапа 2 не включены: в их CI падает тест фильтрации новости про аэропорты США.

## Причины ошибок старого алгоритма

В main.py фактически использовался порог SBERT 0.7. Исходный алгоритм сравнивал эмбеддинг новости со средним эмбеддингом кластера и считал ближайшую группу одним событием. Так могут объединиться разные партии, события одного предприятия, отдельные месторождения и общие отраслевые новости. Он выбирал самую раннюю статью основной; несколько commit на операцию грозили неполным сохранением.

## Что сделано

- SBERT по удвоенному заголовку и 500 символам вступления только ищет кандидатов; модель импортируется лениво, и сборщик не зависит от DLL sklearn.
- processing/event_matching.py консервативно сравнивает предприятие, действие, объект, модели, числа с единицами, явно указанные даты, разницу дат публикаций. Совпадающие технические параметры подкрепляют решения только вместе с компанией, действием, объектом. По умолчанию candidate_threshold=0.67, threshold=0.79. Это гиперпараметры, а не универсальные гарантии.
- Проверка complete-link: новый материал должен быть согласован с КАЖДЫМ членом кластера. Не допускается цепочка взаимно несовместимых событий.
- Представитель выбирается по информативности. Для подготовленных/утверждённых групп не изменяются заголовок и основная новость; утверждённый Post.post_text никогда не трогается.
- Несколько URL разных источников сохраняются через news.cluster_id, News.url и исторические raw records.
- Одна транзакция на кластер+членство+эмбеддинг. В PostgreSQL advisory-lock предотвращает параллельное создание конфликтующих кластеров. Повторный запуск пропускает уже кластеризованные News.
- Новая команда main.py cluster запускает ТОЛЬКО кластеризацию. main.py process по умолчанию блокируется, чтобы не вызывать суммаризацию и тэггирование.
- Миграция БД для этапа 3 не требуется; этап 2 и её миграции должны быть предварительно проверены.

## Корпус и качество

Зелёный CI-артефакт этапа 2, GitHub Actions run 44283065624:
231 реальная статья из 17 источников, 156 accepted и 75 rejected, включая 21 точную копию. Семантическую кластеризацию оцениваем на 156 принятых уникальных статьях. По тексту исходных публикаций никаких изменений не вносится.

tests/fixtures/stage3_real_pairs_100_provisional.json содержит 100 предварительно просмотренных реальных пар (2 предполагаемых совпадения и 98 различий), привязанных к SHA-256 входного JSON. Это НЕ независимо проверенный gold standard; метрики должны иметь статус provisional. В отчёте также 156 решений распределения по кластерам. Ошибочные слияния важнее пропущенных.

Старый алгоритм сравнивается честно на том же корпусе: полная статья+заголовок, средний эмбеддинг и порог 0.7. Новый: заголовок+lead, SBERT кандидаты, фактическая проверка и complete-link. Файл audit_stage3.py никогда не подключается к PostgreSQL или MAX.

## Команды (PowerShell, без записи в БД)

```powershell
cd D:\Projects\news_aggregation
git status --short
git fetch origin
git switch --track origin/feat/stage3-event-clustering-2026-11-18
# Если локальная ветка уже есть: git switch feat/stage3-event-clustering-2026-11-18
git pull --ff-only

$env:MAX_PUBLISHING_ENABLED="false"
$env:NEWS_IMAGES_ENABLED="false"

.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_stage3_clustering.py" -v

gh run download 44283065624 --repo irishavrednisha/news_aggregation --name stage2-live-231-replay-ubuntu-latest --dir collector_test_results

# Только дымовой тест алгоритма, НЕ настоящие SBERT-оценки
.\.venv\Scripts\python.exe scripts\audit_stage3.py --input collector_test_results\stage2_live_231_latest.json --gold tests\fixtures\stage3_real_pairs_100_provisional.json --backend lexical --min-decisions 100 --output collector_test_results\stage3_lexical.json
```

## WSL: настоящий SBERT в уже работающем Python-окружении

Не переустанавливайте Windows и scikit-learn ради аудита. В существующем WSL окружении, где есть sentence-transformers:

```bash
cd /mnt/d/Projects/news_aggregation
python scripts/audit_stage3.py \
  --input collector_test_results/stage2_live_231_latest.json \
  --gold tests/fixtures/stage3_real_pairs_100_provisional.json \
  --backend sbert --min-decisions 100 \
  --output collector_test_results/stage3_sbert_real.json
```

В JSON-отчёте будут baseline/revised число кластеров, singles, размеры, multi-source, 100 пар с FP/FN/TP/TN, score и reason, исходные URL, fingerprint входа, модель и порог. Если gold не независим — метрики помечаются provisional.

## Только после проверки: намеренное изменение PostgreSQL

```powershell
# Только если PostgreSQL и предыдущие миграции уже проверены, backup сохранён
.\.venv\Scripts\python.exe main.py cluster --hours 48
.\.venv\Scripts\python.exe main.py status
```

Для более ранних ещё не кластеризованных новостей можно осознанно расширить окно (например, --hours 720). Не запускайте main.py process, prepare, telegram, max.

## Перед переходом к этапу 4

CI-тесты Windows+Ubuntu и настоящий SBERT-аудит должны проходить; проверить и исправить фактические ложные объединения, независимо пересмотреть 100 пар, протестировать локальную PostgreSQL и защиту уже утверждённых Post. Количество кластеров и успешные тесты не доказывают точность без независимых меток. Дальнейшую суммаризацию не запускать.
