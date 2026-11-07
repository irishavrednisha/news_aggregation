# Проверка соответствия тегов тексту

**Цель:** максимальная точность тематических тегов без автоматического
назначения тегов на основании случайных упоминаний отрасли.

Все команды ниже только читают локальные JSON/CSV. PostgreSQL и MAX не
используются. Обычный сборщик, фильтр и публикация не запускаются.

## Отдельная синтетическая проверка — без обучения на реальных новостях

В папке `benchmarks/` добавлены два **полностью вымышленных**
файла, созданных специально для проверки:
- `synthetic_tagging_120.json` — 120 искусственно составленных заголовков и
  текстов: 72 профильных (по 4 для каждого из 18 тегов), 24 непрофильных,
  24 неоднозначных;
- `synthetic_tagging_labels_120.csv` — ожидаемые ответы отдельно от текстов.
  Все метки имеют `label_origin=synthetic` и `review_status=draft`.
  События, предприятия и факты в этих примерах вымышлены, их нельзя
  публиковать как новости.

**Модель Sentence-BERT не переобучается и не обновляет веса.**
Эти примеры не загружаются в базу и не смешиваются с 100 реальными
статьями. Синтетика используется как **стресс-тест**, не как объективная
оценка качества на реальных данных. В тексты не включаются ожидаемые теги
или указания, какой ответ верный.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_synthetic_tagging_data.py" -v

.\.venv\Scripts\python.exe check_focused_tagging.py benchmarks/synthetic_tagging_120.json --limit 120 --output collector_test_results/synthetic_tagging_audit_120.json

.\.venv\Scripts\python.exe evaluate_tagging.py collector_test_results/synthetic_tagging_audit_120.json benchmarks/synthetic_tagging_labels_120.csv --include-drafts --output collector_test_results/synthetic_tagging_evaluation_120.json

.\.venv\Scripts\python.exe check_tag_role_challenger.py collector_test_results/synthetic_tagging_audit_120.json --reference benchmarks/synthetic_tagging_labels_120.csv --include-drafts --output collector_test_results/synthetic_role_challenger_120.json
```

Не применять `--optimize` к синтетическому датасету: нельзя выбирать
правила или пороги только потому, что они дали высокую метрику на
сгенерированных примерах. Лучше разбирать ошибки по типам и затем
проверять изменения на **новых** реальных статьях, не участвовавших
в создании правил.

Для свежего набора можно запустить исходный сборщик, затем использовать
`check_focused_tagging.py --exclude-urls-from tagging_benchmark_100.json`,
чтобы исключить URL старой выборки. Если сборщик возвращает тексты
с повторяющимися сюжетами под разными URL, исключить такие дубликаты
при ручной проверке отдельно. Результаты по синтетике и новым реальным
статьям нельзя усреднять в одну метрику.

## Результаты синтетического стресс-теста и следующий прогон

На 120 **вымышленных** статьях после первой проверки получено:

| Метрика | SBERT + фильтр событий | Экспериментальный reranker |
|---|---:|---:|
| Precision | 0.7091 | 0.7379 |
| Recall | 0.8211 | 0.8000 |
| Micro-F1 | 0.7610 | 0.7677 |
| Exact match | 0.6667 | 0.6917 |
| Лишние теги на статьях без тегов | 0.0625 | 0.0 |

Это **не** реальная оценка качества: авторские эталонные метки
не подтверждены независимой разметкой, а часть синтетических материалов
допускает несколько разумных отраслевых тегов. Разница F1 мала, и
экспериментальная версия потеряла два верных тега.

После этого исправлены **только общие языковые случаи** экспериментального
тэггера: разработка/испытание БПЛА и его крыльев, конструкция катеров,
различение обычного применения дронов и разработки аппаратов.
Основной `processing/tagger.py` и веса Sentence-BERT не менялись.

Для честного сравнения новой версии с прежними синтетическими
результатами используйте **тот же сохранённый аудит** с оценками модели:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_tag_role_challenger.py" -v

.\.venv\Scripts\python.exe check_tag_role_challenger.py collector_test_results/synthetic_tagging_audit_120.json --reference benchmarks/synthetic_tagging_labels_120.csv --include-drafts --output collector_test_results/synthetic_role_challenger_120_v2.json
```

Повторно рассчитывать эмбеддинги для этого сравнения не нужно.
Не запускать обучение на реальных статьях для получения лучшей метрики
на этих примерах; все улучшения должны подтверждаться независимой
выборкой новых материалов и содержательной проверкой спорных меток.

## Комбинированный режим: SBERT + проверка главного события

Режим `hybrid` теперь доступен в основном
`processing/tagger.py`, но **не включён по умолчанию**.

Схема:
1. Обычный Sentence-BERT вычисляет сходство заголовка и summary
   с описаниями 18 тегов, учитывает статью и ключевые слова;
2. обычный порог и максимальный отрыв формируют предварительный
   набор тегов;
3. `processing/tag_role_challenger.py` проверяет соответствие тегов
   именно действию и объекту новости. Например, морские перевозки
   не тождественны судостроению, а применение готового беспилотника
   не тождественно его разработке;
4. при явном сообщении о выпуске или модернизации линии может быть
   предложен дополнительный тег `технологическое производство`,
   если его SBERT-оценка не ниже внутреннего экспериментального порога.

**Никакого обучения / обновления весов Sentence-BERT не выполняется.**
Режим NLI (`NEWS_TAGGER_BACKEND=nli`) также не меняется.
Сбор новостей не изменяется.

### Сначала проверить без БД и MAX

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_hybrid_tagging_integration.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_tag_role_challenger.py" -v

.\.venv\Scripts\python.exe check_focused_tagging.py benchmarks/synthetic_tagging_120.json --limit 120 --tagger-mode hybrid --min-score 0.38 --output collector_test_results/synthetic_hybrid_audit_120.json

.\.venv\Scripts\python.exe evaluate_tagging.py collector_test_results/synthetic_hybrid_audit_120.json benchmarks/synthetic_tagging_labels_120.csv --include-drafts --output collector_test_results/synthetic_hybrid_evaluation_120.json
```

Для сравнения на старом реальном наборе можно аналогично указать
`tagging_benchmark_100.json` и `--tagger-mode hybrid`. Но даже если
гибрид побеждает на этих 100 статьях, это **не независимая оценка**:
регулярные выражения обсуждались и корректировались после просмотра
этой выборки. Для новой выборки исключайте старые URL с помощью
`--exclude-urls-from tagging_benchmark_100.json`.

### Включение при штатном тэггировании — только осознанно

В окне PowerShell, из которого запускается основной процесс приложения:

```powershell
$env:NEWS_TAGGER_BACKEND = "hybrid"
```

Для возврата старого поведения в этом же окне:

```powershell
$env:NEWS_TAGGER_BACKEND = "legacy"
```

Это задаёт окружение только для текущего процесса PowerShell и
процессов, запущенных из него. Проверочный
`check_focused_tagging.py --tagger-mode hybrid` не меняет
настройки основного процесса и не публикует ничего в MAX.

**Важное предостережение:** гибридный режим может менять состав тегов
и статусы `ready/not_for_publication` при реальном запуске приложения.
До новых независимых проверок используйте его в read-only тестах,
а не в автоматической публикации.

## Сравнение гибрида и SBERT при одном пороге

Отчёт `synthetic_hybrid_evaluation_120.json` дал F1=0.738,
Precision=0.750 и Recall=0.7263. Предыдущая оценка SBERT
(0.761, 0.7091, 0.8211 соответственно) была выполнена с
`--min-score 0.35`, а гибрид — с `--min-score 0.38`.
Поэтому считать падение F1 результатом именно reranker нельзя.

Скрипт `check_tag_role_challenger.py` теперь переоценивает **оба
алгоритма** из одного сохранённого `ranked` при выбранном пороге.
SBERT повторно не запускается, обучение отсутствует, БД и MAX
не затрагиваются.

```powershell
.\.venv\Scripts\python.exe check_tag_role_challenger.py collector_test_results/synthetic_tagging_audit_120.json --reference benchmarks/synthetic_tagging_labels_120.csv --include-drafts --min-score 0.35 --output collector_test_results/synthetic_matched_035.json

.\.venv\Scripts\python.exe check_tag_role_challenger.py collector_test_results/synthetic_tagging_audit_120.json --reference benchmarks/synthetic_tagging_labels_120.csv --include-drafts --min-score 0.38 --output collector_test_results/synthetic_matched_038.json
```

Оба JSON включают `evaluation.baseline` и
`evaluation.candidate`. Только их следует сравнивать в пределах
одного и того же `min_score`. Если вы исследуете пороги по тем же
120 статьям, результаты становятся ещё менее независимыми;
настройки нельзя автоматически переносить на новые реальные новости.

## Подготовка

Поместить в корень репозитория два файла, собранных из предыдущих
проверок новостей:

- `tagging_benchmark_100.json` — 100 реальных собранных публикаций,
  включая непрофильные (полные тексты хранятся локально).
- `tagging_reference_100.csv` — предварительные ожидаемые теги.
  Метки **созданы как черновик**, а не подтверждённая эталонная разметка.
  Столбец `needs_review` указывает спорные случаи.

В CSV столбец `expected_tags` содержит точные имена тегов, разделённые
точкой с запятой. Пустое поле означает: подходящего тематического тега
**нет**. Не нужно притягивать новость к неподходящей рубрике. Для
подтверждённых меток установить `review_status=verified`.

## Тестирование

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_focused_tagging.py" -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_tagging_evaluation.py" -v

.\.venv\Scripts\python.exe check_focused_tagging.py tagging_benchmark_100.json --limit 100 --output collector_test_results/tagging_audit_100.json
```

Черновая проверка (для ориентира, **не финальное качество**):

```powershell
.\.venv\Scripts\python.exe evaluate_tagging.py collector_test_results/tagging_audit_100.json tagging_reference_100.csv --include-drafts --optimize
```

После ручного подтверждения меток убрать `--include-drafts`. Можно
добавить `--baseline ai_tag_results_v2_45.json`, если файл остался
в корне проекта: старый и новый алгоритмы сравниваются **по совпадающим
URL**, а не по двум несопоставимым выборкам.

## Защита от неподходящих тегов по типу события

Добавлена небольшая самостоятельная проверка
`processing/tagging_event_guard.py`. Она рассматривает **только заголовок**,
чтобы не считать любое упоминание компании/отрасли основанием для тега.
Отклоняются явно непрофильные события: обучение, конкурсы, фестивали,
дивиденды, акции, биржевые рейтинги, скидки и статистика перевозок.

Важное исключение: если в заголовке выставки/конференции явно сообщается
о новом оборудовании, выпуске, строительстве или модернизации, решение
оставляется Sentence-BERT. Собранные новости не удаляются; не меняется
этап сбора или фильтрации.

На старом JSON можно оценить её эффект **без повторного запуска
эмбеддинг-модели**:

```powershell
.\.venv\Scripts\python.exe evaluate_tagging.py collector_test_results/tagging_audit_100.json tagging_reference_100.csv --include-drafts --event-guard --output collector_test_results/tagging_event_guard_evaluation.json
```

Результат содержит before/after и сравнение train/holdout. Разметка
`draft` — черновик, не эталон. Правила созданы после просмотра всех
100 примеров, поэтому метрики holdout для нового правила не являются
строго независимым подтверждением качества и не должны использоваться
как окончательная характеристика модели.

Для повторного запуска текущей модели (уже со встроенной защитой):

```powershell
.\.venv\Scripts\python.exe check_focused_tagging.py tagging_benchmark_100.json --limit 100 --output collector_test_results/tagging_audit_100_guarded.json
```

В JSON будут `event_guard_reason` и `raw_tags_before_event_guard`,
что позволяет проверять каждое исключение. Уже опубликованные новости
не пересчитываются автоматически.

## Что показывают оценки

- **Micro Precision:** доля правильно поставленных тегов среди
  всех поставленных.
- **Micro Recall:** доля найденных тегов среди всех ожидаемых.
- **Micro F1:** гармоническое среднее Precision/Recall.
- **Exact match:** доля статей, у которых список тегов совпал полностью.
- **False tag rate on untagged:** доля непрофильных статей, которым
  ошибочно назначен хотя бы один тег.
- **Macro F1:** среднее по тегам с поддержкой в эталоне.

Подбор `--optimize` проверяет несколько сочетаний веса заголовка +
summary, минимальной оценки тега и максимального отрыва от лучшего
тега. Конфигурация подбирается на `train` (70%), проверяется на
`holdout` (30%). Выбор параметров **не переписывает** рабочий
`processing/tagger.py`.

**Ограничение:** первоначальный черновик включает много негативных
примеров и неравномерно покрывает 18 тематик; все числа до ручной
проверки и расширения положительных примеров предварительны.
Одинаковые сюжеты в нескольких источниках также могут создавать
коррелированные примеры. Ни в коем случае не считать эти метрики
окончательной статистикой качества.

**Отдельная проблема таксономии:** в фактических 18 категориях
отсутствует самостоятельный тег «логистика». Новости, посвящённые
только перевозкам и отгрузкам, не следует автоматически относить к
«машиностроению» только ради непустого результата.
