import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

from database.models import Cluster, News
from processing.clustering import build_embedding



NOT_PUBLICATION_KEYWORDS = [
    "недвижимость",
    "земельный участок",
    "земельные участки",
    "изъятие имущества",
    "изымают недвижимость",
    "комплексное развитие территории",
    "крт",
    "земля и имущество",
    "право собственности",
    "собственник недвижимости",
    "владельцы недвижимости",
    "девелопер",
    "застройщик",
    "жилая застройка",
    "жилой комплекс",
    "ипотека",
    "аренда",
    "купля-продажа",
"кадровый резерв",
        "инженерная школа",
        "ипотека",
        "музей",
        "выставка",
        "соболезнования",
        "дивиденды",
        "акционеры",
        "туристический маршрут",
]

TAG_NEGATIVE_KEYWORDS = {
    "авиация": [
        "судно на подводных крыльях",
        "подводных крыльях",
        "судно",
        "корабль",
        "верфь",
        "флот",
        "вмф",
        "оск",
        "севмаш",
        "краболов",
        "фрегат",
        "корвет",
        "тральщик",
        "подводная лодка",
        "апл",
    ],

    "судостроение": [
        "самолет",
        "самолёт",
        "вертолет",
        "вертолёт",
        "авиадвигатель",
        "авиационный двигатель",
        "sj-100",
        "суперджет",
        "мс-21",
        "пд-8",
        "пд-14",
        "оак",
    ],

    "атомная отрасль": [
        "тепловая электростанция",
        "тэс",
        "грэс",
        "тэц",
        "угольная генерация",
        "газовая генерация",
        "нефтегаз",
        "нпз",
        "бурение",
        "скважина",
        "металлургия",
        "палладий",
        "никель",
        "скандий",
        "алюминий",
    ],

    "ТЭК": [
        "аэс",
        "атомная электростанция",
        "реактор",
        "ядерное топливо",
        "твэл",
        "самолет",
        "вертолет",
        "судно",
        "фрегат",
        "краболов",
        "металлургический завод",
    ],

    "нефтегазовая промышленность": [
        "аэс",
        "атомная электростанция",
        "реактор",
        "ядерное топливо",
        "тэц",
        "грэс",
        "тэс",
        "самолет",
        "вертолет",
        "судно",
        "корабль",
        "металлургия",
        "палладий",
        "никель",
        "алюминий",
    ],

    "металлургия": [
        "аэс",
        "реактор",
        "тэц",
        "грэс",
        "тэс",
        "нпз",
        "бурение",
        "скважина",
        "самолет",
        "вертолет",
        "судно",
        "корабль",
        "ипотека",
        "банкротство",
    ],

    "машиностроение": [
        "кадровый резерв",
        "инженерная школа",
        "ипотека",
        "музей",
        "выставка",
        "соболезнования",
        "дивиденды",
        "акционеры",
        "туристический маршрут",
    ],

    "заводы": [
        "музей",
        "выставка",
        "соболезнования",
        "дивиденды",
        "акционеры",
        "туристический маршрут",
        "ипотека",
        "кадровый резерв",
        "спортивный праздник",
        "недвижимость",
        "земельный участок",
        "земельные участки",
        "изъятие имущества",
        "комплексное развитие территории",
        "крт",
        "застройщик",
        "девелопер",
    ],

    "технологическое производство": [
        "музей",
        "выставка",
        "соболезнования",
        "дивиденды",
        "акционеры",
        "туристический маршрут",
        "ипотека",
        "кадровый резерв",
    ],

    "электротехническое направление": [
        "судно",
        "корабль",
        "самолет",
        "вертолет",
        "нпз",
        "бурение",
        "аэс",
        "реактор",
        "музей",
        "дивиденды",
    ],

    "технологии": [
        "ипотека",
        "дивиденды",
        "акционеры",
        "музей",
        "соболезнования",
        "туристический маршрут",
        "спортивный праздник",
    ],

    "инновации": [
        "дивиденды",
        "акционеры",
        "ипотека",
        "соболезнования",
        "музей",
        "туристический маршрут",
        "спортивный праздник",
        "праздник",
    ],

    "инжиниринг": [
        "дивиденды",
        "акционеры",
        "ипотека",
        "соболезнования",
        "музей",
        "туристический маршрут",
        "спортивный праздник",
        "кадровый резерв",
        "недвижимость",
        "земельный участок",
        "земельные участки",
        "изъятие имущества",
        "комплексное развитие территории",
        "крт",
        "застройщик",
        "девелопер",
    ],
}

MILITARY_MEDIA_SOURCES = [
    "Военное.рф",
    "ФлотПром",
    "ФлотКом",
]

RUSSIA_MARKERS = [
    "россия", "российский", "российская", "российские", "российской",
    "рф", "в россии", "минпромторг", "ростех", "росатом", "роснефть",
    "газпром", "лукойл", "норникель", "северсталь", "оск", "оак", "одк",
    "камаз", "русал", "калашников", "севмаш", "санкт-петербург",
    "москва", "татарстан", "башкирия", "якутия", "калининград",
    "ленобласть", "челябинская область", "нижегородский нпз"
]

FOREIGN_MARKERS = [
    "турция", "турецкий", "турецкая", "турецкие",
    "сша", "американский", "американская", "американские",
    "вмс сша", "пентагон", "virginia", "northrop grumman",
    "иран", "иранский", "иранская", "иранские",
    "австралия", "австралийский", "норвегия", "норвежский",
    "словакия", "словацкий", "польша", "польский",
    "китай", "китайский", "германия", "немецкий",
    "франция", "французский", "великобритания", "британский",
    "нато", "европа", "европейский", "fnss", "naval today",
    "navy recognition", "konsberg", "anadolu shipyard"
]

FOREIGN_TITLE_MARKERS = [
    "турецкий", "турецкая", "турецкие", "турецкого", "турецком",
    "американский", "американская", "американские", "американских",
    "вмс сша", "сша", "пентагон",
    "иран", "иранский", "иранские",
    "австралия", "норвегия", "словакия", "польша",
    "northrop grumman", "virginia", "naval today", "navy recognition",
    "fnss", "anadolu shipyard",
]

RUSSIAN_CONTEXT_MARKERS = [
    "россия", "российский", "российская", "российские", "рф",
    "в россии", "ростех", "росатом", "оск", "оак", "одк",
    "лукойл", "роснефть", "газпром", "норникель", "русал",
]

TAGS = {
    "заводы": (
        "заводы, фабрики, промышленные предприятия, производственные площадки, "
        "цеха, промышленные комплексы, индустриальные парки, ОЭЗ, запуск завода, "
        "открытие производства, строительство предприятия, расширение мощностей, "
        "модернизация завода, реконструкция производства, новые линии, выпуск продукции, "
        "локализация производства, импортозамещение, производственные мощности"
    ),

    "металлургия": (
        "металлургия, металлургический завод, черная металлургия, цветная металлургия, "
        "сталь, чугун, прокат, листовой металл, алюминий, медь, никель, титан, "
        "ферросплавы, сплавы, плавка, доменная печь, электропечь, шлак, рудное сырье, "
        "металлопрокат, обработка металла, контроль качества металла, Северсталь, Мечел, "
        "Норникель, Красный Октябрь,"
        "РУСАЛ, алюминиевое производство, скандий, оксид скандия, палладий, платина, "
        "горно-металлургический комбинат, металлургическое оборудование"
    ),

    "машиностроение": (
        "машиностроение, станкостроение, промышленное оборудование, станки, агрегаты, "
        "механизмы, двигатели, насосы, компрессоры, турбины, редукторы, подшипники, "
        "производство машин, оборудование для заводов, технологические установки, "
        "механическая обработка, приборостроение, тяжелое машиностроение, "
        "энергетическое машиностроение, ОДК, Ростех,"
        "экскаватор, погрузчик, спецтехника, дорожно-строительная техника, ЧПУ, "
        "обрабатывающий центр, токарный станок, фрезерная голова, поворотный стол"
    ),

    "ТЭК": (
        "топливно-энергетический комплекс, энергетика, электроэнергетика, энергосистема, "
        "электростанции, ТЭС, ГРЭС, ТЭЦ, ГЭС, генерация, энергоснабжение, "
        "теплоснабжение, котельные, турбины, энергоблоки, сети, подстанции, "
        "электроэнергия, мощность, энергосбережение, энергетическая инфраструктура, "
        "модернизация энергообъектов,"
        "тепловые сети, гидравлические испытания, энергообъекты, энергогенерация, "
        "электроснабжение, тепловая генерация, угольные ТЭС, КОММод"
    ),

    "технологическое производство": (
        "технологическое производство, производственный процесс, технологическая линия, "
        "производственная линия, выпуск продукции, серийное производство, опытное производство, "
        "модернизация производства, автоматизация процесса, промышленная технология, "
        "контроль качества, технологический цикл, переработка сырья, обработка материалов, "
        "новый продукт, производственная эффективность, повышение производительности"
    ),

    "авиация": (
        "авиационная промышленность, самолетостроение, вертолетостроение, авиастроение, "
        "самолеты, вертолеты, беспилотные летательные аппараты, БПЛА, дроны, "
        "авиадвигатели, авиационные системы, авионика, авиационные материалы, "
        "сертификация самолетов, летные испытания, аэропортовая техника, "
        "Объединенная авиастроительная корпорация, ОАК, ОДК, Ростех, SJ-100, МС-21,"
        "ПД-8, ПД-14, Суперджет, SSJ, Ил-96, Ил-76, Ту-214, Ансат, Ми-8, Ка-226, "
        "летательные аппараты, авиационный двигатель, авиазавод, Казанский вертолетный завод"
    ),

    "судостроение": (
        "судостроение, кораблестроение, верфи, судоверфи, корабли, суда, флот, ВМФ, "
        "ледоколы, танкеры, траулеры, краболовы, фрегаты, корветы, подводные лодки, "
        "морская техника, судовое оборудование, судовые двигатели, винторулевые колонки, "
        "спуск судна на воду, закладка корабля, строительство судов, ОСК, Севмаш, "
        "Адмиралтейские верфи, Северная верфь, Красное Сормово,"
        "Северное ПКБ, Средне-Невский судостроительный завод, Балтийский завод, "
        "корабелы, кораблестроение, судоремонт, тральщик, краболов, надводный корабль"
    ),

    "электротехническое направление": (
        "электротехническая промышленность, электрооборудование, электротехника, кабели, "
        "кабельная продукция, трансформаторы, электродвигатели, генераторы, распределительные устройства, "
        "подстанции, силовая электроника, преобразователи тока, датчики, реле, "
        "электрические системы, промышленная автоматика, оборудование для энергосетей, "
        "электротехнические компоненты"
    ),

    "атомная отрасль": (
        "атомная отрасль, атомная энергетика, Росатом, АЭС, атомная электростанция, "
        "энергоблок, реактор, ВВЭР, РБМК, ядерное топливо, ТВЭЛ, уран, "
        "атомное машиностроение, ядерные технологии, радиационные технологии, "
        "атомный ледокол, Курская АЭС, Ленинградская АЭС, Аккую, Руппур, "
        "пусконаладочные работы, загрузка ядерного топлива"
    ),

    "нефтегазовая промышленность": (
        "нефтегазовая промышленность, нефтяная отрасль, газовая отрасль, добыча нефти, "
        "добыча газа, бурение, скважины, месторождения, нефтепереработка, НПЗ, "
        "газопереработка, нефтехимия, трубопроводы, газопроводы, компрессорные станции, "
        "СПГ, сжиженный природный газ, моторное топливо, бензин, дизель, МТБЭ, "
        "Газпром, Роснефть, ЛУКОЙЛ, Самотлорнефтегаз, Самаранефтегаз,"
        "нефтедобыча, нефтяники, нефтесервис, гидроразрыв пласта, ГРП, дебит нефти, "
        "энергоэффективность нефтедобычи, Нижегородский НПЗ"
    ),

    "технологии": (
        "промышленные технологии, цифровизация промышленности, цифровая трансформация, "
        "искусственный интеллект, ИИ, машинное обучение, промышленное ПО, MES, ERP, "
        "цифровые платформы, роботизация, автоматизация, датчики, мониторинг, "
        "цифровой двойник, программные решения, управление производством, "
        "информационные системы, обработка данных"
    ),

    "инновации": (
        "инновации, новые разработки, научные разработки, опытные образцы, испытания, "
        "исследования, новые материалы, перспективные технологии, экспериментальные решения, "
        "внедрение новых технологий, опытно-конструкторские работы, импортозамещенные решения, "
        "стартапы, акселератор, высокотехнологичные идеи, технологический прорыв, "
        "новые продукты, новые методы производства"
    ),

    "инжиниринг": (
        "инжиниринг, инженерные решения, проектирование, конструкторские разработки, "
        "конструкторская документация, РКД, НИОКР, опытно-конструкторские работы, "
        "разработка оборудования, техническая документация, промышленное проектирование, "
        "испытания оборудования, расчетные модели, инженерные системы, "
        "проектные институты, КБ, конструкторское бюро, технологический проект"
    ),
}

TAG_KEYWORDS = {
    "заводы": [
        "запуск завода", "открытие завода", "строительство завода",
        "новый завод", "модернизация завода", "реконструкция завода",
        "производственная площадка", "индустриальный парк", "ОЭЗ",
        "производственные мощности", "новая производственная линия",
        "запуск производства", "расширение производства",
    ],

    "металлургия": [
        "металлургия", "металлургический завод", "сталь", "чугун",
        "прокат", "листовой металл", "алюминий", "медь", "никель",
        "титан", "палладий", "платина", "скандий", "оксид скандия",
        "шлак", "электропечь", "доменная печь", "плавка",
        "металлопрокат", "РУСАЛ", "Норникель", "Северсталь",
        "Красцветмет", "Мечел",
        "палладиевая лаборатория",
        "палладиевые технологии",
        "новые материалы из палладия",
        "металлы платиновой группы",
        "МПГ",
    ],

    "машиностроение": [
        "машиностроение", "станкостроение", "станок", "станки",
        "ЧПУ", "обрабатывающий центр", "токарный станок",
        "фрезерная голова", "поворотный стол", "механическая обработка",
        "экскаватор", "погрузчик", "спецтехника",
        "дорожно-строительная техника", "турбина", "насос",
        "компрессор", "редуктор", "СТАН", "Калашников",
    ],

    "ТЭК": [
        "ТЭК", "топливно-энергетический комплекс", "энергетика",
        "электроэнергетика", "энергосистема", "электростанция",
        "ТЭС", "ГРЭС", "ТЭЦ", "ГЭС", "энергоблок", "генерация",
        "теплоснабжение", "электроснабжение", "тепловые сети",
        "котельная", "подстанция", "энергосбережение",
        "энергообъект", "угольная генерация", "энергогенерация",
    ],

    "технологическое производство": [
        "технологическая линия", "производственная линия",
        "серийное производство", "опытное производство",
        "технологический процесс", "производственный процесс",
        "автоматизация производства", "контроль качества",
        "переработка сырья", "обработка материалов",
        "повышение производительности", "производственная эффективность",
    ],

    "авиация": [
        "авиация", "авиастроение", "самолетостроение",
        "вертолетостроение", "самолет", "самолёт", "вертолет",
        "вертолёт", "БПЛА", "беспилотник", "дрон",
        "авиадвигатель", "авиационный двигатель", "ПД-8", "ПД-14",
        "SJ-100", "SSJ", "Суперджет", "МС-21", "Ил-96",
        "Ил-76", "Ту-214", "Ансат", "Ми-8", "Ка-226",
        "ОАК", "ОДК", "Казанский вертолетный завод",
    ],

    "судостроение": [
        "судостроение", "кораблестроение", "судоремонт",
        "верфь", "судоверфь", "корабль", "судно", "флот",
        "ВМФ", "ледокол", "танкер", "траулер", "краболов",
        "фрегат", "корвет", "тральщик", "подводная лодка",
        "АПЛ", "судовое оборудование", "закладка корабля",
        "спуск судна на воду", "ОСК", "Севмаш", "Северная верфь",
        "Адмиралтейские верфи", "Красное Сормово", "Северное ПКБ",
        "СПК",
        "судно на подводных крыльях",
        "подводные крылья",
        "Комета 120М",
        "скоростное пассажирское судно",
        "пассажирское судно",
        "Азово-Черноморский бассейн",
    ],

    "электротехническое направление": [
        "электротехника", "электротехнический", "электрооборудование",
        "кабель", "кабельная продукция", "трансформатор",
        "электродвигатель", "генератор", "распределительное устройство",
        "силовая электроника", "преобразователь тока", "реле",
        "электрические системы", "промышленная автоматика",
    ],

    "атомная отрасль": [
        "Росатом", "атомная отрасль", "атомная энергетика",
        "АЭС", "атомная электростанция", "АСММ", "реактор",
        "ВВЭР", "РБМК", "ядерное топливо", "ТВЭЛ", "уран",
        "атомный ледокол", "Курская АЭС", "Ленинградская АЭС",
        "Аккую", "Руппур", "загрузка ядерного топлива",
    ],

    "нефтегазовая промышленность": [
        "нефтегаз", "нефтяная отрасль", "газовая отрасль",
        "добыча нефти", "добыча газа", "бурение", "скважина",
        "месторождение", "нефтепереработка", "НПЗ",
        "газопереработка", "нефтехимия", "трубопровод",
        "газопровод", "компрессорная станция", "СПГ",
        "бензин", "дизель", "МТБЭ", "гидроразрыв пласта",
        "ГРП", "дебит нефти", "ЛУКОЙЛ", "Роснефть",
        "Газпром", "Самотлорнефтегаз", "Самаранефтегаз",
        "РН-Юганскнефтегаз",
    ],

    "технологии": [
        "цифровизация", "цифровая трансформация",
        "искусственный интеллект", "машинное обучение",
        "промышленное ПО", "MES", "ERP", "цифровая платформа",
        "роботизация", "цифровой двойник", "информационная система",
        "VR", "виртуальная реальность", "облачные технологии",
        "мониторинг оборудования",
    ],

    "инновации": [
        "новая разработка", "новые разработки", "научная разработка",
        "опытный образец", "опытные образцы", "новые материалы",
        "перспективные технологии", "экспериментальное решение",
        "внедрение новых технологий", "импортозамещенное решение",
        "импортозамещённое решение", "стартап", "акселератор",
        "высокотехнологичные идеи", "технологический прорыв",
        "лаборатория",
        "новые материалы",
        "исследование материалов",
        "генерация новых материалов",
        "ИИ-платформа",
    ],

    "инжиниринг": [
        "инжиниринг", "инженерные решения", "проектирование",
        "конструкторская разработка", "конструкторская документация",
        "РКД", "НИОКР", "разработка оборудования",
        "техническая документация", "промышленное проектирование",
        "испытания оборудования", "инженерные системы",
        "конструкторское бюро", "проектный институт",
    ],
}

_tag_embeddings_cache = None


def cosine_sim(vec1, vec2) -> float:
    if vec1 is None or vec2 is None:
        return 0.0

    a = np.array(vec1, dtype=float).reshape(1, -1)
    b = np.array(vec2, dtype=float).reshape(1, -1)

    return float(cosine_similarity(a, b)[0][0])


def get_tag_embeddings() -> dict[str, list[float]]:
    """
    Строит embedding для описаний тегов.
    Кэшируется, чтобы не пересчитывать на каждом кластере.
    """
    global _tag_embeddings_cache

    if _tag_embeddings_cache is not None:
        return _tag_embeddings_cache

    result = {}

    for tag, description in TAGS.items():
        result[tag] = build_embedding(description)

    _tag_embeddings_cache = result

    return result


def get_cluster_text_for_tagging(session, cluster: Cluster) -> str:
    parts = []

    if cluster.title:
        # Заголовок самый важный
        parts.append(cluster.title)
        parts.append(cluster.title)
        parts.append(cluster.title)

    if cluster.summary_text:
        # Summary важнее полного текста
        parts.append(cluster.summary_text)
        parts.append(cluster.summary_text)

    news_items = (
        session.query(News)
        .filter(News.cluster_id == cluster.id)
        .order_by(News.is_primary.desc(), News.published_at.asc().nullslast(), News.collected_at.asc())
        .all()
    )

    for news in news_items[:3]:
        if news.title:
            parts.append(news.title)
            parts.append(news.title)

        if news.text:
            # Не надо брать слишком много текста.
            # В длинном тексте часто появляются посторонние блоки и общие слова.
            parts.append(news.text[:700])

    return "\n".join(parts)

def score_tags(text: str) -> list[tuple[str, float]]:
    """
    Возвращает все теги с итоговым score.

    Итоговый score = embedding similarity + keyword boost.
    """
    if not text:
        return []

    text_embedding = build_embedding(text)
    tag_embeddings = get_tag_embeddings()

    scored_tags = []

    for tag, tag_embedding in tag_embeddings.items():
        embedding_score = cosine_sim(text_embedding, tag_embedding)
        boost = keyword_boost_for_tag(text, tag)

        final_score = embedding_score + boost

        scored_tags.append((tag, final_score))

    scored_tags.sort(key=lambda item: item[1], reverse=True)

    return scored_tags


def define_cluster_tags(
    text: str,
    min_score: float = 0.38,
    max_tags: int = 2
) -> list[str]:
    """
    Определяет теги по embedding-сходству.
    """
    scored_tags = score_tags(text)

    selected_tags = [
        tag for tag, score in scored_tags
        if score >= min_score
    ]

    return selected_tags[:max_tags]


def tag_cluster(
    session,
    cluster_id: int,
    min_score: float = 0.38,
    max_tags: int = 3,
    debug: bool = True
) -> list[str]:
    cluster = session.query(Cluster).filter(Cluster.id == cluster_id).first()

    if not cluster:
        return []

    if is_foreign_news_by_title(cluster):
        cluster.tags = None
        cluster.status = "not_for_publication"
        session.commit()

        if debug:
            print("-" * 100)
            print(f"Кластер #{cluster.id}")
            print(f"Заголовок: {cluster.title}")
            print("Статус: not_for_publication")
            print("Причина: зарубежная новость без связи с РФ")

        return []

    text = get_cluster_text_for_tagging(session, cluster)

    if not cluster_has_content(session, cluster.id):
        cluster.tags = None
        cluster.status = "not_for_publication"

        session.commit()

        if debug:
            print("-" * 100)
            print(f"Кластер #{cluster.id}")
            print("Статус: not_for_publication")
            print("Причина: отсутствует полноценный текст")

        return []

    if is_not_publication_by_keywords(text):
        cluster.tags = None
        cluster.status = "not_for_publication"
        session.commit()

        if debug:
            print("-" * 100)
            print(f"Кластер #{cluster.id}")
            print(f"Заголовок: {cluster.title}")
            print("Статус: not_for_publication")
            print("Причина: стоп-слова нерелевантной темы")

        return []

    scored_tags = score_tags_detailed(text)

    selected_tags = select_tags_by_score_gap(
        scored_tags=scored_tags,
        min_score=min_score,
        max_tags=max_tags,
        max_gap_from_best=0.12
    )

    if selected_tags:
        cluster.tags = ", ".join(selected_tags)
        cluster.status = "ready"
    else:
        cluster.tags = None
        cluster.status = "not_for_publication"

    session.commit()

    if debug:
        print("-" * 100)
        print(f"Кластер #{cluster.id}")
        print(f"Заголовок: {cluster.title}")
        print(f"Статус: {cluster.status}")
        print(f"Выбранные теги: {', '.join(selected_tags) if selected_tags else 'не определены'}")
        print("Похожие теги:")

        for item in scored_tags[:8]:
            print(
                f"  {item['tag']}: "
                f"{item['final_score']:.3f} "
                f"(embedding={item['embedding_score']:.3f}, "
                f"boost={item['keyword_boost']:.3f}, "
                f"penalty={item['keyword_penalty']:.3f})"
            )

    return selected_tags



def tag_new_clusters(
    session,
    min_score: float = 0.38,
    max_tags: int = 2,
    debug: bool = True
):
    """
    Тегирует кластеры, у которых уже есть summary.
    Если кластер уже был not_for_publication, можно перетегировать.
    """
    clusters = (
        session.query(Cluster)
        .filter(Cluster.summary_text.isnot(None))
        .filter(
            (Cluster.tags.is_(None)) |
            (Cluster.status == "not_for_publication")
        )
        .all()
    )

    print(f"Кластеров для тегирования: {len(clusters)}")

    for cluster in clusters:
        tag_cluster(
            session=session,
            cluster_id=cluster.id,
            min_score=min_score,
            max_tags=max_tags,
            debug=debug
        )


def retag_all_clusters(
    session,
    min_score: float = 0.38,
    max_tags: int = 2,
    debug: bool = True
):
    """
    Полностью пересчитывает теги для всех кластеров с summary.
    Удобно использовать после изменения описаний тегов или порога.
    """
    clusters = (
        session.query(Cluster)
        .filter(Cluster.summary_text.isnot(None))
        .all()
    )

    print(f"Кластеров для полного перетегирования: {len(clusters)}")

    for cluster in clusters:
        cluster.tags = None
        session.commit()

        tag_cluster(
            session=session,
            cluster_id=cluster.id,
            min_score=min_score,
            max_tags=max_tags,
            debug=debug
        )

def normalize_for_keywords(text: str) -> str:
    if not text:
        return ""

    text = text.lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\- ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def count_keyword_matches(text: str, keywords: list[str]) -> int:
    normalized_text = normalize_for_keywords(text)

    count = 0

    for keyword in keywords:
        normalized_keyword = normalize_for_keywords(keyword)

        if not normalized_keyword:
            continue

        if normalized_keyword in normalized_text:
            count += 1

    return count

def keyword_boost_for_tag(text: str, tag: str) -> float:
    keywords = TAG_KEYWORDS.get(tag, [])

    if not keywords:
        return 0.0

    matches = count_keyword_matches(text, keywords)

    if matches == 0:
        return 0.0

    return min(0.03 * matches, 0.09)

def score_tags_detailed(text: str) -> list[dict]:
    if not text:
        return []

    text_embedding = build_embedding(text)
    tag_embeddings = get_tag_embeddings()

    result = []

    for tag, tag_embedding in tag_embeddings.items():
        embedding_score = cosine_sim(text_embedding, tag_embedding)
        boost = keyword_boost_for_tag(text, tag)
        penalty = keyword_penalty_for_tag(text, tag)

        final_score = embedding_score + boost - penalty

        result.append({
            "tag": tag,
            "embedding_score": embedding_score,
            "keyword_boost": boost,
            "keyword_penalty": penalty,
            "final_score": final_score,
        })

    result.sort(key=lambda item: item["final_score"], reverse=True)

    return result

def select_tags_by_score_gap(
        scored_tags: list[dict],
        min_score: float = 0.38,
        max_tags: int = 2,
        max_gap_from_best: float = 0.08
) -> list[str]:
    if not scored_tags:
        return []

    best_score = scored_tags[0]["final_score"]

    selected = []

    for item in scored_tags:
        tag = item["tag"]
        score = item["final_score"]

        if score < min_score:
            continue

        if best_score - score > max_gap_from_best:
            continue

        selected.append(tag)

        if len(selected) >= max_tags:
            break

    return selected

def is_foreign_military_news(session, cluster: Cluster) -> bool:
    news_items = (
        session.query(News)
        .filter(News.cluster_id == cluster.id)
        .all()
    )

    has_military_source = any(
        news.source in MILITARY_MEDIA_SOURCES
        for news in news_items
    )

    if not has_military_source:
        return False

    text_parts = []

    if cluster.title:
        text_parts.append(cluster.title)

    if cluster.summary_text:
        text_parts.append(cluster.summary_text)

    for news in news_items:
        if news.title:
            text_parts.append(news.title)
        if news.text:
            text_parts.append(news.text[:1000])

    text = " ".join(text_parts).lower()

    return any(marker.lower() in text for marker in FOREIGN_MARKERS)

def cluster_full_text_for_filter(session, cluster: Cluster) -> str:
    parts = []

    if cluster.title:
        parts.append(cluster.title)

    if cluster.summary_text:
        parts.append(cluster.summary_text)

    news_items = (
        session.query(News)
        .filter(News.cluster_id == cluster.id)
        .all()
    )

    for news in news_items:
        if news.title:
            parts.append(news.title)

        if news.text:
            parts.append(news.text[:1500])

        if news.url:
            parts.append(news.url)

    return " ".join(parts).lower()


def contains_any_marker(text: str, markers: list[str]) -> bool:
    return any(marker.lower() in text for marker in markers)


def is_not_related_to_russia(session, cluster: Cluster) -> bool:
    text = cluster_full_text_for_filter(session, cluster)

    has_foreign = contains_any_marker(text, FOREIGN_MARKERS)
    has_russia = contains_any_marker(text, RUSSIA_MARKERS)

    # Если явно зарубежная новость и нет признаков связи с РФ — не публикуем
    if has_foreign and not has_russia:
        return True

    return False

def normalize_filter_text(text: str) -> str:
    if not text:
        return ""

    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\- ]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_foreign_news_by_title(cluster: Cluster) -> bool:
    title = normalize_filter_text(cluster.title or "")

    has_foreign_marker = any(
        marker in title
        for marker in FOREIGN_TITLE_MARKERS
    )

    has_russian_marker = any(
        marker in title
        for marker in RUSSIAN_CONTEXT_MARKERS
    )

    return has_foreign_marker and not has_russian_marker

def keyword_penalty_for_tag(text: str, tag: str) -> float:
    negative_keywords = TAG_NEGATIVE_KEYWORDS.get(tag, [])

    if not negative_keywords:
        return 0.0

    matches = count_keyword_matches(text, negative_keywords)

    if matches == 0:
        return 0.0

    return min(0.05 * matches, 0.15)

def is_not_publication_by_keywords(text: str) -> bool:
    normalized_text = normalize_for_keywords(text)

    for keyword in NOT_PUBLICATION_KEYWORDS:
        normalized_keyword = normalize_for_keywords(keyword)

        if normalized_keyword in normalized_text:
            return True

    return False

def cluster_has_content(session, cluster_id: int) -> bool:
    news_items = (
        session.query(News)
        .filter(News.cluster_id == cluster_id)
        .all()
    )

    if not news_items:
        return False

    for news in news_items:
        if news.title and news.text and len(news.text.strip()) > 200:
            return True

    return False