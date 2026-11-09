"""Event-grounded industry tagging without LLMs or network calls.

The headline identifies the event and object; a summary or the opening
article passage may only disambiguate that event. A company's name and
incidental sector mentions never suffice as evidence. SBERT scores may break
ties among supported categories, but never create a category on their own.

The pure classifier is used by production and the read-only audit alike.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class TagDecision:
    tags: tuple[str, ...]
    outcome: str                 # tagged | reject | review
    reason: str
    evidence: tuple[str, ...] = ()


def norm(value: str) -> str:
    return " ".join(str(value or "").casefold().replace("ё", "е").split())


def rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.I)


STRONG_EVENT = rx(
    r"\b(?:разработ\w*|созда\w*|создан\w*|изготов\w*|"
    r"выпуст\w*|выпуск\w*|производ\w*|начал\w*|приступ\w*|"
    r"запуст\w*|запущ\w*|запуск\w*|ввел\w*|постро\w*|"
    r"строительств\w*|модерниз\w*|усовершенств\w*|внедр\w*|"
    r"осво\w*|испыт\w*|обкат\w*|спуст\w*|спущ\w*|заклад\w*|"
    r"реконстру\w*|восстан\w*|возрожда\w*|автоматиз\w*|"
    r"перевел\w*|перевод\w*|переоснащ\w*|налаж\w*|"
    r"совершенств\w*|упростил\w*|"
    r"воссозда\w*|воссозд\w*|повысил\w*|повысили|"
    r"подобрал\w*|извлекл\w*|добы\w*|добыт\w*|"
    r"стартовал\w*|стартовало|стартовали|вышл\w*\s+в\s+серию|"
    r"вывел\w*\s+на\s+(?:обкатк\w*|рынок)|"
    r"установил\w*|оснастил\w*|"
    r"откр\w*|переработ\w*|вывод\w*\s+на\s+рынок)\b"
)

# These are main-event exclusions, not bans on mentioning the words in a
# long article. Put the strongest checks ahead of category evidence.
NOT_INDUSTRIAL = (
    ("roundup", rx(
        r"^(?:главн\w*\s+в\s+\w*\s+сми|дайджест\b|"
        r"обзор\s+новостей|итоги\s+недели)"
    )),
    ("finance_or_regulation", rx(
        r"\b(?:банкротств\w*|акционер\w*|дивиденд\w*|котировк\w*|"
        r"некредитн\w*\s+рейтин\w*|страхов\w*\s+взнос\w*|"
        r"ипотек\w*|автокредит\w*|продать\b|продаж\w*\s+завод\w*|"
        r"закон\w*|законодатель\w*|налог\w*|"
        r"субсид\w*|финансировани\w*\s+ремонт\w*|"
        r"обяза\w*\s+создава\w*\s+фонд\w*|"
        r"госфонд\w*|гохран\w*|"
        r"отзыв\w*\s+лицензи\w*|взыска\w*\s+.*ущерб\w*|"
        r"предупредил\w*.{0,55}ответственност\w*|"
        r"итог\w*\s+производственн\w*\s+деятельност\w*|"
        r"остановил\w*\s+рудник\w*)\b"
    )),
    ("education_culture", rx(
        r"\b(?:обучени\w*|олимпиад\w*|конкурс\w*|фестивал\w*|"
        r"праздничн\w*\s+акц\w*|поздравлени\w*|"
        r"сериал\w*|мультсериал\w*|фильм\w*|кинотеатр\w*|"
        r"историческ\w*\s+драм\w*|"
        r"школьник\w*|студент\w*\s+посетил\w*|"
        r"техникум\w*|образовательн\w*\s+проект\w*|"
        r"для\s+обучени\w*|"
        r"турист\w*|юбиле\w*|лети\w*\s+со\s+дня|"
        r"учебн\w*\s+центр\w*|"
        r"центр\w*\s+подбор\w*\s+персонал\w*)\b"
    )),
    ("publicity", rx(
        r"\b(?:выставк\w*|конференци\w*|форум\w*|семинар\w*|"
        r"ярмарк\w*|приглаша\w*|обсудят|обсудили|"
        r"преми\w*\s+фильм\w*|бизнес.диалог\w*|"
        r"договор\w*\s+с\s+проф\w*|"
        r"наград\w*|слет\w*|лидер\s+рынк\w*|реклам\w*|"
        r"крупнейш\w*\s+площадк\w*)\b"
    )),
    ("operations_or_incident", rx(
        r"\b(?:погрузк\w*|отгрузк\w*|грузоперевозк\w*|"
        r"пассажиропоток\w*|авар\w*|дтп\b|"
        r"удар\w*\s+(?:беспилотник\w*|бпла\b)|"
        r"подверг\w*\s+удар\w*|"
        r"протест\w*|рейс\w*|для\s+обучени\w*|"
        r"останов\w*\s+добыч\w*|"
        r"преступлени\w*|дорожа\w*)\b"
    )),
    ("agriculture_or_social", rx(
        r"\b(?:урожай\w*|озим\w*|посев\w*|"
        r"мальк\w*|осетр\w*|рекультивац\w*|"
        r"свадьб\w*|птицефабрик\w*|"
        r"сбор\w*\s+икр\w*|"
        r"художественн\w*\s+фильм\w*)\b"
    )),
)
DEFINITE_DELIVERABLE = rx(
    r"\b(?:разработ\w*|изготов\w*|запуст\w*|"
    r"выпуст\w*|внедр\w*|осво\w*|созда\w*|создан\w*|"
    r"постро\w*|модерниз\w*|нов\w*\s+технолог\w*)\b"
)
NONINDUSTRIAL_INTENT = rx(
    r"\b(?:прокомментировал\w*|заявил\w*|поздравил\w*|"
    r"назвал\w*|выразил\w*|"
    r"подписал\w*\s+соглашени\w*\s+с\s+проф\w*|"
    r"объяснил\w*|прогнозир\w*|призыв\w*|призыва\w*|"
    r"предложил\w*\s+выдавать\w*|"
    r"состо\w*\s+обсужден\w*|провел\w*\s+обучени\w*)\b"
)
FOREIGN_ONLY = rx(
    r"(?:^naval group\b|^saab\b|^финлянди\w*\b|"
    r"^в\s+(?:южной\s+корее|бангладеш|ираке|японии|сша|франции)\b|"
    r"\bевропейск\w*.{0,60}(?:корвет\w*|эсмин\w*|фрегат\w*)|"
    r"\bдатск\w*\s+флот\w*|\bамериканск\w*\s+эсмин\w*)"
)
RUSSIAN_ACTOR = rx(
    r"\b(?:росси\w*|отечественн\w*|ростех\b|оск\b|"
    r"росатом\b|лукойл\b|норникел\w*|русал\b|"
    r"роснефт\w*|газпром\b|северстал\w*)\b"
)

# 18 exact labels from TAG_KEYWORDS in processing/tagger.py. These cues
# describe technical OBJECTS, not company names or implied customers.
CATEGORY = (
    ("судостроение",
     r"\b(?:судостроен\w*|судоверф\w*|верф\w*|"
     r"судн\w*|ледокол\w*|теплоход\w*|танкер\w*|траулер\w*|"
     r"корвет\w*|фрегат\w*|эсмин\w*|подлодк\w*|"
     r"подводн\w*\s+лодк\w*|корабл\w*|катер\w*)\b"),
    ("авиация",
     r"\b(?:самолет\w*|вертолет\w*|авиадвигател\w*|"
     r"авиационн\w*\s+двигател\w*|авиастроен\w*|"
     r"летательн\w*\s+аппарат\w*|бпла\b|"
     r"ил-114\w*|мс-21\b|sj-100\b|ssj\b|беспилотник\w*)\b"),
    ("железнодорожное машиностроение",
     r"\b(?:вагон\w*|локомотив\w*|тепловоз\w*|"
     r"электровоз\w*|электропоезд\w*|трамва\w*|"
     r"поезд\w*\s+нов\w*\s+поколен\w*|"
     r"подвижн\w*\s+состав\w*)\b"),
    ("автомобильная промышленность",
     r"\b(?:автозавод\w*|автомобил\w*|автокомпонент\w*|"
     r"автопром\w*|автобус\w*|электробус\w*|троллейбус\w*|"
     r"грузовик\w*|акпп\b|короб\w*\s+передач\w*|автобренд\w*)\b"),
    ("атомная отрасль",
     r"\b(?:атомн\w*\s+(?:станци\w*|электростанци\w*|"
     r"реактор\w*|ледокол\w*)|аэс\b|ритм-200\b|"
     r"ядерн\w*\s+(?:реактор\w*|топлив\w*))\b"),
    ("нефтегазовая промышленность",
     r"\b(?:добыч\w*\s+нефт\w*|промышленн\w*\s+"
     r"добыч\w*\s+нефт\w*|газоконденсатн\w*|"
     r"нефтегаз\w*|нефтепереработ\w*|нефтедобыч\w*|"
     r"нефтян\w*\s+скважин\w*|газов\w*\s+скважин\w*|"
     r"скважин\w*|газопровод\w*|трубопровод\w*|"
     r"добыч\w*\s+газа|спг\b|бурени\w*\s+нефт\w*|"
     r"нефт\w*\s+месторожд\w*)\b"),
    ("горнодобывающая промышленность",
     r"\b(?:рудник\w*|шахт\w*|золотодобыч\w*|"
     r"горнодобыч\w*|"
     r"добы\w*.{0,45}(?:алмаз\w*|золот\w*|руд\w*|"
     r"угл\w*|боксит\w*)|"
     r"извлек\w*.{0,45}(?:золот\w*|руд\w*|лит\w*)|"
     r"горн\w*\s+(?:работ\w*|масс\w*|пород\w*)|"
     r"карьер\w*|гок\w*)\b"),
    ("металлургия",
     r"\b(?:металлург\w*|сталеплав\w*|доменн\w*|"
     r"выплав\w*|сталелитейн\w*|литейн\w*\s+завод\w*|"
     r"сканди\w*|паллади\w*|"
     r"производ\w*.{0,45}(?:стали|алюминия|никеля|меди)|"
     r"металл\w*\s+платинов\w*\s+групп\w*)\b"),
    ("химическая промышленность",
     r"\b(?:присад\w*|полимер\w*|полипропилен\w*|"
     r"удобрени\w*|нефтехим\w*|газохим\w*|метанол\w*|"
     r"химическ\w*\s+(?:производств\w*|завод\w*|реактор\w*))\b"),
    ("электротехническое направление",
     r"\b(?:радиоэлектрон\w*|микросхем\w*|sim-карт\w*|"
     r"sim-чип\w*|полупроводник\w*|трансформатор\w*|"
     r"предусилител\w*|датчик\w*|электродвигател\w*|"
     r"электробатаре\w*|аккумулятор\w*|"
     r"электронн\w*\s+(?:компонент\w*|плат\w*))\b"),
    ("машиностроение",
     r"\b(?:станок\b|станк\w*|машиностроен\w*|"
     r"насос\w*|компрессор\w*|турбин\w*|"
     r"грузоподъемн\w*\s+техник\w*|земснаряд\w*|"
     r"кран\w*\s+(?:портальн\w*|контейнерн\w*)|"
     r"аппарат\w*\s+для\s+наркоз\w*|"
     r"оборудовани\w*\s+для\s+обработк\w*)\b"),
    ("оборонная промышленность",
     r"\b(?:вооружени\w*|бронетехник\w*|"
     r"ракетн\w*\s+комплекс\w*|боев\w*\s+модул\w*|"
     r"зенитн\w*\s+(?:боев\w*|ракет\w*|комплекс\w*)|"
     r"военн\w*\s+(?:техник\w*|оборудовани\w*))\b"),
    ("ТЭК",
     r"\b(?:электростанци\w*|теплоэлектростанци\w*|"
     r"грэс\b|тэц\b|энергоблок\w*|электрогенераци\w*|"
     r"теплоэнергет\w*|электросет\w*)\b"),
    ("технологии",
     r"\b(?:искусственн\w*\s+интеллект\w*|"
     r"ии\s+созда\w*|цифров\w*\s+технолог\w*|"
     r"видеоаналитик\w*|компьютерн\w*\s+зрен\w*|"
     r"3d-сканер\w*|3d-печат\w*|"
     r"цифров\w*\s+планировани\w*|автоматизаци\w*|алгоритм\w*)\b"),
    ("инжиниринг",
     r"\b(?:инжиниринг\w*|инженерн\w*\s+"
     r"(?:проект\w*|разработк\w*)|проектировани\w*|"
     r"конструкторск\w*\s+разработк\w*)\b"),
    ("инновации",
     r"\b(?:патент\w*|запатентовал\w*|"
     r"инновационн\w*\s+"
     r"(?:материал\w*|разработк\w*|технологи\w*))\b"),
)
COMPILED = tuple((label, rx(pattern)) for label, pattern in CATEGORY)
GENERIC_PRODUCTION = rx(
    r"\b(?:производств\w*|выпуск\w*|"
    r"производственн\w*\s+лини\w*|"
    r"серийн\w*\s+производств\w*|"
    r"технологическ\w*\s+процесс\w*)\b"
)
FACTORY_LAUNCH = rx(
    r"\b(?:завод\w*|фабрик\w*|предприяти\w*)"
    r".{0,80}\b(?:запуст\w*|восстан\w*|"
    r"реконстру\w*|постро\w*|откры\w*)\b|"
    r"\b(?:запуст\w*|запущ\w*|восстан\w*|"
    r"постро\w*|откр\w*)\b.{0,80}"
    r"\b(?:завод\w*|фабрик\w*|предприяти\w*)\b"
)


def _negative_reason(title: str) -> str | None:
    if rx(r"\bобсудят\b|\bобсудили\b").search(title):
        return "conference_about_industry"
    if re.match(
        r"^(?:как\s+|проверить\b|"
        r"(?:компания|предприятие)\s+.{0,60}лидер\s+рынка)",
        title,
    ):
        return "howto_or_advertisement"
    for reason, pattern in NOT_INDUSTRIAL:
        if not pattern.search(title):
            continue
        if reason == "publicity" and DEFINITE_DELIVERABLE.search(title):
            continue
        return reason
    if NONINDUSTRIAL_INTENT.search(title):
        return "nonindustrial_main_event"
    return None


def _evidence(title: str, lead: str) -> dict[str, tuple[str, str]]:
    supported = {}
    for label, pattern in COMPILED:
        match = pattern.search(title)
        if match:
            supported[label] = ("title", match.group())
    if GENERIC_PRODUCTION.search(title):
        supported.setdefault(
            "технологическое производство", ("title", "производство")
        )
    if re.search(r"\bавтоматиз\w*.{0,65}лини\w*", title):
        supported.setdefault(
            "технологическое производство", ("title", "автоматизация линии")
        )
    if re.search(r"\b(?:упростил\w*|оптимизир\w*).{0,65}(?:контрол\w*|процесс\w*)", title):
        supported.setdefault("технологии", ("title", "оптимизация процесса"))
    if FACTORY_LAUNCH.search(title):
        supported.setdefault("заводы", ("title", "завод/запуск"))

    # Background words in the body cannot add extra unrelated tags.
    # The lead is a fallback only when the headline does not identify a
    # domain. This also resolves shortened titles ("реактор на Чукотке").
    if not supported:
        for label, pattern in COMPILED:
            match = pattern.search(lead)
            if match:
                supported[label] = ("lead", match.group())
        if (
            GENERIC_PRODUCTION.search(lead)
            and not re.search(r"\bпредставил\w*\s+разработк\w*", title)
        ):
            supported.setdefault(
                "технологическое производство", ("lead", "производство")
            )
        if FACTORY_LAUNCH.search(lead):
            supported.setdefault("заводы", ("lead", "завод/запуск"))
    return supported


def select_tags(
    title: str,
    summary: str = "",
    *,
    ranked: list[dict] | None = None,
    max_tags: int = 2,
    text: str = "",
) -> TagDecision:
    """Assign at most two well-supported event/industry categories.

    Use the supplied summary to disambiguate an elliptical title. If there
    is no summary, inspect only the opening 550 characters of the article.
    No DB writes, remote inference, or external API calls occur here.
    """
    if max_tags < 1:
        raise ValueError("max_tags должен быть >= 1")
    headline = norm(title)
    if not headline:
        return TagDecision((), "review", "empty_title")
    if FOREIGN_ONLY.search(headline) and not RUSSIAN_ACTOR.search(headline):
        return TagDecision((), "reject", "foreign_project_without_russia")
    neg = _negative_reason(headline)
    if neg:
        return TagDecision((), "reject", neg)
    if (
        re.search(
            r"\b(?:хотят|планируют|могут|предлагают|пытались)\b", headline
        )
        and not re.search(
            r"\b(?:запустил|запущен|выпустил|создал|разработал)\w*\b",
            headline,
        )
    ):
        return TagDecision((), "review", "plan_not_completed")
    if re.search(
        r"\b(?:меморандум|договорились|планиру\w*|намерен\w*)\b",
        headline,
    ):
        return TagDecision((), "review", "only_agreement_or_plan")
    if re.search(
        r"\b(?:порт|терминал)\w*.{0,70}"
        r"(?:запустит|поставит)\w*.{0,35}\bкран\w*",
        headline,
    ):
        return TagDecision((), "review", "port_equipment_operations")
    if not STRONG_EVENT.search(headline):
        return TagDecision((), "review", "no_industrial_event")

    lead = norm(summary)[:550] if summary else norm(text)[:550]
    supported = _evidence(headline, lead)

    if re.search(r"\b(?:воссозда\w*|воссозд\w*)", headline):
        supported.setdefault("технологии", ("title", "обратное проектирование"))

    # Role constraints: end customers, machines USED by a plant, and
    # coincidental company/product terms do not imply another industry.
    if (
        "электротехническое направление" in supported
        and re.search(
            r"\bдатчик\w*\b.{0,50}\bдля\s+химическ\w*\s+реактор\w*",
            headline,
        )
    ):
        supported.pop("химическая промышленность", None)
    if re.search(
        r"\b(?:электрическ\w*|электродвигател\w*)"
        r"\s+(?:речн\w*\s+)?судн\w*",
        headline,
    ):
        supported.pop("электротехническое направление", None)
    if re.search(
        r"\b(?:вагоноопрокидывател\w*|погрузк\w*\s+угл\w*)\b",
        headline,
    ):
        supported.pop("железнодорожное машиностроение", None)
    if re.search(r"\b(?:присад\w*|моторн\w*\s+масл\w*)\b", headline):
        supported.pop("нефтегазовая промышленность", None)
    if re.search(
        r"\b(?:добыч\w*\s+нефт\w*|скважин\w*|месторожд\w*\s+газа)\b",
        headline,
    ):
        supported.pop("горнодобывающая промышленность", None)
    if (
        "электротехническое направление" in supported
        and re.search(
            r"\b(?:алюминиев\w*\s+детал\w*|для\s+радиоэлектрон\w*)",
            headline,
        )
    ):
        supported.pop("электротехническое направление", None)

    # Nuclear reactor of a ship: "Чукотка" alone is not self-explanatory.
    if (
        re.search(r"\bреактор\w*", headline)
        and re.search(r"\b(?:атомн\w*|ледокол\w*|ритм-200)\b", lead)
    ):
        supported["атомная отрасль"] = ("lead", "атомный реактор")

    if re.search(
        r"\b(?:производств\w*\s+электробатарей|"
        r"кластер\s+электробатарей)\b",
        headline,
    ):
        supported["электротехническое направление"] = (
            "title", "электробатареи"
        )
        supported.setdefault("заводы", ("title", "промышленный кластер"))

    if not supported:
        return TagDecision((), "review", "industry_not_identified")

    def evidence_weight(label: str) -> float:
        base = 3.0 if supported[label][0] == "title" else 1.0
        if label in ("инновации", "инжиниринг", "технологии"):
            base -= 0.25
        if label == "технологическое производство":
            base -= 0.30
        if label == "заводы":
            base -= 0.15
        if label in (
            "авиация", "судостроение", "машиностроение", "атомная отрасль"
        ):
            base += 0.05
        return base

    scores: dict[str, float] = {}
    for row in ranked or ():
        if not isinstance(row, dict):
            continue
        tag = row.get("tag")
        try:
            value = float(row.get("final_score") or 0)
        except (ValueError, TypeError):
            continue
        if math.isfinite(value):
            scores[tag] = min(0.7, max(0.0, value))
    ordered = sorted(
        supported,
        key=lambda name: (
            -evidence_weight(name) - scores.get(name, 0.0) * 0.25,
            name,
        ),
    )
    tags = tuple(ordered[:max_tags])
    return TagDecision(
        tags,
        "tagged",
        "grounded_main_event",
        tuple(
            f"{tag}: {supported[tag][0]}={supported[tag][1]}"
            for tag in tags
        ),
    )
