"""Event-grounded industry tagging without LLMs or network calls.

The headline identifies the event and object; the summary and subject-linked
sentences of the full article provide additional evidence. A company's name and
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
        r"\b(?:скидк\w*|распродаж\w*|урожай\w*|озим\w*|посев\w*|"
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
     r"\b(?:металлург\w*|сталеплав\w*|доменн\w*|домн\w*|"
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
    r".{0,80}\b(?:восстан\w*|"
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


# Words used to match a contextual sentence to the main headline subject.
# Never classify an industry from an arbitrary secondary paragraph.
_TOPIC_STOP = frozenset(
    "который которая которые этого этой после новый новые новые "
    "компания компании предприятия предприятия завод заводе "
    "производства производство российский российские россии "
    "проекта проект технологии развитие рамках начала работ".split()
)
_SUBJECT_EQUIVALENTS = {
    "дрон": "беспилот", "беспилот": "беспилот",
    "самолет": "самолет", "самолетов": "самолет",
    "ледокол": "ледокол", "ледокола": "ледокол",
    "судно": "судн", "судна": "судн",
}


def _topic_stems(value: str) -> set[str]:
    stems = set()
    for raw in re.findall(r"[а-яa-z0-9-]{5,}", norm(value)):
        if raw in _TOPIC_STOP:
            continue
        stems.add(_SUBJECT_EQUIVALENTS.get(raw, raw[:5]))
        if raw.startswith(("дрон", "беспилот")):
            stems.add("беспилот")
    return stems


def _sentences(value: str, *, max_chars: int = 50000) -> list[str]:
    # Inspect the complete normal-sized source article, not just its lead.
    value = str(value or "")[:max_chars].replace("\r", "\n")
    return [
        norm(chunk) for chunk in
        re.split(r"(?<=[.!?])\s+|[\n;]+", value)
        if chunk and len(chunk.strip()) >= 12
    ]


def _contextual_evidence(
    title: str,
    summary: str,
    text: str,
    *,
    title_hints: dict[str, tuple[str, str]],
) -> dict[str, tuple[float, str, str]]:
    """Rank *supported* categories from headline, summary and article.

    Summary is stronger than raw text. A raw-text sentence must be about
    the headline subject and contain an industrial action; context mentions
    on their own never create tags. The full article is examined, with
    bounded input for adversarial/very long pages.
    """
    votes = {
        tag: (
            (6.0 if tag not in ("заводы", "технологическое производство")
             else 4.0),
            source, term
        ) for tag, (source, term) in title_hints.items()
    }
    headline_topic = _topic_stems(title)
    has_concrete_title_tag = any(
        label not in ("заводы", "технологическое производство")
        for label in title_hints
    )

    def consider(segments: list[str], source: str) -> None:
        for index, piece in enumerate(segments):
            overlap = len(headline_topic & _topic_stems(piece))
            early = index <= 1
            action = bool(STRONG_EVENT.search(piece))
            if source == "text":
                # Only a sentence related to the actual main event is
                # eligible, even if it is in the middle/end of the article.
                if not action or overlap < 1:
                    continue
            elif not action and not early:
                continue
            for tag, pattern in COMPILED:
                match = pattern.search(piece)
                if not match:
                    continue
                if tag not in title_hints and has_concrete_title_tag:
                    # Avoid unrelated second sectors from a company's
                    # profile or later paragraphs.
                    continue
                if source == "text" and not overlap and not early:
                    continue
                strength = (
                    (3.5 if action else 2.5) if source == "summary"
                    else (2.1 if early and overlap else 1.6)
                )
                if overlap:
                    strength += min(overlap, 2) * 0.3
                if tag in votes:
                    # Corroboration affects only the confidence of a tag,
                    # never makes it outrank a stronger headline sector.
                    prev, src, word = votes[tag]
                    votes[tag] = (
                        max(prev, strength) + min(strength * 0.06, 0.20),
                        src, word
                    )
                else:
                    votes[tag] = (strength, source, match.group(0))
            if (
                source == "summary"
                and not title_hints
                and action and GENERIC_PRODUCTION.search(piece)
            ):
                votes.setdefault(
                    "технологическое производство",
                    (2.3, "summary", "производство"),
                )

    consider(_sentences(summary, max_chars=8000), "summary")
    consider(_sentences(text), "text")
    return votes


def select_tags(
    title: str,
    summary: str = "",
    *,
    ranked: list[dict] | None = None,
    max_tags: int = 2,
    text: str = "",
) -> TagDecision:
    """Offline tags from main event + summary + subject-related full text.

    Geographic origin does not influence the labels. SBERT can break ties
    only among labels that already have textual evidence.
    """
    if max_tags < 1:
        raise ValueError("max_tags должен быть >= 1")
    headline = norm(title)
    if not headline:
        return TagDecision((), "review", "empty_title")
    reason = _negative_reason(headline)
    if reason:
        return TagDecision((), "reject", reason)

    # A proposed industrial investment or signed memorandum is not
    # equivalent to a factory already being built or commissioned.
    if re.search(
        r"\b(?:подписал\w*.{0,65}соглашени\w*|"
        r"соглашени\w*.{0,70}строительств\w*|"
        r"меморандум\w*|договорились|планиру\w*|"
        r"намерен\w*|хотят|могут|предлагают)\b",
        headline,
    ) and not re.search(
        r"\b(?:уже\s+запуст\w*|построил\w*|"
        r"начал\w*\s+(?:промышленн\w*\s+)?добыч\w*)\b", headline
    ):
        return TagDecision((), "review", "agreement_or_plan")

    if re.search(
        r"\b(?:порт|терминал)\w*.{0,70}"
        r"(?:запустит|поставит)\w*.{0,35}\bкран\w*",
        headline,
    ):
        return TagDecision((), "review", "port_equipment_operations")

    title_hints = _evidence(headline, "")
    # A press release about unrelated business activity must not pick up
    # a sector from boilerplate even if the company owns factories.
    action_in_title = bool(STRONG_EVENT.search(headline))
    if not action_in_title:
        first_context = " ".join(
            _sentences(summary, max_chars=2500)[:2]
            or _sentences(text, max_chars=1300)[:2]
        )
        has_context_event = bool(STRONG_EVENT.search(first_context))
        related = bool(_topic_stems(headline) & _topic_stems(first_context))
        if not (has_context_event and related):
            return TagDecision((), "review", "no_industrial_event")
        # Do not label a pure transport/use headline as aircraft
        # manufacturing just because a drone is mentioned in the article.
        if re.search(
            r"\b(?:дрон\w*|беспилотник\w*).{0,45}"
            r"(?:следят|наблюда\w*|снима\w*|патрулиру\w*)",
            headline,
        ) and not re.search(
            r"\b(?:разработан\w*|нов\w*\s+комплекс\w*|"
            r"создан\w*\s+комплекс\w*)\b",
            first_context,
        ):
            return TagDecision((), "review", "drone_operation_not_manufacturing")

    votes = _contextual_evidence(
        headline, summary, text, title_hints=title_hints
    )
    if re.search(r"\b(?:воссозда\w*|воссозд\w*)", headline):
        votes.setdefault(
            "технологии", (4.1, "title", "обратное проектирование")
        )

    # Industry is the delivered technical object, not a consumer industry.
    if re.search(
        r"\bдатчик\w*\b.{0,50}\bдля\s+химическ\w*\s+реактор\w*",
        headline,
    ):
        votes.pop("химическая промышленность", None)
    if re.search(
        r"\b(?:электрическ\w*|электродвигател\w*)"
        r"\s+(?:речн\w*\s+)?судн\w*", headline
    ):
        votes.pop("электротехническое направление", None)
    if re.search(r"\b(?:присад\w*|моторн\w*\s+масл\w*)\b", headline):
        votes.pop("нефтегазовая промышленность", None)

    # A "pipeline system" can be water, chemicals, heat, not only oil/gas.
    if re.search(r"\bтрубопровод\w*", headline) and not re.search(
        r"\b(?:нефт\w*|газ\w*|нефтепровод\w*|газопровод\w*|"
        r"скважин\w*|спг\b)", headline + " " + norm(summary)[:500]
    ):
        votes.pop("нефтегазовая промышленность", None)

    # The primary news event can be the commissioning of a reactor on an
    # icebreaker: the ship is context, not necessarily a new-build event.
    if re.search(r"\bреактор\w*", headline):
        reactor = re.search(
            r"\b(?:атомн\w*|ядерн\w*|ритм-200|ледокол\w*)\b",
            norm(summary) + " " + norm(text)[:1500],
        )
        if reactor:
            votes["атомная отрасль"] = (
                6.4, "context", reactor.group(0)
            )
            votes.pop("судостроение", None)

    if re.search(
        r"\b(?:производств\w*\s+электробатарей|"
        r"кластер\s+электробатарей)\b", headline,
    ):
        votes["электротехническое направление"] = (
            6.0, "title", "электробатареи"
        )
        if re.search(r"\bоткр\w*\s+кластер", headline):
            votes.setdefault("заводы", (3.8, "title", "новый кластер"))

    if not votes:
        return TagDecision((), "review", "industry_not_identified")

    score_bonus = {}
    for row in ranked or ():
        if not isinstance(row, dict):
            continue
        try:
            value = float(row.get("final_score") or 0)
            if math.isfinite(value):
                score_bonus[str(row.get("tag"))] = min(
                    max(value, 0.0), 1.0
                ) * 0.2
        except (ValueError, TypeError):
            continue
    ordering = sorted(
        votes,
        key=lambda tag: (-votes[tag][0] - score_bonus.get(tag, 0.0), tag)
    )
    picked = tuple(ordering[:min(max_tags, 2)])
    return TagDecision(
        picked,
        "tagged",
        "evidence_from_title_summary_and_subject_text",
        tuple(
            f"{tag}: {votes[tag][1]}={votes[tag][2]}"
            for tag in picked
        ),
    )
