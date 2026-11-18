"""Stage 2 conservative, auditable technical/semantic triage.

No ML, database, network, geographical restrictions or destructive deletion.
The legacy topic_filter API remains available for older diagnostics; the
production intake path uses this tri-state contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import re
from urllib.parse import urlsplit

from processing.topic_filter import has_bad_content, has_bad_url


@dataclass(frozen=True)
class IntakeDecision:
    status: str  # accepted | review | rejected
    reason: str
    stage: str  # technical | semantic
    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def _norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold().replace("ё", "е")).strip()


def _published(value) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def technical_decision(article, *, now: date | datetime | None = None,
                       min_text_len: int = 250, max_age_days: int = 30) -> IntakeDecision:
    title = str(getattr(article, "title", "") or "").strip()
    url = str(getattr(article, "url", "") or "").strip()
    text = str(getattr(article, "text", "") or "").strip()
    if not title:
        return IntakeDecision("rejected", "missing_title", "technical")
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        return IntakeDecision("rejected", "invalid_url", "technical")
    if has_bad_url(url):
        return IntakeDecision("rejected", "navigation_page", "technical")
    if not text:
        return IntakeDecision("review", "empty_text", "technical")
    if has_bad_content(text):
        return IntakeDecision("rejected", "site_boilerplate", "technical")
    if len(text) < min_text_len:
        return IntakeDecision("review", "short_text", "technical")
    if getattr(article, "text_source", "html") != "html":
        return IntakeDecision("review", "rss_preview_only", "technical")
    published = _published(getattr(article, "published_at", None))
    if now is None:
        today = datetime.now(timezone.utc).date()
    else:
        today = now.date() if isinstance(now, datetime) else now
    if published is not None:
        if published > today + timedelta(days=1):
            return IntakeDecision("review", "future_date", "technical")
        if max_age_days > 0 and published < today - timedelta(days=max_age_days):
            return IntakeDecision("review", "stale_article", "technical")
    return IntakeDecision("accepted", "technical_ok", "technical")


# Rules match the headline, not arbitrary mentions deep inside good articles.
_ADVERTISEMENT = re.compile(
    r"\b(?:скидк\w*|кешбэк\w*|промокод\w*|распродаж\w*|розыгрыш\w*|"
    r"программа лояльности|акция для покупател\w*|выгодн\w* предложени\w*)\b")
_CONGRATULATIONS = re.compile(
    r"\b(?:поздравил\w*|поздравля\w*|поздравлени\w*|с днем рожден\w*|с юбилеем|"
    r"отмечает \d{1,3}.?(?:летие|лет)|отмечает юбилей|юбилей (?:завода|предприяти\w*|компани\w*|флот\w*)|торжеств\w* юбиле\w*)\b")
_VACANCY = re.compile(
    r"\b(?:ваканси\w*|требуются сотрудник\w*|приглашаем на работу|"
    r"открыт набор персонала|ищем специалист\w*|ярмарка вакансий|центр подбора персонала)\b")
_DIGEST = re.compile(
    r"\b(?:дайджест\w*|подборка ссылок|обзор сми|главное в отраслевых сми|"
    r"итоги недели|новости за неделю|топ.?\d+ новостей)\b")
_OFFTOPIC = re.compile(
    r"\b(?:егэ|единый государственный экзамен|обыск\w*|задержали за взятку|"
    r"уголовн\w* дел\w*|похищени\w*|гороскоп\w*|"
    r"расписание школьн\w* экзамен\w*)\b")
_CONSUMER = re.compile(
    r"\b(?:смартфон\w*|наушник\w*|косметик\w*|потребительск\w*"
    r"|айфон\w*|iphone\w*|цены на икру)\b")
_CONSUMER_ACTION = re.compile(
    r"\b(?:продаж\w*|подорожал\w*|подешевел\w*|"
    r"купить|скидк\w*|в магазин\w*|вышел в продаж\w*)\b")
_INDUSTRIAL = re.compile(
    r"\b(?:завод\w*|фабрик\w*|предприяти\w*|промышлен\w*|производств\w*|"
    r"машиностроени\w*|судостроени\w*|верф\w*|самолет\w*|авиацион\w*|"
    r"двигател\w*|металлург\w*|добыч\w*|месторождени\w*|шахт\w*|"
    r"нефт\w*|газов\w*|энергетик\w*|электростанц\w*|электросет\w*|"
    r"оборудовани\w*|станк\w*|инженер\w*|технологи\w*|комбинат\w*|цех\w*|"
    r"судн\w*|суда|судов|сухогруз\w*|краболов\w*|корабл\w*|станци\w*|линия\w*|атомн\w*|робот\w*|"
    r"руд\w*|угл\w*|скважин\w*|золот\w*|электродвигател\w*|автокран\w*|дрон\w*|бпла|беспилот\w*|фанер\w*|металл\w*|нефтехими\w*|реактор\w*|pvd|покрыти\w*|инструмент\w*|"
    r"энергопотреблен\w*|электроэнерг\w*|генераци\w*|трансформатор\w*|турбин\w*|истребител\w*|вертолет\w*)\b")
_EVENT = re.compile(
    r"\b(?:запуст\w*|ввел\w*|ввели|введ\w*|открыл\w*|создал\w*|"
    r"разработ\w*|произвел\w*|изготов\w*|испыт\w*|модернизир\w*|модернизац\w*|создан\w*|"
    r"постро\w*|инвестир\w*|влож\w*|обновил\w*|добы\w*|спуст\w*|представ\w*|отгруз\w*|отправ\w*|нараст\w*|достиг\w*|"
    r"расшир\w*|начал выпуск\w*|поставил\w*|заработал\w*|внедри\w*|увеличил\w*|выпуст\w*|выпуск\w*|сертифицир\w*|одобр\w*|"
    r"освоил\w*|контракт\w*|проект\w*|оснащ\w*|оснаст\w*|собрал\w*|сборк\w*|переработ\w*|восстанов\w*|мощност\w*|"
    r"перевыполн\w*|вывел\w* из эксплуатации|ликвидир\w*|перезапуст\w*|заверш\w*|превыс\w*|сократил\w*|снизил\w*)\b")


# Industrial organizations may anchor a specific technical/production story
# even when the title omits the generic word "factory".
_INDUSTRY_ACTORS = re.compile(
    r"\b(?:оск|одк|оак|ростех|росэл|крэ т|крет|норникел\w*|"
    r"свеза|алроса|селигдар|русолово|швабе|тмк|гмк|уралхимпласт|"
    r"лукойл|газпром|р.флот)\b"
)
# A company name alone cannot turn a social or medical story into an
# industrial event; require a material, component or technical output.
_ACTOR_TECHNICAL_OUTPUT = re.compile(
    r"\b(?:панел\w* для .{0,3}чист\w* помещени\w*|ии.лаборатори\w*|"
    r"нов\w* материал\w*|промышленн\w* издели\w*|"
    r"серийн\w* выпуск\w*|техническ\w* решени\w*|"
    r"комплектующ\w* для оборудовани\w*)\b"
)
_INDUSTRY_REPORT = re.compile(
    r"\b(?:итоги|результат\w*) производственн\w* деятельност\w*|"
    r"\bпроизводственн\w* показател\w*"
)
_CONSUMER_PRICES = re.compile(
    r"\b(?:цен\w*|рост\w* цен) на (?:красную )?икру\b"
)
_HYPOTHETICAL_ONLY = re.compile(
    r"\bмодел\w* добыч\w* .{0,50}\bастерои\w*"
)
# HR, PR and governance mentions are not proof of a production event.
_ORG_ONLY = re.compile(
    r"\b(?:обучени\w*|наставничеств\w*|стажировк\w*|олимпиад\w*|"
    r"чемпионат\w*|фестивал\w*|конкурс\w*|"
    r"преми\w*|наград\w*|призов\w* мест\w*|"
    r"совет директоров|собрани\w* акционер\w*|"
    r"акционер\w*|дивиденд\w*|мастер.класс\w*)\b"
)
_FINANCE_ONLY = re.compile(
    r"\b(?:самой дорогой компани\w*|котировк\w*|"
    r"капитализаци\w*|курс акци\w*|цены на икру)\b"
)
_CRIME_ONLY = re.compile(
    r"\b(?:осудил\w*|осужд\w*|незаконн\w* добыч\w*|"
    r"возбудил\w* уголовн\w* дел\w*)\b"
)


def semantic_decision(article) -> IntakeDecision:
    title = _norm(getattr(article, "title", ""))
    # A promotion remains a promotion even if it mentions factories/products.
    for pattern, reason in (
        (_ADVERTISEMENT, "advertisement"),
        (_CONGRATULATIONS, "congratulation"),
        (_VACANCY, "vacancy"),
        (_DIGEST, "digest"),
    ):
        if pattern.search(title):
            return IntakeDecision("rejected", reason, "semantic")
    industrial = bool(_INDUSTRIAL.search(title) or (
        _INDUSTRY_ACTORS.search(title) and _ACTOR_TECHNICAL_OUTPUT.search(title)
    ))
    event = bool(_EVENT.search(title))
    if _FINANCE_ONLY.search(title):
        return IntakeDecision("rejected", "financial_markets", "semantic")
    if _CONSUMER_PRICES.search(title):
        return IntakeDecision("rejected", "consumer_only", "semantic")
    if _HYPOTHETICAL_ONLY.search(title):
        return IntakeDecision("review", "speculative_industry", "semantic")
    if _CRIME_ONLY.search(title):
        return IntakeDecision("rejected", "offtopic_event", "semantic")
    if _ORG_ONLY.search(title):
        return IntakeDecision("review", "corporate_or_hr_event", "semantic")
    if _OFFTOPIC.search(title):
        # Industrial safety investigations can be relevant, unlike a generic raid.
        if industrial:
            return IntakeDecision("review", "offtopic_event_with_industry", "semantic")
        return IntakeDecision("rejected", "offtopic_event", "semantic")
    if _CONSUMER.search(title) and _CONSUMER_ACTION.search(title) and not (industrial and event):
        return IntakeDecision("rejected", "consumer_only", "semantic")
    if industrial and _INDUSTRY_REPORT.search(title):
        return IntakeDecision("accepted", "industrial_report", "semantic")
    if industrial and event:
        return IntakeDecision("accepted", "industrial_event", "semantic")
    if industrial:
        return IntakeDecision("review", "industry_without_clear_event", "semantic")
    # Do not conflate lack of a keyword with proof of non-industry.
    return IntakeDecision("review", "uncertain_industry", "semantic")


def classify_intake(article, *, now: date | datetime | None = None,
                    min_text_len: int = 250, max_age_days: int = 30) -> IntakeDecision:
    technical = technical_decision(article, now=now, min_text_len=min_text_len,
                                   max_age_days=max_age_days)
    if technical.status == "rejected" or technical.reason in ("empty_text", "short_text", "rss_preview_only"):
        return technical
    semantic = semantic_decision(article)
    # Obvious junk remains junk regardless of age. Valuable older stories
    # go to review instead of being dropped as off-topic.
    if semantic.status == "rejected":
        return semantic
    if technical.status == "review":
        return technical
    return semantic
