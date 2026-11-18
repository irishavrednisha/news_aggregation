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
    r"\b(?:поздравля\w*|поздравлени\w*|с днем рожден\w*|с юбилеем|"
    r"отмечает \d{1,3}.?(?:летие|лет)|торжеств\w* юбиле\w*)\b")
_VACANCY = re.compile(
    r"\b(?:ваканси\w*|требуются сотрудник\w*|приглашаем на работу|"
    r"открыт набор персонала|ищем специалист\w*|ярмарка вакансий)\b")
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
    r"оборудовани\w*|станк\w*|инженер\w*|технологи\w*|"
    r"судн\w*|корабл\w*|станци\w*|линия\w*|атомн\w*|робот\w*)\b")
_EVENT = re.compile(
    r"\b(?:запуст\w*|ввел\w*|введ\w*|открыл\w*|создал\w*|"
    r"разработал\w*|произвел\w*|испытал\w*|модернизир\w*|"
    r"постро\w*|инвестир\w*|обновил\w*|добы\w*|"
    r"расширил\w*|начал выпуск\w*|поставил\w*|"
    r"освоил\w*|контракт\w*|проект\w*|оснащ\w*)\b")


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
    industrial = bool(_INDUSTRIAL.search(title))
    event = bool(_EVENT.search(title))
    if _OFFTOPIC.search(title):
        # Industrial safety investigations can be relevant, unlike a generic raid.
        if industrial and event:
            return IntakeDecision("review", "offtopic_event_with_industry", "semantic")
        return IntakeDecision("rejected", "offtopic_event", "semantic")
    if _CONSUMER.search(title) and _CONSUMER_ACTION.search(title) and not (industrial and event):
        return IntakeDecision("rejected", "consumer_only", "semantic")
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
    if technical.status != "accepted":
        return technical
    return semantic_decision(article)
