"""Консервативный первичный фильтр промышленных новостей России.

Отбрасываем только очевидно устаревшие, служебные и непрофильные
публикации. Сложные решения оставляем кластеризации и тегированию.
Логика намеренно не использует source name или наличие одного стоп-слова
во всём тексте как доказательство нерелевантности.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from typing import NamedTuple
from urllib.parse import urlsplit
import os
import re


DEFAULT_MAX_AGE_DAYS = 30

# Служебные заголовки; не запрещаем «видео» и «анонс» внутри
# технического описания, например «система видеоаналитики».
BAD_TITLE_KEYWORDS = (
    "дайджест", "главное в отраслевых сми", "поздравляем",
    "поздравление", "итоги недели", "обзор сми",
    "фоторепортаж",
)

# Навигационные и правовые страницы, а не URL обычных новостей.
BAD_URL_PARTS = (
    "contacts", "contact", "about", "privacy", "policy",
    "personal", "governance", "career", "personnel-policy",
    "tenders", "legal", "documents", "internaldocuments",
    "contractor", "sustainability", "tag", "category", "rss", "feed",
)

# Только сочетание нескольких служебных блоков говорит о том,
# что вместо статьи мы извлекли меню/подвал сайта.
BAD_CONTENT_KEYWORDS = (
    "свидетельство о регистрации сми",
    "политика конфиденциальности",
    "согласие на обработку",
    "все права защищены",
    "личный кабинет",
    "юридическая информация",
    "на информационном ресурсе применяются рекомендательные технологии",
)

# Высокодостоверные отрицательные заголовочные признаки. Более
# неоднозначные сюжеты (выставки с новыми технологиями, санкции
# против завода, производственная отчётность) здесь НЕ отбрасываем.
OFFTOPIC_TITLE_RULES = (
    ("consumer_promotion", r"\b(?:скидк[аиуе]|кешбэк|розыгрыш|программа лояльности)\b"),
    ("consumer_promotion", r"\bна\s+азс\b.{0,85}\bакци[яию]\b"),
    ("financial_markets", r"\b(?:котировк[иае]|капитализаци[яиюи]|биржев(?:ые|ая|ой)\s+торг[иаов]|"
                          r"рейтинг\s+акций|акци[яию]\s+присвоен\s+рейтинг)\b"),
    ("financial_markets", r"\b(?:акра\s+присвоил[ао]?|самой\s+дорогой\s+компани[ейию])\b"),
    ("financial_markets", r"^акции\b.{0,120}\b(?:подешевел\w*|подорожал\w*|упал\w*|вырос\w*|рухнул\w*)\b"),
    ("corporate_governance", r"\bакционеры\b.{0,100}\b(?:устав|утвердил[иа])\b"),
    ("corporate_governance", r"\b(?:совет\s+директоров|собрание\s+акционеров|"
                             r"устав\s+компании\s+в\s+новой\s+редакции)\b"),
    ("corporate_pr", r"^(?:социальная\s+деятельность|"
                     r"благотворительная\s+деятельность)\b.{0,100}\b(?:наград|преми|отмечен)"),
    ("event_only", r"\b(?:фестиваль|битв[аы]\s+роботов|конкурс\s+красоты)\b"),
    ("event_only", r"\b(?:приглашает\s+к\s+участию|"
                   r"открыл[ась]*\s+выставк[ау]|"
                   r"стартует\s+финал\s+чемпионата)\b"),
    ("training_only", r"\b(?:прошли\s+обучени[ея]|"
                      r"провели\s+обучени[ея]|"
                      r"познакомились\s+с\s+работой\s+торгово|"
                      r"студенты\s+посетили)\b"),
    ("internal_meeting", r"\b(?:подвели\s+итоги\s+работы|"
                         r"заседание\s+правления\s+палаты)\b"),
    ("consumer_prices", r"\b(?:цен[аы]?|рост(?:ом|а|у)?\s+цен)\s+на\s+(?:красную\s+)?икру\b"),

    # Прямые признаки бытового происшествия, режима аэропортов и
    # потребительской повестки. Не блокируем слова "аэропорт",
    # "обучение", "автобус" или "кредит" сами по себе: важен тип события.
    ("traffic_incident", r"\bдтп\b.{0,110}\b(?:скорой|пострадал\w*|авари\w*)\b"),
    ("airport_operations", r"\bаэропорт\w*\b.{0,110}\b(?:приостановил\w*\s+работ\w*|принимает\s+и\s+отправляет\s+рейсы)\b"),
    ("personal_crime", r"\b(?:сексуальн\w*\s+преступлен\w*|криминалист\w*|деморализаци\w*\s+всу)\b"),
    ("consumer_finance", r"\b(?:выдача\s+ипотек\w*|портфел\w+\s+автокредит\w*|грант\w*\s+на\s+свадьб\w*)\b"),
    ("consumer_rights", r"\b(?:вернуть\s+товар\s+с\s+распродаж\w*|обменять\s+или\s+вернуть\s+товар\w*)\b"),
    ("nonindustrial_cultural_event", r"\b(?:праздничн\w*\s+акци\w*|мультсериал\w*|историческ\w*\s+драм\w*|юбилейн\w*\s+конференци\w*)\b"),
    ("nonindustrial_publicity_event", r"\b(?:бизнес\s*диалог|планшетн\w*\s+выставк\w*|олимпиад\w*\s+по\s+судостроени\w*)\b"),


)

# Не фильтруем по слову «Китай» в середине статьи о «Норникеле».
# Только явно зарубежная тема заголовка, когда во вступлении
# нет российского участника/технологии.
FOREIGN_LEAD_TITLE = re.compile(
    r"^(?:в\s+(?:бангладеш|сша|японии|индии|евросоюзе|британии|"
    r"германии|бразилии|корее|китае|австралии)\b)",
    re.IGNORECASE,
)
# Явно зарубежные действующие лица в заголовке. Учитываем
# только отсутствие российской связи во вступлении — обычные
# упоминания иностранных стран внутри статьи не запрещены.
# Явная зарубежная локация/иностранный разработчик: применяется
# ТОЛЬКО вместе с проверкой на российскую связь во вступлении.
# Иностранные проекты российских компаний должны сохраниться.
FOREIGN_EXPLICIT_LOCATION_OR_ACTOR = re.compile(
    r"(?:^в\s+южной\s+корее\b|^naval\s+group\b|^saab\b|"
    r"\bв\s+ираке\b)",
    re.IGNORECASE,
)
FOREIGN_TITLE_ACTOR = re.compile(
    r"^(?:хусит\w*|финлянд\w*|саудовск\w*|израил\w*|"
    r"иран\w*|турц\w*|япон\w*|герман\w*|китай\w*|"
    r"датск\w*|американск\w*)\b",
    re.IGNORECASE,
)
RUSSIA_CONTEXT = re.compile(
    r"\b(?:росси\w*|рф|отечественн\w*|росатом\w*|"
    r"ростех\w*|норникел\w*|лукойл\w*|газпром\w*|"
    r"гиредмет\w*|алрос[аы]\w*|свез[аы]\w*|"
    r"одк|оск|евраз\w*)\b",
    re.IGNORECASE,
)


class FilterDecision(NamedTuple):
    accepted: bool
    reason: str


def normalize(text: str) -> str:
    return str(text or "").lower().replace("ё", "е")


def _published_date(value) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            # YYYY-MM-DD and datetime strings from the diagnostic JSON.
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _max_age_setting(value: int | None) -> int:
    if value is not None:
        return max(0, int(value))
    try:
        return max(0, int(os.getenv("NEWS_MAX_AGE_DAYS", str(DEFAULT_MAX_AGE_DAYS))))
    except ValueError:
        return DEFAULT_MAX_AGE_DAYS


def has_bad_title(title: str) -> bool:
    title = normalize(title).strip()
    if any(re.search(r"\b" + re.escape(word) + r"\b", title)
           for word in BAD_TITLE_KEYWORDS):
        return True
    return _offtopic_reason(title) is not None


def _offtopic_reason(title: str) -> str | None:
    for reason, pattern in OFFTOPIC_TITLE_RULES:
        if re.search(pattern, title, flags=re.IGNORECASE):
            return reason
    return None


def has_bad_url(url: str) -> bool:
    path = urlsplit(url or "").path.lower()
    segments = [part for part in path.split("/") if part]
    return any(part in BAD_URL_PARTS for part in segments)


def has_bad_content(text: str) -> bool:
    content = normalize(text)
    return sum(phrase in content for phrase in BAD_CONTENT_KEYWORDS) >= 2


def classify_article(
    article,
    min_text_len: int = 250,
    now: datetime | date | None = None,
    max_age_days: int | None = None,
) -> FilterDecision:
    title = str(getattr(article, "title", "") or "").strip()
    url = str(getattr(article, "url", "") or "").strip()
    text = str(getattr(article, "text", "") or "").strip()
    if not title or not url:
        return FilterDecision(False, "missing_title_or_url")
    if has_bad_url(url):
        return FilterDecision(False, "navigation_page")
    if len(text) < min_text_len:
        return FilterDecision(False, "short_or_empty_text")
    if has_bad_content(text):
        return FilterDecision(False, "site_boilerplate")

    if now is None:
        today = datetime.now(timezone.utc).date()
    elif isinstance(now, datetime):
        today = now.date()
    else:
        today = now

    published = _published_date(getattr(article, "published_at", None))
    age_limit = _max_age_setting(max_age_days)
    if published is not None:
        if published > today + timedelta(days=1):
            return FilterDecision(False, "future_date")
        if age_limit and (today - published).days > age_limit:
            return FilterDecision(False, "stale_article")

    normalized_title = normalize(title)
    if any(re.search(r"\b" + re.escape(word) + r"\b", normalized_title)
           for word in BAD_TITLE_KEYWORDS):
        return FilterDecision(False, "editorial_digest_or_service")

    off_topic = _offtopic_reason(normalized_title)
    if off_topic is not None:
        return FilterDecision(False, "offtopic_" + off_topic)

    if (FOREIGN_LEAD_TITLE.search(normalized_title)
            or FOREIGN_TITLE_ACTOR.search(normalized_title)
            or FOREIGN_EXPLICIT_LOCATION_OR_ACTOR.search(normalized_title)):
        context = normalize(title + " " + text[:650])
        if not RUSSIA_CONTEXT.search(context):
            return FilterDecision(False, "foreign_without_russian_connection")

    if published is None:
        return FilterDecision(True, "accepted_undated")
    return FilterDecision(True, "accepted")


def is_valid_article(
    article, min_text_len: int = 250, *,
    now: datetime | date | None = None,
    max_age_days: int | None = None,
) -> bool:
    return classify_article(
        article, min_text_len=min_text_len, now=now,
        max_age_days=max_age_days
    ).accepted


def filter_articles(
    articles: list, *, now: datetime | date | None = None,
    max_age_days: int | None = None,
) -> list:
    return [
        article for article in articles
        if classify_article(
            article, now=now, max_age_days=max_age_days
        ).accepted
    ]


def print_filter_report(
    articles: list, filtered: list, *,
    now: datetime | date | None = None,
    max_age_days: int | None = None,
):
    reasons = Counter(
        classify_article(a, now=now, max_age_days=max_age_days).reason
        for a in articles
    )
    print("\n" + "=" * 100)
    print("Результаты первичной фильтрации")
    print(f"Всего собрано: {len(articles)}")
    print(f"Оставлено: {len(filtered)}")
    print(f"Удалено: {len(articles) - len(filtered)}")
    print(f"Максимальный возраст статьи: {_max_age_setting(max_age_days)} дней "
          "(0 = проверка выключена)")
    for reason, count in reasons.most_common():
        print(f"  {reason}: {count}")

    print("\nОСТАВЛЕННЫЕ НОВОСТИ:")
    for a in filtered:
        print(f"[OK] {a.source}: {a.title}")

    print("\nУДАЛЁННЫЕ МАТЕРИАЛЫ:")
    for a in articles:
        if a not in filtered:
            reason = classify_article(a, now=now, max_age_days=max_age_days).reason
            print(f"[NO/{reason}] {a.source}: {a.title}")
            print(a.url)
