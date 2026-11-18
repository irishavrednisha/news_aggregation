"""Conservative exact-duplicate matching, separate from semantic clustering.

Canonical URL equality, genuine identical article text, and independently
corroborated equal titles. A matching headline alone NEVER causes a merge.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
import re
import unicodedata
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

TRACKING = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "fbclid", "gclid", "yclid", "mc_cid", "mc_eid",
    "from", "ref", "referrer", "igshid",
})
MIRRORS = frozenset({"военное.рф", "отраслевое.рф", "flot.com", "flotprom.ru"})


def normalize_url(url: str) -> str:
    try:
        parts = urlsplit(str(url or "").strip())
        if parts.scheme.lower() not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            return ""
        host = parts.hostname.encode("idna").decode("ascii").lower()
        if host.startswith("www."):
            host = host[4:]
        port = parts.port
        if port and (parts.scheme.lower(), port) not in (("http", 80), ("https", 443)):
            host += f":{port}"
        path = quote(unquote(parts.path or "/"), safe="/-._~!$&'()*+,;=:@")
        if path != "/":
            path = path.rstrip("/")
        query = []
        for key, val in parse_qsl(parts.query, keep_blank_values=True):
            if key.lower() not in TRACKING and not key.lower().startswith("utm_"):
                query.append((key, val))
        return urlunsplit(("https", host, path, urlencode(sorted(query)), ""))
    except (ValueError, UnicodeError):
        return ""


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", str(title or "")).casefold().replace("ё", "е")
    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()


def normalize_text(body: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize(
        "NFKC", str(body or "")).casefold().replace("ё", "е")).strip()


def checksum(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _date(value) -> date | None:
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


def _dates_close(a, b, days=3) -> bool:
    a, b = _date(a), _date(b)
    return a is None or b is None or abs((a - b).days) <= days


def _mirror_key(url: str) -> str | None:
    c = normalize_url(url)
    parsed = urlsplit(c)
    try:
        host = parsed.hostname.encode("ascii").decode("idna")
    except (AttributeError, UnicodeError):
        return None
    path = parsed.path.casefold()
    # A generic /news or a home page is not an exact-article identifier.
    if host in MIRRORS and re.search(r"/20\d{2}/(?:\d{1,2}/){0,2}[^/]{6,}", path):
        return path + ("?" + parsed.query if parsed.query else "")
    return None


def _body_key(article) -> str | None:
    text = normalize_text(getattr(article, "text", ""))
    return checksum(text) if len(text) >= 250 else None


def _content_similarity(a, b) -> float:
    ta = set(re.findall(r"\w+", normalize_text(getattr(a, "text", ""))))
    tb = set(re.findall(r"\w+", normalize_text(getattr(b, "text", ""))))
    if len(ta) < 35 or len(tb) < 35:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _numbers(article):
    """Changes to quantities, model IDs or dates block title-based merging."""
    return sorted(re.findall(r"\d+(?:[.,]\d+)*", normalize_text(
        getattr(article, "text", "")
    )))


@dataclass(frozen=True)
class DuplicateMatch:
    news_id: int
    reason: str


def match_exact(article, candidate) -> str | None:
    url1, url2 = normalize_url(article.url), normalize_url(candidate.url)
    if url1 and url1 == url2:
        return "duplicate_url"
    t1, t2 = normalize_title(article.title), normalize_title(candidate.title)
    body1, body2 = _body_key(article), _body_key(candidate)
    if body1 and body1 == body2 and _dates_close(
        getattr(article, "published_at", None),
        getattr(candidate, "published_at", None), 30
    ):
        return "duplicate_text"
    m1, m2 = _mirror_key(article.url), _mirror_key(candidate.url)
    if m1 and m1 == m2 and _dates_close(
        getattr(article, "published_at", None),
        getattr(candidate, "published_at", None), 3
    ):
        return "duplicate_mirror"
    # Identical headline alone is not proof. Require nearly-identical body
    # and aligned publication dates, not merely a shared generic event name.
    if t1 and t1 == t2 and _dates_close(
        getattr(article, "published_at", None),
        getattr(candidate, "published_at", None), 1
    ) and _numbers(article) == _numbers(candidate) and _content_similarity(article, candidate) >= 0.97:
        return "duplicate_title_content"
    return None


class ExactDuplicateIndex:
    def __init__(self):
        self.urls = defaultdict(list)
        self.mirrors = defaultdict(list)
        self.bodies = defaultdict(list)
        self.titles = defaultdict(list)

    def add(self, article):
        url = normalize_url(article.url)
        if url:
            self.urls[url].append(article)
        mirror = _mirror_key(article.url)
        if mirror:
            self.mirrors[mirror].append(article)
        body = _body_key(article)
        if body:
            self.bodies[body].append(article)
        title = normalize_title(article.title)
        if title:
            self.titles[title].append(article)

    def find(self, article) -> DuplicateMatch | None:
        pools = (
            self.urls.get(normalize_url(article.url), ()),
            self.mirrors.get(_mirror_key(article.url), ()),
            self.bodies.get(_body_key(article), ()),
            self.titles.get(normalize_title(article.title), ()),
        )
        seen = set()
        for pool in pools:
            for candidate in pool:
                news_id = getattr(candidate, "id", None)
                if news_id is None or news_id in seen:
                    continue
                seen.add(news_id)
                reason = match_exact(article, candidate)
                if reason:
                    return DuplicateMatch(news_id, reason)
        return None
