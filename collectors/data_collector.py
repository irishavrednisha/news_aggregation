import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import time
import os
from contextvars import ContextVar
from datetime import datetime, timezone
import feedparser
# На Windows Requests обычно использует certifi вместо доверенных
# сертификатов Windows. truststore (если установлен) позволяет безопасно
# проверять HTTPS по системному хранилищу. Проверку TLS не отключаем.
if os.name == "nt":
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass

import requests
import re
import urllib3

from bs4 import BeautifulSoup
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse, urljoin
from playwright.sync_api import sync_playwright

from utils.text_utils import (
    clean_text,
    normalize_title,
    normalize_article_text,
    filter_paragraphs,
)





@dataclass
class Article:
    source: str
    title: str
    url: str
    published_at: object
    text: str
    text_source: str


@dataclass
class ArticlePreview:
    source: str
    title: str
    url: str
    published_at: object
    preview_text: str


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


# =========================
# Общие вспомогательные функции
# =========================

def parse_datetime(value):
    if not value:
        return None

    try:
        return parsedate_to_datetime(value)
    except Exception:
        return None


def get_domain(url: str) -> str:
    domain = urlparse(url).netloc.replace("www.", "").lower()

    try:
        domain = domain.encode("ascii").decode("idna")
    except Exception:
        pass

    return domain


def get_path(url: str) -> str:
    return urlparse(url).path.rstrip("/")


def same_domain(url_1: str, url_2: str) -> bool:
    return get_domain(url_1) == get_domain(url_2)


def extract_rss_text(entry) -> str:
    if entry.get("summary"):
        return normalize_article_text(entry.get("summary"))

    if entry.get("description"):
        return normalize_article_text(entry.get("description"))

    if entry.get("content"):
        content = entry.get("content")

        if isinstance(content, list) and content:
            return normalize_article_text(content[0].get("value", ""))

    return ""


# =========================
# Загрузка страниц
# =========================

def _request_ca_bundle():
    """Явно доверенная цепочка CA или системные корни. Никаких verify=False."""
    bundle = os.environ.get("NEWS_CA_BUNDLE", "").strip()
    if not bundle:
        return True
    if not Path(bundle).is_file():
        print(f"Не найден NEWS_CA_BUNDLE: {bundle}")
        return True
    return bundle


# Явная имитация «всё равно перейти» ТОЛЬКО для локального read-only
# тестового сборщика и только двух известных официальных доменов.
# По умолчанию TLS проверяется. В production флаг не передаётся.
_ALLOW_UNVERIFIED_TLS = ContextVar("news_allow_unverified_tls", default=False)


def _may_ignore_https_errors(url: str) -> bool:
    from urllib.parse import urlsplit
    host = (urlsplit(url).hostname or "").lower()
    return (
        _ALLOW_UNVERIFIED_TLS.get()
        and host in {
            "rostec.ru", "www.rostec.ru",
            "rosneft.ru", "www.rosneft.ru", "limited.rosneft.ru",
        }
        and urlsplit(url).scheme == "https"
    )


# Каждый источник получает отдельный журнал загрузки. ContextVar безопасен
# для параллельных задач планировщика и не меняет API парсеров.
_FETCH_AUDIT = ContextVar("news_collector_fetch_audit", default=None)


def _record_fetch_problem(url: str, kind: str, detail: str = "") -> None:
    stat = _FETCH_AUDIT.get()
    if stat is None:
        return
    failures = stat.setdefault("fetch_failures", [])
    if len(failures) < 100:
        failures.append({
            "url": url,
            "kind": kind,
            "detail": str(detail)[:350],
        })


def fetch_html_with_requests(url: str) -> str:
    from requests.exceptions import (
        ConnectionError as RequestConnectionError, ReadTimeout, SSLError,
        RequestException,
    )
    from urllib.parse import urlsplit

    domain = urlsplit(url).hostname or ""
    # Медленный промпортал: два ограниченных по времени запроса.
    attempts = 2 if domain in ("promvest.info", "www.promvest.info") else 1
    verify = _request_ca_bundle()

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                url, headers=HEADERS,
                timeout=(10, 35 if domain.endswith("promvest.info") else 30),
                verify=verify,
            )
            if response.status_code in (403, 429):
                print(f"HTTP {response.status_code} (запрет/лимит): {url}")
                _record_fetch_problem(url, "http_denied", str(response.status_code))
                return ""
            try:
                response.raise_for_status()
            except RequestException as exc:
                _record_fetch_problem(url, "http_error", str(exc))
                raise
            if response.encoding is None or response.encoding.lower() in ("iso-8859-1", "latin-1"):
                response.encoding = response.apparent_encoding
            return response.text
        except SSLError as exc:
            print(f"Ошибка проверки HTTPS-сертификата {url}: {exc}")
            print(
                "Не отключайте TLS: установите truststore "
                "(python -m pip install truststore) или настройте NEWS_CA_BUNDLE."
            )
            _record_fetch_problem(url, "ssl_certificate_error", str(exc))
            return ""
        except (ReadTimeout, RequestConnectionError) as exc:
            print(f"Сетевой сбой {url}, попытка {attempt}/{attempts}: {type(exc).__name__}: {exc}")
            _record_fetch_problem(url, "connection_or_timeout", f"{type(exc).__name__}: {exc}")
            if attempt == attempts:
                return ""
            time.sleep(1)
        except RequestException as exc:
            print(f"Ошибка HTTP {url}: {type(exc).__name__}: {exc}")
            _record_fetch_problem(url, "http_request_exception", f"{type(exc).__name__}: {exc}")
            return ""
    return ""


def fetch_html_with_browser(url: str, ignore_https_errors: bool = False) -> str:
    # Даже если вызывающий код ошибочно передал True, разрешён
    # строго явный opt-in для конкретных хостов, не для других сайтов.
    bypass = ignore_https_errors and _may_ignore_https_errors(url)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                if bypass:
                    print(
                        f"ВНИМАНИЕ: HTTPS-сертификат НЕ ПРОВЕРЯЕТСЯ "
                        f"для {url} (только тестовый сбор)"
                    )
                    audit = _FETCH_AUDIT.get()
                    if audit is not None:
                        audit.setdefault("unverified_tls_urls", []).append(url)
                    context = browser.new_context(
                        user_agent=HEADERS["User-Agent"],
                        viewport={"width": 1280, "height": 900},
                        ignore_https_errors=True,
                    )
                    try:
                        page = context.new_page()
                        page.goto(url, wait_until="domcontentloaded", timeout=45000)
                        page.wait_for_timeout(1500)
                        return page.content()
                    finally:
                        context.close()
                else:
                    page = browser.new_page(
                        user_agent=HEADERS["User-Agent"],
                        viewport={"width": 1280, "height": 900},
                    )
                    page.goto(url, wait_until="domcontentloaded", timeout=45000)
                    page.wait_for_timeout(1500)
                    return page.content()
            finally:
                browser.close()
    except Exception as exc:
        print(f"Браузер не смог загрузить {url}: {type(exc).__name__}: {exc}")
        _record_fetch_problem(url, "browser_error", f"{type(exc).__name__}: {exc}")
        return ""


def _rosneft_limited_url(url: str) -> str | None:
    """Запасной официальный облегчённый хост с теми же путями /press/news/."""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(url)
    if (parts.hostname or "").lower() not in ("www.rosneft.ru", "rosneft.ru"):
        return None
    if not parts.path.startswith("/press/news/"):
        return None
    return urlunsplit(("https", "limited.rosneft.ru", parts.path, parts.query, ""))


def fetch_html(url: str) -> str:
    domain = get_domain(url)

    # Временная альтернатива нажатию «Всё равно перейти» в браузере.
    # Не используем requests(verify=False), чтобы этот обход не
    # наследовался переходами на чужие домены.
    if _may_ignore_https_errors(url):
        html = fetch_html_with_browser(url, ignore_https_errors=True)
        if html:
            return html
        mirror = _rosneft_limited_url(url)
        if mirror and mirror != url and _may_ignore_https_errors(mirror):
            return fetch_html_with_browser(mirror, ignore_https_errors=True)
        return ""

    browser_domains = (
        "nornickel.ru", "заводы.рф", "xn--80az8a.xn--p1ai",
        "производства.рф", "xn--80adahnf5bdekrm.xn--p1ai",
        "tpprf.ru", "news.tpprf.ru",
    )

    if any(domain_name in domain for domain_name in browser_domains):
        html = fetch_html_with_browser(url)
    else:
        html = fetch_html_with_requests(url)
        if not html:
            html = fetch_html_with_browser(url)

    # РИА и ТАСС иногда возвращают HTTP 200 с пустой JS-оболочкой
    # вместо текста статьи. Не считаем это успешно загруженной новостью.
    if html and domain in ("tass.ru", "ria.ru"):
        text_len = len(BeautifulSoup(html, "html.parser").get_text(" ", strip=True))
        if text_len < 120:
            print(f"{domain}: пустая HTML-оболочка ({text_len} символов), пробуем Chromium")
            rendered = fetch_html_with_browser(url)
            rendered_len = (
                len(BeautifulSoup(rendered, "html.parser").get_text(" ", strip=True))
                if rendered else 0
            )
            if rendered_len >= 250:
                html = rendered
            else:
                _record_fetch_problem(
                    url, "site_shell_or_access_denied",
                    f"requests_text={text_len}, chromium_text={rendered_len}",
                )
                return ""

    if html:
        if "Error establishing a database connection" in html:
            _record_fetch_problem(url, "origin_database_error", "Сайт не может подключиться к собственной базе данных")
        return html

    # Официальная мобильная/облегчённая версия Роснефти. Сохраняем
    # исходный URL новости — меняется только адрес загрузки HTML.
    mirror = _rosneft_limited_url(url)
    if mirror:
        print(f"Пробуем облегчённый официальный сайт Роснефти: {mirror}")
        html = fetch_html_with_requests(mirror)
        if not html:
            html = fetch_html_with_browser(mirror)
        if html:
            return html

    return ""


# =========================
# Обрезка хвостов похожих новостей
# =========================

def cut_related_news(text: str) -> str:
    markers = [
        "Главное в отраслевых СМИ",
        "Читайте также",
        "Другие материалы",
        "Материалы по теме",
        "Новости по теме",
        "Еще по теме",
        "Ещё по теме",
        "Смотрите также",
        "Популярное",
        "Последние новости",
        "Ранее сообщалось",
        "Подписывайтесь",
        "Печатная версия статьи",
        "Печатная версия",
        "Фотогалерея",
    ]

    lowered = text.lower()
    positions = []

    for marker in markers:
        pos = lowered.find(marker.lower())

        if pos != -1:
            positions.append(pos)

    if positions:
        text = text[:min(positions)]

    return text.strip()


# =========================
# Извлечение текста статьи
# =========================

def extract_by_selectors(
    soup: BeautifulSoup,
    selectors: list[str],
    min_len: int = 250,
    title: str | None = None
) -> str:
    """
    Универсальный fallback для старых селекторов «article p»:
    если статьи содержат списки, извлекаем их вместе с абзацами
    из того же контейнера, не теряя <li> между <p>.
    """
    for selector in selectors:
        blocks = soup.select(selector)
        if not blocks:
            continue

        # Большинство старых экстракторов выбирают лишь p.
        # Поднимаемся ТОЛЬКО до явно выбранного селектором
        # article/main/.entry-content, а не до всей страницы.
        if selector.endswith(" p"):
            container_selector = selector[:-2]
            for container in soup.select(container_selector)[:5]:
                if not container.select("ul li, ol li"):
                    continue
                structured = _article_paragraphs(container, title)
                if len(structured) >= min_len:
                    return structured

        paragraphs = [
            block.get_text(" ", strip=True)
            for block in blocks
        ]
        paragraphs = filter_paragraphs(paragraphs, title=title)
        result = normalize_article_text(" ".join(paragraphs), title=title)
        result = cut_related_news(result)
        if len(result) >= min_len:
            return result

    return ""


def extract_text_generic(soup: BeautifulSoup, title: str | None = None) -> str:
    selectors = [
        "article p",
        "main p",
        ".entry-content p",
        ".post-content p",
        ".article p",
        ".news p",
        ".content p",
        "div[itemprop='articleBody'] p",
        "div[class*='content'] p",
        "div[class*='text'] p",
        "div[class*='article'] p",
    ]

    return extract_by_selectors(soup, selectors, min_len=250, title=title)


def extract_text_nedradv(soup: BeautifulSoup, title: str | None = None) -> str:
    """
    NedraDV: абзацы только ПОСЛЕ H1 статьи и ДО секции «Читайте также».
    Не читаем все p на странице: там находятся чужие новости и аукционы.
    """
    h1 = soup.find("h1")
    if h1 is None:
        return ""

    stop_headings = (
        "читайте также", "последние аукционы", "последние публикации",
        "комментарии", "страницы сайта", "наши контакты", "теги:",
    )
    paragraphs, seen = [], set()

    for node in h1.next_elements:
        if not getattr(node, "name", None):
            continue

        name = node.name.lower()
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            heading = clean_text(node.get_text(" ", strip=True)).lower()
            if any(heading.startswith(stop) for stop in stop_headings):
                break

        if name not in ("p", "blockquote", "li"):
            continue
        # Списки технических характеристик NedraDV также часть статьи.
        # Вложенные p не дублируем.
        if name == "p" and node.find_parent(["blockquote", "li"]):
            continue
        if name == "li" and node.find_parent(["nav", "aside", "footer"]):
            continue
        if name == "li":
            item = BeautifulSoup(str(node), "html.parser")
            for nested in item.select("ul, ol"):
                nested.decompose()
            value = clean_text(item.get_text(" ", strip=True))
        else:
            value = clean_text(node.get_text(" ", strip=True))
        lowered = value.lower()
        if any(lowered.startswith(stop) for stop in stop_headings):
            break
        if len(value) < (4 if name == "li" else 30) or (title and value.lower() == title.lower()):
            continue
        if any(x in lowered for x in ("подписаться", "поделиться", "реклама", "cackle")):
            continue
        if value not in seen:
            seen.add(value)
            if name == "li" and value[-1] not in ".!?;:":
                value += "."
            paragraphs.append(value)

    return clean_text(" ".join(paragraphs))

def extract_text_tpprf(soup: BeautifulSoup, title: str | None = None) -> str:
    """
    Извлекает текст новости с сайта ТПП РФ.

    Для этого источника нельзя брать div, потому что в них часто
    попадают фотогалерея, дата, город и повтор начала статьи.
    Берём только абзацы p из основного блока новости.
    """

    # Удаляем служебные блоки до извлечения текста
    bad_selectors = [
        ".gallery",
        ".photo-gallery",
        ".photogallery",
        ".news-gallery",
        ".slider",
        ".slick-slider",
        ".breadcrumbs",
        ".breadcrumb",
        ".share",
        ".social",
        ".print",
        ".subscribe",
        ".tags",
    ]

    for selector in bad_selectors:
        for block in soup.select(selector):
            block.decompose()

    selectors = [
        "article",
        ".news-detail",
        ".news-detail__text",
        ".news-detail-text",
        ".article__text",
        ".article-text",
        ".main-content",
        "main",
    ]

    content_block = None

    for selector in selectors:
        block = soup.select_one(selector)

        if block:
            content_block = block
            break

    if content_block is None:
        content_block = soup

    paragraphs = []

    # ВАЖНО: только p, без div
    for p in content_block.find_all("p"):
        text = clean_text(p.get_text(" ", strip=True))

        if not text:
            continue

        if len(text) < 40:
            continue

        lower = text.lower()

        bad_fragments = [
            "поделиться",
            "версия для печати",
            "читайте также",
            "торгово-промышленная палата российской федерации",
            "все права защищены",
            "контактная информация",
            "подписаться",
            "фотогалерея",
            "назад к списку",
        ]

        if any(fragment in lower for fragment in bad_fragments):
            continue

        # Убираем строки вида: Казань, 14 мая 2026 г.
        if re.match(
            r"^[А-ЯЁA-Z][а-яёa-zA-Z\- ]+,\s*\d{1,2}\s+[а-яё]+\s+\d{4}\s*г\.?$",
            text
        ):
            continue

        # Если заголовок случайно попал в текст, не добавляем его
        if title and clean_text(title).lower() == text.lower():
            continue

        paragraphs.append(text)

    # Убираем полные дубли абзацев
    cleaned = []
    seen = set()

    for paragraph in paragraphs:
        key = paragraph.lower()

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(paragraph)

    article_text = " ".join(cleaned)

    # Если слово "Фотогалерея" всё же попало внутрь склеенного текста,
    # обрезаем всё после него
    article_text = re.split(
        r"\bФотогалерея\b",
        article_text,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    # Убираем город и дату, если они попали в середину текста
    article_text = re.sub(
        r"\b[А-ЯЁ][а-яё\- ]+,\s*\d{1,2}\s+[а-яё]+\s+\d{4}\s*г\.",
        " ",
        article_text
    )

    article_text = normalize_article_text(article_text, title=title)
    article_text = cut_related_news(article_text)

    return article_text

def extract_text_proizvodstva(soup: BeautifulSoup, title: str | None = None) -> str:
    selectors = [
        "article p",
        ".post p",
        ".post-content p",
        ".entry-content p",
        ".wp-block-post-content p",
        ".single-post p",
        ".article p",
        ".article-content p",
        "main p",
        "div[class*='post'] p",
        "div[class*='article'] p",
        "div[class*='content'] p",
        "div[class*='text'] p",
    ]

    text = extract_by_selectors(soup, selectors, min_len=150, title=title)

    if text:
        return text

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
    ]

    paragraphs = filter_paragraphs(paragraphs, title=title)
    text = normalize_article_text(" ".join(paragraphs), title=title)
    text = cut_related_news(text)

    if len(text) >= 120:
        return text

    return ""


def extract_text_zavody(soup: BeautifulSoup, title: str | None = None) -> str:
    selectors = [
        "article p",
        ".publication p",
        ".publication-content p",
        ".article-content p",
        ".post-content p",
        ".entry-content p",
        "main p",
        "div[class*='publication'] p",
        "div[class*='article'] p",
        "div[class*='content'] p",
        "div[class*='text'] p",
    ]

    text = extract_by_selectors(soup, selectors, min_len=150, title=title)

    if text:
        return text

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
    ]

    paragraphs = filter_paragraphs(paragraphs, title=title)
    text = normalize_article_text(" ".join(paragraphs), title=title)
    text = cut_related_news(text)

    if len(text) >= 120:
        return text

    return ""


def extract_text_rosatom(soup: BeautifulSoup, title: str | None = None) -> str:
    selectors = [
        ".news-detail",
        ".detail_text",
        ".detail-text",
        ".news-detail__text",
        ".content",
        ".page-content",
        ".main-content",
        "main",
        "article",
    ]

    for selector in selectors:
        block = soup.select_one(selector)

        if not block:
            continue

        paragraphs = [
            p.get_text(" ", strip=True)
            for p in block.find_all("p")
        ]

        if paragraphs:
            paragraphs = filter_paragraphs(paragraphs, title=title)
            text = " ".join(paragraphs)
        else:
            text = block.get_text(" ", strip=True)

        text = normalize_article_text(text, title=title)
        text = cut_related_news(text)

        if len(text) >= 250:
            return text

    return ""


def extract_text_military_media(soup: BeautifulSoup, title: str | None = None) -> str:
    selectors = [
        "article p",
        "main p",
        ".article p",
        ".article_text p",
        ".article-text p",
        ".news_text p",
        ".news-text p",
        ".material p",
        ".material-text p",
        ".content p",
        ".text p",
        "div[itemprop='articleBody'] p",
        "div[class*='article'] p",
        "div[class*='material'] p",
        "div[class*='content'] p",
        "div[class*='text'] p",
    ]

    text = extract_by_selectors(
        soup=soup,
        selectors=selectors,
        min_len=120,
        title=title
    )

    if text:
        return text

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
    ]

    paragraphs = filter_paragraphs(paragraphs, title=title)
    text = normalize_article_text(" ".join(paragraphs), title=title)
    text = cut_related_news(text)

    if len(text) >= 120:
        return text

    return ""

def extract_article_title(url: str, soup: BeautifulSoup, fallback_title: str | None = None) -> str:
    """
    Извлекает полный заголовок со страницы статьи.
    Если h1 не найден, возвращает fallback_title из preview.
    """

    h1 = soup.find("h1")

    if h1:
        title = clean_text(h1.get_text(" ", strip=True))

        if title:
            return normalize_title(title, "")

    meta_title = soup.find("meta", property="og:title")

    if meta_title and meta_title.get("content"):
        title = clean_text(meta_title.get("content"))

        if title:
            return normalize_title(title, "")

    if fallback_title:
        return clean_text(fallback_title)

    return ""

def extract_article_text(url: str, title: str | None = None) -> str:
    html = fetch_html(url)

    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    domain = get_domain(url)

    for tag in soup([
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "button",
        "noscript",
    ]):
        tag.decompose()

    if "производства.рф" in domain or "xn--80adahnf5bdekrm.xn--p1ai" in domain:
        return extract_text_proizvodstva(soup, title=title)

    if "заводы.рф" in domain or "xn--80az8a.xn--p1ai" in domain:
        return extract_text_zavody(soup, title=title)

    if "rosatom.ru" in domain:
        return extract_text_rosatom(soup, title=title)

    if (
        "военное.рф" in domain
        or "xn--b1aga5aadd.xn--p1ai" in domain
        or "flotprom.ru" in domain
        or "отраслевое.рф" in domain
        or "xn--80aegqufhcjg6b.xn--p1ai" in domain
        or "flot.com" in domain
    ):
        return extract_text_military_media(soup, title=title)

    if "nedradv.ru" in domain:
        return extract_text_nedradv(soup, title=title)

    if "tpprf.ru" in domain:

        text = extract_text_tpprf(soup, title=title)

        if text:
            return text

    if "lukoil.ru" in domain:
        lukoil_title = extract_lukoil_title(soup, fallback_title=title)
        return extract_text_lukoil(soup, title=lukoil_title)

    return extract_text_generic(soup, title=title)


# =========================
# Проверка ссылок
# =========================

BAD_URL_PARTS = [
    "#",
    "mailto:",
    "javascript:",
    "/tag/",
    "/tags/",
    "/category/",
    "/author/",
    "/search",
    "/contacts",
    "/contact",
    "/about",
    "/rss",
    "/feed",
    "/privacy",
    "/personal",
    "/policy",
    "/sitemap",
    "/governance",
    "/company/",
    "/career",
    "/personnel-policy",
    "/tenders",
    "/tendersandauctions",
    "/legal",
    "/documents",
    "/internaldocuments",
    "/contractor",
    "/sustainability",
    "/issues",
    "/books/",
    "/flotcominprintedmedia",
    "/join/",
    "max.ru",
    "translate.google",
    "/miller-journal",
    "/conference",
    "/media-library",
    "/reports",
    "/forum",
]


def is_bad_url(url: str) -> bool:
    lowered = url.lower()

    return any(part in lowered for part in BAD_URL_PARTS)


def is_listing_page(url: str) -> bool:
    lowered = url.lower().rstrip("/")

    listing_pages = [
        "https://promvest.info/ru/novosti-promyishlennosti",
        "https://promvest.info/ru/novosti-kompaniy",
        "https://promvest.info/ru/novosti-avtoproma",
        "https://promvest.info/ru/novosti-metallurgii",
        "https://promvest.info/ru/novosti-mashinostroeniya",
        "https://promvest.info/ru/novosti-energetiki",
        "https://www.nornickel.ru/news-and-media/press-releases-and-news",
        "https://nornickel.ru/news-and-media/press-releases-and-news",
    ]

    return lowered in listing_pages


def is_probably_article_url(url: str) -> bool:
    if is_bad_url(url):
        return False

    if is_listing_page(url):
        return False

    lowered = url.lower()

    article_parts = [
        "/press-center/news/",
        "/media/news/",
        "/press/news/",
        "/journalist/news/",
        "/2026/",
        "/ru/vazhno/",
        "/post/",
        "/publication/",
        "/press-releases/",
        "/pressreleases/",
        "/news-and-media/",

        # новые источники
        ".html",                         # MASHNEWS
        "/doc/",                         # Коммерсантъ
        "/news/promyshlennost/",         # DixiNews
        "/nedradv/ru/news/",             # НедраДВ
        "/ru/news/",                     # ТПП РФ
        "/mining/",                      # Добывающая промышленность
        "/geology/",
        "/metallurgy/",
        "/oilgas/",
        "/nedradv/ru/page_news",
        "/ru/news/",
    ]
    return any(part in lowered for part in article_parts)


# =========================
# Сбор превью из RSS
# =========================

INDUSTRY_RSS_MARKERS = (
    "промышлен", "производств", "завод", "предприяти", "машиностро",
    "металлург", "станк", "судостро", "верф", "судоремонт", "судн",
    "авиацион", "авиастро", "самолетостро", "самолётостро",
    "нефт", "газпром", "газопровод", "газодобы", "газоперераб",
    "газификац", "газохранил", "горнодобы", "добыч", "месторожд",
    "рудник", "шахт", "атомн", "реактор", "электроэнерг", "энергетик",
    "электростанц", "оборудован", "двигател", "инженер", "робот",
    "автоматизац", "микросхем", "полупроводник", "электроник",
    "технолог", "инновац", "железнодорож", "нефтехим", "химическ",
    "логист", "грузоперевоз", "оборонн", "вооружени", "портов",
)


def matches_industrial_rss(title: str, summary: str) -> bool:
    text = (title + " " + summary).lower().replace("ё", "е")
    return any(marker in text for marker in INDUSTRY_RSS_MARKERS)


GENERAL_RSS_FOREIGN_SECTIONS = (
    "/world/", "/mezhdunarodnaya-panorama/",
)

GENERAL_RSS_FOREIGN_ACTOR = re.compile(
    r"^(?:сша|американ\w*|иран\w*|израил\w*|хусит\w*|"
    r"саудовск\w*|финлянд\w*|герман\w*|китай\w*|"
    r"франци\w*|британи\w*|египет\w*|турци\w*)\b",
    re.IGNORECASE,
)

GENERAL_RSS_RUSSIA_CONTEXT = re.compile(
    r"\b(?:росси\w*|рф|отечественн\w*|росатом\w*|"
    r"ростех\w*|роснефт\w*|газпром\w*|оск|"
    r"норникел\w*|лукойл\w*)\b", re.IGNORECASE,
)


def is_foreign_general_rss(url: str, title: str, summary: str) -> bool:
    """Исключаем явно зарубежные сюжеты без связи с промышленностью РФ."""
    path = urlparse(url).path.lower()
    foreign_section = any(
        path.startswith(section) for section in GENERAL_RSS_FOREIGN_SECTIONS
    )
    foreign_actor = bool(GENERAL_RSS_FOREIGN_ACTOR.search(title.strip()))
    if not foreign_section and not foreign_actor:
        return False
    return GENERAL_RSS_RUSSIA_CONTEXT.search(title + " " + summary) is None

def collect_rss_previews(
    source, limit: int, *, apply_filters: bool = True
) -> list[ArticlePreview]:
    """Читаем RSS. apply_filters=False оставляет все записи для проверки сбора."""
    previews = []
    if not source.rss_url:
        return previews

    print("=" * 100)
    print(f"RSS-источник: {source.name}")
    print(f"RSS: {source.rss_url}")

    try:
        response = requests.get(
            source.rss_url,
            headers=HEADERS,
            timeout=(10, 30),
            verify=_request_ca_bundle(),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Ошибка RSS {source.name}: {type(exc).__name__}: {exc}")
        _record_fetch_problem(
            source.rss_url, "rss_fetch_error", f"{type(exc).__name__}: {exc}"
        )
        return previews

    feed = feedparser.parse(response.content)
    if feed.bozo:
        print(f"Ошибка/предупреждение RSS {source.name}: {feed.bozo_exception}")
        if not feed.entries:
            _record_fetch_problem(
                source.rss_url, "rss_parse_error", str(feed.bozo_exception)
            )

    print(f"Найдено записей: {len(feed.entries)}")
    general_feed = source.name in {"РИА Новости", "ТАСС", "Интерфакс"}
    scan_limit = max(300, limit * 30) if general_feed else limit
    seen_urls = set()

    for entry in feed.entries[:scan_limit]:
        title = clean_text(entry.get("title", ""))
        url = str(entry.get("link", "")).strip()
        if not title or not url or url in seen_urls:
            continue
        if urlparse(url).scheme not in ("http", "https"):
            continue
        if not same_domain(url, source.url):
            continue

        preview_text = extract_rss_text(entry)
        if apply_filters and general_feed and not matches_industrial_rss(title, preview_text):
            continue
        if apply_filters and general_feed and is_foreign_general_rss(url, title, preview_text):
            continue

        published_raw = entry.get("published") or entry.get("updated")
        published_at = _parse_published(published_raw)
        seen_urls.add(url)
        previews.append(ArticlePreview(
            source=source.name,
            title=normalize_title(title, source.name),
            url=url,
            published_at=published_at,
            preview_text=preview_text,
        ))
        if len(previews) >= limit:
            break

    return previews

# =========================
# Специальные сборщики ссылок
# =========================

MILPRESS_SOURCES = frozenset({
    "Военное.рф", "ФлотПром", "Отраслевое.рф", "ФлотКом",
})

REACTIVATED_SOURCES = MILPRESS_SOURCES | frozenset({
    "Газпром", "РИА Новости", "ТАСС", "Интерфакс",
})

# Очевидно зарубежные сюжеты Mil.Press без российской связи.
# Не проверяем по одному иностранному топониму: в новости
# о российских поставках может упоминаться иностранный заказчик.
MILPRESS_FOREIGN_TITLE = re.compile(
    r"^(?:финлянд|дания|датск|герман|япони|южн\w+\s+коре|"
    r"саудов|француз|британ|великобритан|американ|сша|китай|"
    r"швеци|канад|грец|испан|индийск|италия|турци|норвеги|"
    r"австрали|подгориц|тиран|албан|черногор|пентагон|naval group|navantia|saab)\w*\b",
    re.IGNORECASE,
)

MILPRESS_RUSSIA_CONTEXT = re.compile(
    r"\b(?:росси\w*|отечественн\w*|рф|росатом\w*|"
    r"ростех\w*|оск|янтар\w*|северн\w*\s+верф\w*|"
    r"адмиралтейск\w*\s+верф\w*|чукотк\w*|"
    r"калининград\w*|петербург\w*|москв\w*|"
    r"владивосток\w*|северодвинск\w*|комсомольск\w*|"
    r"хабаровск\w*)\b",
    re.IGNORECASE,
)


def clearly_foreign_milpress_article(title: str, text: str) -> bool:
    """Для Mil.Press допускаем к публикации только российский контекст.

    Эти четыре сайта одновременно отображают зарубежные новости ВПК.
    Публикации без проверяемой связи с Россией не должны попадать
    в дайджест промышленных новостей России.
    """
    # Заголовок с иностранным главным действующим лицом не становится
    # российской промышленной новостью лишь из-за упоминания России
    # как геополитического противника в конце статьи.
    if MILPRESS_FOREIGN_TITLE.search(title or "") and not (
        MILPRESS_RUSSIA_CONTEXT.search(title or "")
    ):
        return True
    context = f"{title or ''} {(text or '')[:850]}"
    return MILPRESS_RUSSIA_CONTEXT.search(context) is None


def is_milpress_consumer_news(title: str) -> bool:
    """Заголовочные сюжеты о бытовых ценах, а не производстве."""
    return bool(re.search(
        r"\b(?:дорожа\w*|подорожа\w*|цены\s+на)\s+"
        r"(?:куриц\w*|яйц\w*|мяс\w*|хлеб\w*|молок\w*)\b",
        title or "", re.IGNORECASE,
    ))


def collect_milpress_previews(source, limit: int) -> list[ArticlePreview]:
    """Собираем только статьи текущего года, исключая архив и рубрики."""
    from types import SimpleNamespace

    year = datetime.now().year
    # URL архивов Военное.рф / ФлотПром нужно обновлять каждый год.
    # Отраслевое.рф и ФлотКом показывают новости на главной странице.
    listing_url = re.sub(r"/20\d{2}/?$", f"/{year}/", source.url)
    listing_source = SimpleNamespace(name=source.name, url=listing_url)
    return collect_links_by_path(
        source=listing_source,
        limit=limit,
        required_prefix=f"/{year}/",
        exclude_exact=f"/{year}/",
    )


def collect_rosatom_previews(source, limit: int) -> list[ArticlePreview]:
    return collect_links_by_path(
        source=source,
        limit=limit,
        required_prefix="/journalist/news/",
        exclude_exact="/journalist/news"
    )


def collect_links_by_path(
        source,
        limit: int,
        required_prefix: str,
        exclude_exact: str | None = None
) -> list[ArticlePreview]:
    previews = []

    print("=" * 100)
    print(f"HTML-источник: {source.name}")
    print(f"URL: {source.url}")

    html = fetch_html(source.url)

    if not html:
        return previews

    soup = BeautifulSoup(html, "html.parser")
    seen_urls = set()

    for a in soup.select("a[href]"):
        href = a.get("href")

        if not href:
            continue

        absolute_url = urljoin(source.url, href)
        absolute_url = absolute_url.split("#")[0]

        if not same_domain(absolute_url, source.url):
            continue

        path = get_path(absolute_url)

        if not path.startswith(required_prefix.rstrip("/")):
            continue

        if exclude_exact and path == exclude_exact.rstrip("/"):
            continue

        if absolute_url in seen_urls:
            continue

        title = extract_link_title(a)

        if not title or len(title) < 15:
            continue

        if title.lower() in ["читать далее", "подробнее", "новости"]:
            continue

        seen_urls.add(absolute_url)

        previews.append(
            ArticlePreview(
                source=source.name,
                title=normalize_title(title, source.name),
                url=absolute_url,
                published_at=None,
                preview_text="",
            )
        )

        if len(previews) >= limit:
            break

    print(f"Найдено HTML-кандидатов: {len(previews)}")
    return previews


def collect_vestnikprom_previews(source, limit: int) -> list[ArticlePreview]:
    previews = []

    print("=" * 100)
    print(f"HTML-источник: {source.name}")
    print(f"URL: {source.url}")

    html = fetch_html(source.url)

    if not html:
        return previews

    soup = BeautifulSoup(html, "html.parser")
    seen_urls = set()

    heading = soup.find(
        lambda tag:
        tag.name in ["h1", "h2", "h3", "h4"]
        and "Промышленные новости" in tag.get_text(" ", strip=True)
    )

    if not heading:
        print("Блок 'Промышленные новости' не найден")
        return previews

    block = heading

    for _ in range(6):
        if not block.parent:
            break

        block = block.parent
        links = block.select("a[href]")

        article_links = []

        for a in links:
            href = a.get("href")

            if not href:
                continue

            absolute_url = urljoin(source.url, href)
            path = urlparse(absolute_url).path

            if not same_domain(absolute_url, source.url):
                continue

            if re.match(r"^/\d{4}/\d{2}/\d{2}/[^/]+/?$", path):
                article_links.append(a)

        if len(article_links) >= 2:
            break

    for a in article_links:
        href = a.get("href")

        absolute_url = urljoin(source.url, href)
        absolute_url = absolute_url.split("#")[0]

        if absolute_url in seen_urls:
            continue

        title = extract_link_title(a)

        if not title or len(title) < 15:
            continue

        if title.lower() in ["читать далее", "подробнее", "новости", "проновости"]:
            continue

        seen_urls.add(absolute_url)

        previews.append(
            ArticlePreview(
                source=source.name,
                title=normalize_title(title, source.name),
                url=absolute_url,
                published_at=None,
                preview_text="",
            )
        )

        if len(previews) >= limit:
            break

    print(f"Найдено HTML-кандидатов: {len(previews)}")
    return previews

def is_title_fragment(text: str, title: str | None) -> bool:
    """
    Проверяет, является ли абзац фрагментом заголовка.
    Нужно для случаев, когда сайт обрезает начало заголовка
    или вставляет его в первый абзац.
    """

    if not text or not title:
        return False

    text_norm = clean_text(text).lower()
    title_norm = clean_text(title).lower()

    if not text_norm or not title_norm:
        return False

    # Полное совпадение
    if text_norm == title_norm:
        return True

    # Абзац входит в заголовок
    if text_norm in title_norm and len(text_norm) > 20:
        return True

    # Заголовок входит в абзац
    if title_norm in text_norm:
        return True

    # Сравнение без кавычек и лишней пунктуации
    simple_text = re.sub(r"[«»\"'.,:;!?()\[\]—-]", " ", text_norm)
    simple_title = re.sub(r"[«»\"'.,:;!?()\[\]—-]", " ", title_norm)

    simple_text = clean_text(simple_text)
    simple_title = clean_text(simple_title)

    if simple_text in simple_title and len(simple_text) > 20:
        return True

    if simple_title in simple_text:
        return True

    return False


def remove_title_prefix(text: str, title: str | None) -> str:
    """
    Удаляет из начала текста заголовок или его фрагмент.
    Например:
    'Карьер 26»: обзор обновлённой системы диспетчеризации Импортонезависимая...'
    """

    if not text or not title:
        return text

    text = clean_text(text)
    title = clean_text(title)

    # Пробуем удалить полный заголовок
    if text.lower().startswith(title.lower()):
        return clean_text(text[len(title):])

    # Берём значимые хвосты заголовка и удаляем их из начала текста
    title_parts = [
        title,
        title.replace("«", "").replace("»", ""),
    ]

    words = title.replace("«", "").replace("»", "").split()

    # хвосты заголовка: последние 4, 5, 6... слов
    for n in range(min(len(words), 10), 3, -1):
        fragment = " ".join(words[-n:])

        title_parts.append(fragment)

    for fragment in title_parts:
        fragment = clean_text(fragment)

        if len(fragment) < 20:
            continue

        if text.lower().startswith(fragment.lower()):
            return clean_text(text[len(fragment):])

    return text

def extract_lukoil_title(soup: BeautifulSoup, fallback_title: str | None = None) -> str:
    """
    Извлекает настоящий заголовок новости ЛУКОЙЛа.
    В RSS у ЛУКОЙЛа заголовок иногда приходит как 'Пресс-релиз',
    поэтому заголовок лучше брать со страницы статьи.
    """

    selectors = [
        "h1",
        ".press-release__title",
        ".pressrelease-title",
        ".news-detail__title",
        ".article__title",
        ".detail-title",
        ".page-title",
        "meta[property='og:title']",
    ]

    bad_titles = {
        "пресс-релиз",
        "пресс-релизы",
        "новости",
        "новость",
    }

    for selector in selectors:
        tag = soup.select_one(selector)

        if not tag:
            continue

        if tag.name == "meta":
            title = tag.get("content", "")
        else:
            title = tag.get_text(" ", strip=True)

        title = clean_text(title)

        if not title:
            continue

        if title.lower() in bad_titles:
            continue

        return title

    return clean_text(fallback_title or "")

def extract_text_lukoil(soup: BeautifulSoup, title: str | None = None) -> str:
    """
    Извлекает текст пресс-релиза ЛУКОЙЛа.
    """

    selectors = [
        "article p",
        ".press-release p",
        ".pressrelease p",
        ".press-release__text p",
        ".news-detail p",
        ".news-detail__text p",
        ".article p",
        ".article__text p",
        ".content p",
        ".main-content p",
        "main p",
        "div[class*='press'] p",
        "div[class*='article'] p",
        "div[class*='content'] p",
        "div[class*='text'] p",
    ]

    text = extract_by_selectors(
        soup=soup,
        selectors=selectors,
        min_len=120,
        title=title,
    )

    if text:
        return text

    paragraphs = [
        p.get_text(" ", strip=True)
        for p in soup.find_all("p")
    ]

    paragraphs = filter_paragraphs(paragraphs, title=title)
    text = normalize_article_text(" ".join(paragraphs), title=title)
    text = cut_related_news(text)

    if len(text) >= 120:
        return text

    return ""

def extract_text_dprom(soup: BeautifulSoup, title: str | None = None) -> str:
    """
    Извлекает текст статьи с dprom.online.
    У этого источника в начало текста иногда попадает фрагмент заголовка,
    поэтому дополнительно удаляем title и похожие на него фрагменты.
    """

    selectors = [
        "article p",
        ".entry-content p",
        ".post-content p",
        ".article-content p",
        ".content p",
        "main p",
        "div[class*='content'] p",
        "div[class*='article'] p",
        "div[class*='text'] p",
    ]

    paragraphs = []

    for selector in selectors:
        blocks = soup.select(selector)

        if not blocks:
            continue

        for block in blocks:
            text = clean_text(block.get_text(" ", strip=True))

            if not text:
                continue

            if len(text) < 40:
                continue

            lower = text.lower()

            bad_fragments = [
                "читайте также",
                "поделиться",
                "фотогалерея",
                "реклама",
                "подписывайтесь",
                "другие материалы",
                "материалы по теме",
                "источник:",
                "главное",
            ]

            if any(fragment in lower for fragment in bad_fragments):
                continue

            if is_title_fragment(text, title):
                continue

            text = remove_title_prefix(text, title)

            if len(text) >= 40:
                paragraphs.append(text)

        if paragraphs:
            break

    # убираем дубли
    cleaned = []
    seen = set()

    for paragraph in paragraphs:
        key = paragraph.lower()

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(paragraph)

    article_text = " ".join(cleaned)
    article_text = normalize_article_text(article_text, title=title)
    article_text = cut_related_news(article_text)

    return article_text

def collect_mashnews_previews(source, limit: int) -> list[ArticlePreview]:
    return collect_links_by_path(
        source=source,
        limit=limit,
        required_prefix="/",
        exclude_exact=None,
    )

def extract_link_title(a) -> str:
    """
    Достаёт нормальный заголовок из ссылки.
    Нужен, чтобы в title не попадал весь текст карточки.
    """
    if not a:
        return ""

    for selector in ["h1", "h2", "h3", "h4", ".title", ".entry-title"]:
        tag = a.select_one(selector)
        if tag:
            title = clean_text(tag.get_text(" ", strip=True))
            if len(title) >= 10:
                return title

    title = clean_text(a.get("title", ""))

    if not title:
        title = clean_text(a.get_text(" ", strip=True))

    bad_parts = [
        "Читать далее",
        "Подробнее",
        "Проновости",
        "Новости",
        "Добыча",
        "Металлургия",
        "Геология",
        "Нефтегаз",
    ]

    for part in bad_parts:
        title = re.sub(rf"\b{re.escape(part)}\b", "", title, flags=re.IGNORECASE)

    title = re.sub(r"\s+", " ", title).strip(" —-")

    # если случайно попало описание, режем по первому нормальному концу предложения
    if len(title) > 180:
        title = re.split(r"(?<=[.!?])\s+", title)[0].strip()

    return title


def collect_kommersant_industry_previews(source, limit: int) -> list[ArticlePreview]:
    # На странице /theme/3631 общая лента идёт ПЕРЕД разделом
    # «Промышленность». Отбираем ссылки только из раздела.
    return collect_html_previews(source, limit=limit)

def collect_dixinews_industry_previews(source, limit: int) -> list[ArticlePreview]:
    return collect_links_by_path(
        source=source,
        limit=limit,
        required_prefix="/news/promyshlennost/",
        exclude_exact="/news/promyshlennost",
    )


def collect_nedradv_previews(source, limit: int) -> list[ArticlePreview]:
    """
    Собирает новости с НедраДВ.

    У этого сайта страница списка новостей находится по адресу:
    https://nedradv.ru/nedradv/ru/news

    Но сами статьи имеют вид:
    https://nedradv.ru/nedradv/ru/page_news?obj=...
    """

    previews = []

    print("=" * 100)
    print(f"HTML-источник: {source.name}")
    print(f"URL: {source.url}")

    html = fetch_html(source.url)

    if not html:
        return previews

    soup = BeautifulSoup(html, "html.parser")
    seen_urls = set()

    for a in soup.select("a[href]"):
        title = clean_text(a.get_text(" ", strip=True))
        href = a.get("href")

        if not title or not href:
            continue

        if len(title) < 20:
            continue

        absolute_url = urljoin(source.url, href)

        if not same_domain(absolute_url, source.url):
            continue

        if "/nedradv/ru/page_news" not in absolute_url:
            continue

        if "obj=" not in absolute_url:
            continue

        if absolute_url in seen_urls:
            continue

        seen_urls.add(absolute_url)

        previews.append(
            ArticlePreview(
                source=source.name,
                title=normalize_title(title, source.name),
                url=absolute_url,
                published_at=None,
                preview_text="",
            )
        )

        if len(previews) >= limit:
            break

    print(f"Найдено HTML-кандидатов: {len(previews)}")
    return previews


def collect_tpprf_previews(source, limit: int) -> list[ArticlePreview]:
    return collect_links_by_path(
        source=source,
        limit=limit,
        required_prefix="/ru/news/",
        exclude_exact="/ru/news",
    )


def collect_dprom_previews(source, limit: int) -> list[ArticlePreview]:
    previews = []

    print("=" * 100)
    print(f"HTML-источник: {source.name}")
    print(f"URL: {source.url}")

    html = fetch_html(source.url)

    if not html:
        return previews

    soup = BeautifulSoup(html, "html.parser")
    seen_urls = set()

    allowed_parts = [
        "/mining/",
        "/geology/",
        "/metallurgy/",
        "/oilgas/",
        "/technologies/",
        "/events/",
    ]

    for a in soup.select("a[href]"):
        href = a.get("href")

        if not href:
            continue

        absolute_url = urljoin(source.url, href)
        absolute_url = absolute_url.split("#")[0]

        if not same_domain(absolute_url, source.url):
            continue

        if not any(part in absolute_url for part in allowed_parts):
            continue

        if absolute_url in seen_urls:
            continue

        title = extract_link_title(a)

        if not title or len(title) < 15:
            continue

        if title.lower() in ["читать далее", "подробнее", "добыча"]:
            continue

        seen_urls.add(absolute_url)

        previews.append(
            ArticlePreview(
                source=source.name,
                title=normalize_title(title, source.name),
                url=absolute_url,
                published_at=None,
                preview_text="",
            )
        )

        if len(previews) >= limit:
            break

    print(f"Найдено HTML-кандидатов: {len(previews)}")
    return previews


# Строгие шаблоны URL для сайтов, на которых общий сбор пропускал
# новости или забирал ссылки на рубрики вместо статей.
_SOURCE_ARTICLE_PATHS = {
    "ИАПН": re.compile(r"^/[^/]{6,}/?$"),
    "Промышленный вестник / ПВ.РФ": re.compile(
        r"^/ru/(?:novosti-promyishlennosti|vazhno|novosti-kompaniy)/[^/]+/?$"
    ),
    "Производства.рф": re.compile(
        r"^/(?:publication|publikacii|news|novosti|post|article|articles)/[^/]+/?$"
    ),
    "Коммерсантъ — Промышленность": re.compile(r"^/doc/\d+/?$"),
    "Ростех": re.compile(r"^/media/news/[^/]+/?$"),
    "Газпром": re.compile(r"^/press/news/\d{4}/[a-z]+/article\d+/?$", re.I),
    "Роснефть": re.compile(r"^/press/news/item/\d+/?$"),
}

_BAD_ARTICLE_PATHS = (
    "/contact", "/about", "/privacy", "/tag/", "/category/",
    "/author/", "/feed", "/rss", "/search", "/page/", "/project",
)


def _is_real_article_link(source_name: str, url: str) -> bool:
    from urllib.parse import urlsplit

    path = urlsplit(url).path
    if any(path.lower().startswith(s) for s in _BAD_ARTICLE_PATHS):
        return False
    pattern = _SOURCE_ARTICLE_PATHS.get(source_name)
    if pattern:
        if pattern.match(path):
            return True
        # На «Производства.рф» структура меняется. Только карточки
        # с явно оформленным заголовком проходят запасной путь.
        return source_name == "Производства.рф" and len(path.strip("/").split("/")) >= 2
    return is_probably_article_url(url)


def _article_link_heading(anchor) -> str:
    for selector in (
        "h2", "h3", "h4", "[class*='title']", "[class*='headline']",
        "[class*='uho__name']", "[itemprop='headline']",
    ):
        h = anchor.select_one(selector)
        if h:
            value = clean_text(h.get_text(" ", strip=True))
            if len(value) >= 14:
                return value
    if anchor.parent and anchor.parent.name in ("h2", "h3", "h4"):
        return clean_text(anchor.get_text(" ", strip=True))
    # Визуальная карточка может иметь имя в title.
    return clean_text(anchor.get("title", ""))


def _legacy_collect_html_previews(source, limit: int) -> list[ArticlePreview]:
    previews = []

    print("=" * 100)
    print(f"HTML-источник: {source.name}")
    print(f"URL: {source.url}")

    if not source.url:
        return previews

    html = fetch_html(source.url)

    if not html:
        return previews

    soup = BeautifulSoup(html, "html.parser")
    seen_urls = set()

    for a in soup.select("a[href]"):
        title = clean_text(a.get_text(" ", strip=True))
        href = a.get("href")

        if not title or not href or len(title) < 15:
            continue

        absolute_url = urljoin(source.url, href)

        if not same_domain(absolute_url, source.url):
            continue

        if absolute_url in seen_urls:
            continue

        if not is_probably_article_url(absolute_url):
            continue

        seen_urls.add(absolute_url)

        previews.append(
            ArticlePreview(
                source=source.name,
                title=normalize_title(title, source.name),
                url=absolute_url,
                published_at=None,
                preview_text="",
            )
        )

        if len(previews) >= limit:
            break

    print(f"Найдено HTML-кандидатов: {len(previews)}")

    return previews




# Дополнительные страницы с подтверждёнными лентами новостей.
# Основной URL источника всегда проверяем первым.
_SOURCE_LISTING_ALTERNATIVES = {
    "Ростех": (
        "https://www.rostec.ru/media/news/",
        "https://rostec.ru/media/",
    ),
    "Роснефть": (
        "https://www.rosneft.ru/press/news/2026/",
        "https://rosneft.ru/press/news/",
        "https://limited.rosneft.ru/press/news/",
    ),
    "Производства.рф": (
        "https://производства.рф/articles/",
    ),
    "Промышленный вестник / ПВ.РФ": (
        "https://promvest.info/",
    ),
}


def _canonical_source_link(source, absolute_url: str) -> str | None:
    """
    Ограничиваем сбор проверенными доменами одного источника.
    Лёгкая версия Роснефти — официальный поддомен rosneft.ru:
    сохраняем URL на основном домене во избежание дубликатов.
    """
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(absolute_url)
    domain = (parts.hostname or "").lower()
    if source.name == "Роснефть" and domain == "limited.rosneft.ru":
        canonical_host = urlsplit(source.url).netloc
        return urlunsplit(("https", canonical_host, parts.path, parts.query, ""))
    if same_domain(absolute_url, source.url):
        # rostec.ru и www.rostec.ru — один источник, а не две новости.
        # Нормализуем домен и схему к URL источника из БД.
        origin = urlsplit(source.url)
        return urlunsplit((
            origin.scheme, origin.netloc, parts.path, parts.query, ""
        ))
    return None


def _listing_page_anchors(soup, source_name):
    """Для ИАПН берём текущую ленту, а не старые блоки архива."""
    if source_name == "ИАПН":
        heading = soup.find(
            lambda t: t.name in ("h1", "h2")
            and clean_text(t.get_text(" ", strip=True)).lower() == "новости"
        )
        if not heading:
            return []
        result = []
        for node in heading.next_elements:
            if getattr(node, "name", None) in ("h1", "h2"):
                break
            if getattr(node, "name", None) == "a":
                if node.find_parent(("h3", "h4")):
                    result.append(node)
        return result

    if source_name == "Коммерсантъ — Промышленность":
        heading = soup.find(
            lambda t: t.name in ("h1", "h2")
            and clean_text(t.get_text(" ", strip=True)).lower().startswith("промышленность")
        )
        if not heading:
            print("Нет заголовка раздела «Промышленность»")
            return []
        links = []
        for node in heading.next_elements:
            if getattr(node, "name", "") in ("h1", "h2"):
                marker = clean_text(node.get_text(" ", strip=True)).lower()
                if "новости компаний" in marker:
                    break
            if getattr(node, "name", "") == "a":
                links.append(node)
        return links

    return soup.select("a[href]")


def collect_html_previews(source, limit: int) -> list[ArticlePreview]:
    """Сбор по первичной ленте, затем по альтернативной ленте сайта."""
    from types import SimpleNamespace

    urls = list(dict.fromkeys(
        [source.url] + list(_SOURCE_LISTING_ALTERNATIVES.get(source.name, ()))
    ))
    previews, seen = [], set()

    for listing_url in urls:
        if len(previews) >= limit:
            break
        html = fetch_html(listing_url)
        if not html:
            print(f"Нет HTML списка: {source.name} — {listing_url}")
            continue
        if "Error establishing a database connection" in html:
            print(f"Сайт {source.name} возвращает ошибку собственной БД: {listing_url}")
            continue

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()

        anchors = _listing_page_anchors(soup, source.name)
        count_before = len(previews)
        for anchor in anchors:
            href = anchor.get("href")
            if not href:
                continue
            absolute = urljoin(listing_url, href).split("#")[0]
            absolute = _canonical_source_link(source, absolute)
            if not absolute or absolute in seen:
                continue
            if not _is_real_article_link(source.name, absolute):
                continue

            title = _article_link_heading(anchor)
            if not title:
                title = clean_text(anchor.get_text(" ", strip=True))
            if len(title) > 210:
                title = next((
                    clean_text(x) for x in anchor.stripped_strings
                    if 14 <= len(clean_text(x)) <= 210
                    and not re.fullmatch(r"[\d. /:]+", clean_text(x))
                ), "")
            if len(title) < 14 or len(title) > 210:
                continue

            seen.add(absolute)
            previews.append(ArticlePreview(
                source=source.name,
                title=normalize_title(title, source.name),
                url=absolute,
                published_at=None,
                preview_text="",
            ))
            if len(previews) >= limit:
                break

        found = len(previews) - count_before
        print(
            f"{source.name}: {found} новых ссылок из {listing_url}, "
            f"в HTML {len(html)} символов, всего a[href]={len(anchors)}"
        )

    return previews


def collect_rostec_previews(source, limit: int):
    return collect_html_previews(source, limit)


def collect_rosneft_previews(source, limit: int):
    return collect_html_previews(source, limit)

# =========================
# Сбор полной новости
# =========================


# Селекторы в порядке от наиболее узких к общим.
# Широкий поиск по всем <p> страницы ЗАПРЕЩЁН.
_ARTICLE_SELECTORS = {
    "Газпром": (".article-body", ".news-detail__text", "article"),
    "Военное.рф": ("[itemprop='articleBody']", ".article_text", ".news_text", "article"),
    "ФлотПром": ("[itemprop='articleBody']", ".article_text", ".news_text", "article"),
    "Отраслевое.рф": ("[itemprop='articleBody']", ".article_text", ".news_text", "article"),
    "ФлотКом": ("[itemprop='articleBody']", ".article_text", ".news_text", "article"),
    "Ростех": (".media-news-detailed__content", "[itemprop='articleBody']", ".news-detail__content", ".news-detail__text", "article"),
    "ОСК": (".news-detail__text", ".detail_text", ".news-detail", "article"),
    "Роснефть": (".content-column .common-text", ".common-text", ".news-detail", ".news-body", ".news-text", "article"),
    "Заводы.рф": (".article-txt-wrap .text-bl", ".article-txt-wrap", ".publication-content", ".publication__body", ".article-content"),
    "ЛУКОЙЛ": (".press-release__text", ".pressrelease__text", ".press-release__body", ".news-detail__text", "article"),
    "Норильский никель": (".news-detail__content", ".press-release__text", ".article__content", "article"),
    "Энергетика и промышленность России": (".news-detail", ".article-body", ".article-text", "article"),
    "Вестник промышленности": (".entry-content", ".post-content", ".wp-block-post-content", "article"),
    "MASHNEWS": (".article__content", ".article-content", ".article__body", "article"),
    "Коммерсантъ — Промышленность": (".article_text_wrapper", ".article_text", "[itemprop='articleBody']", "article"),
    "DixiNews — Промышленность": (".article-content", ".news-detail", ".news-text", "article"),
    "НедраДВ": (),
    "ТПП РФ": (".news-detail__text", ".news-detail-text", ".article__text", "article"),
    "Добывающая промышленность": (".entry-content", ".post-content", ".article-content", "article"),
    "ИАПН": (".entry-content", ".post-content", ".td-post-content", "article"),
    "Промышленный вестник / ПВ.РФ": (".article-content", ".news-detail", ".content-news", "article"),
    "Производства.рф": (".post-content", ".entry-content", ".article-content", "article"),
}

_RU_MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
    # На страницах Роснефти встречается «05 Ноябрь 2026», а не «ноября».
    "январь": 1, "февраль": 2, "март": 3, "апрель": 4,
    "май": 5, "июнь": 6, "июль": 7, "август": 8,
    "сентябрь": 9, "октябрь": 10, "ноябрь": 11, "декабрь": 12,
}


def _parse_published(value):
    if not value:
        return None
    from email.utils import parsedate_to_datetime
    try:
        if isinstance(value, (list, tuple)):
            return _parse_published(value[0]) if value else None
        raw = str(value).strip()
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        # Проект хранит даты без часового пояса; для aware UTC.
        if dt.tzinfo:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (TypeError, ValueError):
        pass

    try:
        dt = parsedate_to_datetime(str(value))
        if dt.tzinfo:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (TypeError, ValueError, IndexError):
        pass

    match = re.search(
        r"\b(\d{1,2})[ .]+([а-яё]+)[ .]+(\d{4})(?:\s*г(?:ода|\.)?)?\s*,?\s*(?:в\s+)?(\d{1,2}:\d{2})?",
        str(value).lower()
    )
    if match and match.group(2) in _RU_MONTHS:
        try:
            hour, minute = (0, 0)
            if match.group(4):
                hour, minute = (int(x) for x in match.group(4).split(":"))
            return datetime(
                int(match.group(3)), _RU_MONTHS[match.group(2)],
                int(match.group(1)), hour, minute
            )
        except ValueError:
            pass
    match = re.search(
        r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})(?:\s+(\d{1,2}):(\d{2})(?::(\d{2}))?)?\b",
        str(value),
    )
    if match:
        try:
            return datetime(
                int(match.group(3)), int(match.group(2)), int(match.group(1)),
                int(match.group(4) or 0), int(match.group(5) or 0),
                int(match.group(6) or 0),
            )
        except ValueError:
            pass
    return None


def _jsonld_article(soup):
    import json
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or script.get_text())
        except (ValueError, TypeError):
            continue
        nodes = data if isinstance(data, list) else [data]
        while nodes:
            node = nodes.pop()
            if not isinstance(node, dict):
                continue
            if isinstance(node.get("@graph"), list):
                nodes.extend(node["@graph"])
            kind = node.get("@type", "")
            if isinstance(kind, list):
                kinds = kind
            else:
                kinds = [kind]
            if any(str(k).lower() in ("newsarticle", "article", "blogposting", "reportage") for k in kinds):
                return node
    return {}


def _page_date(soup, jsonld):
    for value in (
        jsonld.get("datePublished"),
        jsonld.get("dateCreated"),
    ):
        date = _parse_published(value)
        if date:
            return date

    for selector, attr in (
        ('meta[property="article:published_time"]', "content"),
        ('meta[itemprop="datePublished"]', "content"),
        ('meta[name="pubdate"]', "content"),
        ('meta[name="date"]', "content"),
        ('time[datetime]', "datetime"),
        ('[itemprop="datePublished"][datetime]', "datetime"),
    ):
        node = soup.select_one(selector)
        if node:
            date = _parse_published(node.get(attr))
            if date:
                return date

    # Локальная дата у заголовка, не дата случайной боковой карточки.
    h1 = soup.find("h1")
    if h1:
        for parent in (h1.parent, getattr(h1.parent, "parent", None)):
            if parent:
                for node in parent.select("time, .date, [class*='date'], [class*='time']"):
                    date = _parse_published(node.get_text(" ", strip=True))
                    if date:
                        return date
        # NEDRADV: основной абзац содержит «5 ноября 2026 года, NEDRADV».
        # Ищем только до первого заголовка «Читайте также».
        first_text = []
        for node in h1.next_elements:
            if getattr(node, "name", "") in ("h2", "h3"):
                marker = clean_text(node.get_text(" ", strip=True)).lower()
                if "читайте также" in marker:
                    break
            if getattr(node, "name", "") == "p":
                first_text.append(clean_text(node.get_text(" ", strip=True)))
                if len(first_text) >= 3:
                    break
        if first_text:
            combined = " ".join(first_text)
            match = re.search(
                r"(?i)(\d{1,2}\s+[а-яё]+\s+\d{4}\s+года),?\s*NEDRADV",
                combined,
            )
            if match:
                return _parse_published(match.group(1))
    # Корпоративные CMS: дата находится отдельной строкой сразу после
    # основного заголовка. Не ищем даты по всему телу статьи.
    title_nodes = soup.find_all("h1")
    if title_nodes:
        headline = title_nodes[-1]
        seen_strings = 0
        for node in headline.next_elements:
            if isinstance(node, str):
                value = clean_text(node)
                if not value:
                    continue
                seen_strings += 1
                if seen_strings > 12 or len(value) > 210:
                    break
                if re.fullmatch(
                    r"\d{1,2}\s+[а-яё]+\s+\d{4}\s*г?\.?",
                    value.lower(),
                ) or re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}(?:\s+\d{2}:\d{2}(?::\d{2})?)?", value):
                    date = _parse_published(value)
                    if date:
                        return date
    return None


def _article_paragraphs(block, title):
    """
    Извлекает содержимое только уже найденного контейнера статьи.
    В отличие от прежней реализации, сохраняет пункты ul/ol и
    подзаголовки внутри статьи без меню и рекомендательных ссылок.
    """
    from bs4 import BeautifulSoup

    # Работаем с копией, не изменяем HTML для поиска даты и заголовка.
    block = BeautifulSoup(str(block), "html.parser")
    exclude = (
        "script,style,nav,footer,aside,form,button,"
        ".related,.related-news,.read-also,.recommendations,.other-news,"
        "[class*='related'],[class*='recommend'],[class*='sidebar'],"
        "[class*='share'],[class*='social'],[class*='advert'],"
        "[class*='subscribe'],[class*='comments'],"
        "[class*='navigation'],[class*='breadcrumbs'],"
        ".post-navigation,.article-tags,.news-list,.menu,.toc,"
        ".article-list-wrap,.table-of-contents,.contents-list,"
        "[class^='toc-'],[class*=' toc-']"
    )
    for child in block.select(exclude):
        child.decompose()

    parts, seen = [], set()
    saw_paragraph = False
    for node in block.select("p, blockquote, li, h2, h3, h4"):
        if node.name == "p" and node.find_parent(["blockquote", "li"]):
            # Вложенный текст <p> будет учтён один раз в li/blockquote.
            continue
        if node.name in ("h2", "h3", "h4") and node.find_parent("li"):
            continue
        if node.name == "blockquote" and node.find_parent("blockquote"):
            continue

        if node.name == "li":
            # Не дублируем вложенные списки: каждый li отдельно.
            item = BeautifulSoup(str(node), "html.parser")
            for nested in item.select("ul, ol"):
                nested.decompose()
            # Ссылки внутри самого материала допустимы: например,
            # названия моделей оборудования в техническом списке.
            # Оглавления/меню удалены выше на уровне контейнера.
            value = clean_text(item.get_text(" ", strip=True))
        else:
            value = clean_text(node.get_text(" ", strip=True))

        if not value:
            continue
        lower = value.casefold()
        if title and lower == clean_text(title).casefold():
            continue
        if any(lower.startswith(stop) for stop in (
            "читайте также", "последние новости", "последние публикации",
            "последние аукционы", "обсудить новости промышленности",
            "подписывайтесь", "поделиться", "департамент информации и рекламы",
            "фото:", "источник:", "на правах рекламы",
        )):
            continue
        value = re.split(
            r"(?i)\b(?:обсудить новости промышленности|департамент информации и рекламы ПАО)\b",
            value, maxsplit=1
        )[0].strip()
        if not value:
            continue

        if node.name in ("h2", "h3", "h4"):
            # Начальный h3 в корпоративных пресс-релизах часто дублирует
            # лид; сохраняем только внутренние подзаголовки статьи.
            if not saw_paragraph or len(value) < 12:
                continue
        elif node.name == "p":
            # Не теряем короткие измерения вроде «500 МВт».
            if len(value) < 12 and not re.search(r"\d", value):
                continue
            saw_paragraph = True
        elif node.name == "blockquote":
            if len(value) < 12:
                continue
            saw_paragraph = True
        elif node.name == "li":
            if len(value) < 4:
                continue

        # Длина и цифры важных пунктов («Ту-95А», «500 МВт»)
        # не должны приводить к их удалению.
        key = re.sub(r"[^а-яёa-z0-9]+", " ", lower).strip()
        if not key or key in seen:
            continue
        seen.add(key)

        if node.name == "li":
            # Граница пункта должна сохраниться и после нормализации
            # пробелов, иначе два элемента списка сливаются.
            if value[-1] not in ".!?;:":
                value += "."
        elif node.name in ("h2", "h3", "h4") and value[-1] not in ".!?;:":
            value += "."

        parts.append(value)

    return clean_text(" ".join(parts))


def _lukoil_published_date(soup):
    """
    В реальном HTML ЛУКОЙЛ дата в span.pressrelise-article-date
    (именно 'pressrelise' — так написан класс на сайте).
    Предыдущий обход next_elements прерывался на script перед датой.
    """
    node = soup.select_one(
        ".pressrelise-article-date, .pressrelease-article-date"
    )
    if node:
        parsed = _parse_published(node.get_text(" ", strip=True))
        if parsed:
            return parsed

    # Запасной вариант без чтения JS и дат в тексте соседних статей.
    headline = soup.select_one("h1.press-release-page-title")
    if headline is None:
        all_headlines = soup.find_all("h1")
        headline = all_headlines[-1] if all_headlines else None
    if headline:
        for candidate in headline.find_all_next("span", limit=8):
            value = clean_text(candidate.get_text(" ", strip=True))
            if re.fullmatch(
                r"\d{1,2}\s+[а-яё]+\s+\d{4}\s*г?\.?",
                value.lower(),
            ):
                return _parse_published(value)
    return None


def _source_article_headline_date(soup, source_name, title):
    """
    Ростех публикует дату непосредственно ПЕРЕД заголовком
    (06.11.2026 01:07); Роснефть — ПОСЛЕ него
    (03 Ноябрь 2026). Ищем только рядом с заголовком этой статьи,
    не сканируем основной текст, блоки рекомендаций и боковую ленту.
    """
    if source_name not in ("Ростех", "Роснефть") or not title:
        return None

    wanted = clean_text(title).casefold()
    heading = next((
        node for node in soup.select("h1, h2, h3")
        if clean_text(node.get_text(" ", strip=True)).casefold() == wanted
    ), None)
    if heading is None:
        return None

    date_pattern = re.compile(
        r"(?:\d{1,2}\.\d{1,2}\.\d{4}"
        r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?"
        r"|\d{1,2}\s+[а-яё]+\s+\d{4}"
        r"(?:\s*г\.?)?(?:\s*,?\s*\d{1,2}:\d{2})?)",
        flags=re.IGNORECASE,
    )

    def near_date(nodes):
        inspected = 0
        for node in nodes:
            if not isinstance(node, str):
                continue
            value = clean_text(node)
            if not value:
                continue
            # next_elements начинается с собственного текста h1/h2/h3.
            # Для длинных заголовков (например, Роснефть /224924/)
            # прежде этот текст превышал 85 символов и поиск
            # завершался ДО стоящего сразу после заголовка span с датой.
            if value.casefold() == wanted:
                continue
            inspected += 1
            if inspected > 16:
                break
            if len(value) > 85:
                # Длинное предложение — уже другая часть страницы.
                break
            if date_pattern.fullmatch(value):
                result = _parse_published(value)
                if result:
                    return result
        return None

    if source_name == "Ростех":
        return near_date(heading.previous_elements) or near_date(heading.next_elements)
    return near_date(heading.next_elements) or near_date(heading.previous_elements)


def _source_specific_published_date(soup, source_name):
    """Даты из подтверждённых HTML-контейнеров конкретных сайтов."""
    if source_name in MILPRESS_SOURCES:
        heading = soup.find("h1")
        if heading:
            # Mil.Press: «Важное 6 ноября 2026 в 10:07» рядом с h1.
            date_pattern = re.compile(
                r"\b\d{1,2}\s+[а-яё]+\s+20\d{2}\s+в\s+"
                r"\d{1,2}:\d{2}\b", re.I,
            )
            headline = clean_text(heading.get_text(" ", strip=True))
            inspected = 0
            for node in heading.next_elements:
                if not isinstance(node, str):
                    continue
                value = clean_text(node)
                if not value or value == headline:
                    continue
                inspected += 1
                if inspected > 15 or len(value) > 220:
                    break
                match = date_pattern.search(value)
                if match:
                    return _parse_published(match.group())

    if source_name == "Ростех":
        # Реальный HTML: <div class="media-news-single__date">
        #   <span class="date">06.11.2026</span>
        #   <span class="date">04:25</span>
        # </div>, сразу перед h2.media-news-single__name.
        # Обе части принадлежат одному заголовку, поэтому нельзя
        # подставлять время из ленты других новостей.
        for heading in soup.select(
            "h1.media-news-single__name, h2.media-news-single__name"
        ):
            info = heading.find_parent(class_="media-news-single__info")
            if info is None:
                continue
            date_block = info.select_one(".media-news-single__date")
            if date_block is None:
                continue
            date_text = clean_text(date_block.get_text(" ", strip=True))
            if not re.fullmatch(
                r"\d{1,2}\.\d{1,2}\.\d{4}"
                r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?",
                date_text,
            ):
                continue
            parsed = _parse_published(date_text)
            if parsed:
                return parsed

    selectors = {
        "ЛУКОЙЛ": (
            ".pressrelise-article-date",
            ".pressrelease-article-date",
        ),
        "MASHNEWS": (
            ".grarticle-header .grarticle-items .grarticle-item",
            ".grarticle-header .grarticle-item",
        ),
        "ТПП РФ": (
            ".news-detail__date .ribbon",
            ".news-detail__date",
        ),
        "Заводы.рф": (
            "main .article-head__top .date",
            "main span.date",
        ),
    }
    for selector in selectors.get(source_name, ()):
        for node in soup.select(selector)[:5]:
            result = _parse_published(node.get_text(" ", strip=True))
            if result:
                return result
    if source_name == "ЛУКОЙЛ":
        return _lukoil_published_date(soup)
    return None

def _lukoil_main_body(soup, title):
    """Тело пресс-релиза до блока «Вас может заинтересовать»."""
    headings = soup.find_all(["h1", "h2"])
    heading = next(
        (
            h for h in headings
            if title
            and clean_text(h.get_text(" ", strip=True)).casefold() == clean_text(title).casefold()
        ),
        None,
    )
    if heading is None:
        heading = next(
            (h for h in headings if len(clean_text(h.get_text(" ", strip=True))) > 22),
            None,
        )
    if heading is None:
        return ""

    stops = (
        "вас может заинтересовать", "контактная информация",
        "пресс-релиз в формате pdf", "все новости",
        "другие материалы", "читайте также", "поделиться",
    )
    paragraphs, seen = [], set()
    for tag in heading.next_elements:
        if not getattr(tag, "name", None):
            continue
        if tag.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            heading_text = clean_text(tag.get_text(" ", strip=True)).casefold()
            if any(stop in heading_text for stop in stops):
                break
        if tag.name not in ("p", "blockquote", "li"):
            continue
        if tag.name == "p" and tag.find_parent(["blockquote", "li"]):
            continue
        if tag.name == "li":
            if tag.find_parent(["nav", "aside", "footer"]):
                continue
            item = BeautifulSoup(str(tag), "html.parser")
            for nested in item.select("ul, ol"):
                nested.decompose()
            value = clean_text(item.get_text(" ", strip=True))
        else:
            value = clean_text(tag.get_text(" ", strip=True))
        lower = value.casefold()
        if any(stop in lower for stop in stops):
            break
        if len(value) < (4 if tag.name == "li" else 28) or value in seen:
            continue
        seen.add(value)
        if tag.name == "li" and value[-1] not in ".!?;:":
            value += "."
        paragraphs.append(value)

    return clean_text(" ".join(paragraphs))


def extract_text_kommersant(soup, title: str | None = None) -> str:
    """
    У «Коммерсанта» содержимое статьи находится в блоках
    .article_text_wrapper .doc-content__text-block, а не в <p>.
    Фото, подписи к фото и статьи боковой ленты не включаем.
    """
    wrapper = soup.select_one(".article_text_wrapper")
    if wrapper is None:
        return ""

    parts = []
    seen = set()
    for block in wrapper.select(".doc-content__text-block, ul li, ol li"):
        if block.name == "li":
            # Если список уже находится внутри text-block, он включён
            # в текст родителя; второй раз пункт не добавляем.
            if block.find_parent(class_="doc-content__text-block"):
                continue
            if block.find_parent(["nav", "aside", "footer"]):
                continue
            list_copy = BeautifulSoup(str(block), "html.parser")
            for nested in list_copy.select("ul, ol"):
                nested.decompose()
            text = clean_text(list_copy.get_text(" ", strip=True))
            min_len = 4
        else:
            text = clean_text(block.get_text(" ", strip=True))
            min_len = 30
        if not text or len(text) < min_len:
            continue
        if title and text.lower() == title.lower():
            continue
        if text in seen:
            continue
        seen.add(text)
        if block.name == "li" and text[-1] not in ".!?;:":
            text += "."
        parts.append(text)

    return clean_text(" ".join(parts))


def _extract_ria_article_text(soup) -> str:
    """Извлекает основной текст РИА из div.article__text без меню сайта.

    РИА размечает текстовые блоки div, а не тегами <article>/<p>.
    Не используем все параграфы страницы: там много меню и рекомендаций.
    """
    # Ограничиваемся телом статьи, если контейнер присутствует.
    body = soup.select_one(".article__body")
    root = body if body is not None else soup
    parts, seen = [], set()
    for block in root.select(".article__text"):
        if block.find_parent(
            class_=re.compile(
                r"article__(?:related|inject|recommend|advert|sidebar|promo)",
                re.I,
            )
        ):
            continue
        paragraph = clean_text(block.get_text(" ", strip=True))
        if len(paragraph) < 20 or paragraph in seen:
            continue
        seen.add(paragraph)
        parts.append(paragraph)
    text = clean_text(" ".join(parts))
    return text if len(text) >= 60 else ""


def _extract_main_text(soup, source_name, title, jsonld):
    """
    1. JSON-LD, если автор сайта указал articleBody.
    2. Основной блок по селекторам сайта.
    3. Ранее работавшие экстракторы, если структура сайта другая.

    Для НедраДВ НЕ используем запасной широкий поиск: там он
    гарантированно смешивает несколько соседних публикаций.
    """
    if source_name in MILPRESS_SOURCES:
        # Реальные страницы Mil.Press имеют articleBody, где только
        # первый абзац размечен <p>, а продолжение идёт текстовыми
        # узлами в <div>. Извлечение только <p> теряло большую часть
        # новости. Контейнер articleBody ограничивает нас статьёй.
        for block in soup.select("[itemprop='articleBody']")[:2]:
            isolated = BeautifulSoup(str(block), "html.parser")
            for noise in isolated.select(
                "nav,aside,footer,script,style,"
                "[class*='related'],[class*='recommend'],"
                "[class*='share'],[class*='social']"
            ):
                noise.decompose()
            text = clean_text(isolated.get_text(" ", strip=True))
            if text and len(text) >= 200:
                return text
        # Старые Mil.Press шаблоны без itemprop.
        for block in soup.select("article.detail")[:2]:
            text = _article_paragraphs(block, title)
            if len(text) >= 200:
                return text

    if source_name == "РИА Новости":
        text = _extract_ria_article_text(soup)
        if text:
            return text

    if source_name == "НедраДВ":
        return extract_text_nedradv(soup, title=title)
    if source_name == "ЛУКОЙЛ":
        return _lukoil_main_body(soup, title=title)
    if source_name == "Коммерсантъ — Промышленность":
        return extract_text_kommersant(soup, title=title)

    body = jsonld.get("articleBody", "")
    jsonld_text = (
        clean_text(body)
        if isinstance(body, str) and len(body.strip()) >= 140
        else ""
    )

    selectors = _ARTICLE_SELECTORS.get(source_name, ())
    selectors += (
        "[itemprop='articleBody']", ".entry-content",
        ".post-content", ".article-body", "article",
    )

    # У части CMS JSON-LD articleBody содержит только абзацы,
    # тогда как технические характеристики и перечисления находятся
    # в HTML <li>. Предпочитаем полный структурированный DOM только
    # когда его объём сопоставим с JSON-LD, чтобы не потерять текст.
    if jsonld_text:
        for selector in dict.fromkeys(selectors):
            for block in soup.select(selector)[:4]:
                if not block.select("ul li, ol li"):
                    continue
                structured = _article_paragraphs(block, title)
                if len(structured) >= max(140, int(len(jsonld_text) * 0.8)):
                    return structured
        return jsonld_text

    for selector in dict.fromkeys(selectors):
        for block in soup.select(selector)[:4]:
            text = _article_paragraphs(block, title)
            if len(text) >= 140:
                return text

    # Обязательный fallback: предыдущая версия из этих HTML извлекала
    # полноценные тексты. Без него 8 работавших источников стали пустыми.
    # Вызовы делаются на отдельной копии: некоторые функции удаляют узлы.
    legacy_extractors = {
        "ЛУКОЙЛ": extract_text_lukoil,
        "Заводы.рф": extract_text_zavody,
        "Производства.рф": extract_text_proizvodstva,
        "ТПП РФ": extract_text_tpprf,
        "Добывающая промышленность": extract_text_dprom,
    }
    legacy_extractor = legacy_extractors.get(source_name, extract_text_generic)
    legacy_soup = BeautifulSoup(str(soup), "html.parser")

    # Убираем боковые блоки и рекомендации до запуска широкого fallback.
    for node in legacy_soup.select(
        "nav, aside, footer, [class*='related'], [class*='sidebar'], "
        "[class*='recommend'], [class*='share'], [class*='social'], "
        "[class*='advert'], [class*='subscribe'], [class*='comments']"
    ):
        node.decompose()

    text = legacy_extractor(legacy_soup, title=title)
    if text and len(text) >= 120:
        return text

    # Последняя попытка: контейнер непосредственно рядом с заголовком.
    h1 = soup.find("h1")
    if h1:
        parent = h1.parent
        for _ in range(3):
            if parent is None:
                break
            if len(parent.select("a[href]")) <= 5:
                text = _article_paragraphs(parent, title)
                if len(text) >= 140:
                    return text
            parent = parent.parent

    return ""

def _strip_source_boilerplate(text: str, source_name: str) -> str:
    """Убирает подтверждённые служебные вставки ПО КРАЯМ статьи.

    Не ищем универсальные стоп-слова по всему тексту: «выставка»,
    «источник», «проект» могут быть частью значимых фактов.
    """
    if not text:
        return ""

    if source_name in MILPRESS_SOURCES:
        # В HTML после новости размещается постоянная подпись
        # MilPressWeekly (в слове «интeресные» иногда латинская e).
        # Удаляем только подтверждённый хвост, не затрагивая тело статьи.
        text = re.sub(
            r"\s+Не хотите пропускать инт[еe]ресные материалы\?"
            r".{0,230}?t\.me/MilPressWeekly\.?\s*$",
            "", text, count=1, flags=re.I | re.S,
        )
        text = re.sub(r"\s+Версия для печати\s*$", "", text, flags=re.I)

    if source_name == "Вестник промышленности":
        # В реальном JSON 06.11.2026 три статьи начинаются «Новости.».
        text = re.sub(r"^\s*Новости\.\s+", "", text, count=1, flags=re.I)

    elif source_name == "НедраДВ":
        # Страница уже передаёт published_at, а сам текст начинается
        # дополнительным "06 ноября 2026." и содержит датлайн NEDRADV.
        text = re.sub(
            r"^\s*\d{1,2}\s+[а-яё]+\s+\d{4}\.\s+",
            "", text, count=1, flags=re.I,
        )
        text = re.sub(
            r"\b\d{1,2}\s+[а-яё]+\s+\d{4}\s+года,\s*NEDRADV\.\s*",
            "", text, count=1, flags=re.I,
        )

    elif source_name == "ИАПН":
        # Последняя строка «Источник новости: пресс-служба...» —
        # указание первоисточника, а не абзац самой статьи.
        text = re.sub(
            r"\s+Источник новости:\s*[^.!?]{1,130}$",
            "", text, count=1, flags=re.I,
        )

    elif source_name == "DixiNews — Промышленность":
        # Длинная обязательная юридическая оговорка про Meta* на
        # конце статьи Apple — не часть новости об акциях компаний.
        # Удаляем только конкретную оговорку ПОСЛЕ основного текста.
        notice = re.search(r"\s*\*\s*Деятельность\s+Meta\s*\*", text, re.I)
        if notice and notice.start() >= 100:
            # Юридическая оговорка иногда длиннее половины
            # короткой статьи, поэтому процент от длины ненадёжен.
            tail = text[notice.start():].lower()
            if "экстремист" in tail and "запрещен" in tail:
                text = text[:notice.start()]

    elif source_name == "Добывающая промышленность":
        # Рекламная ссылка на чужой проект в конце статьи о калориферах:
        # «Пермский край, м.о. Добрянский, ... Перейти на страницу
        # проекта "Рудник. Урал 2026"».
        promo = re.search(
            r"\s+Перейти на страницу проекта\b.{0,140}$",
            text, re.I,
        )
        if promo and promo.start() > len(text) * 0.6:
            text = text[:promo.start()]
            text = re.sub(
                r"\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)*\s+край,"
                r"\s*м\.о\..{0,120}$",
                "", text, flags=re.I,
            )

    return text.strip()


def build_full_article(preview: ArticlePreview) -> Article:
    html = fetch_html(preview.url)
    title = preview.title
    text, text_source = "", "none"
    published_at = preview.published_at

    if html:
        soup = BeautifulSoup(html, "html.parser")
        jsonld = _jsonld_article(soup)

        # РИА: диагностика уже загруженной страницы без дополнительных
        # запросов. Нужна, если RSS отдаёт ссылки, но тело не извлекается.
        if preview.source == "РИА Новости":
            audit = _FETCH_AUDIT.get()
            if audit is not None and len(audit.get("article_probes", [])) < 3:
                h1_probe = soup.find("h1")
                audit.setdefault("article_probes", []).append({
                    "url": preview.url,
                    "html_chars": len(html),
                    "visible_text_chars": len(soup.get_text(" ", strip=True)),
                    "visible_text_start": soup.get_text(" ", strip=True)[:240],
                    "h1": clean_text(h1_probe.get_text(" ", strip=True))[:150] if h1_probe else "",
                    "jsonld_body_chars": len(str(jsonld.get("articleBody", "") or "")),
                    "article_nodes": len(soup.select("article")),
                    "article_body_nodes": len(soup.select("[itemprop='articleBody']")),
                    "ria_text_blocks": len(soup.select(".article__text")),
                    "ria_text_chars": sum(len(x.get_text(" ", strip=True)) for x in soup.select(".article__text")),
                    "preview_chars": len(preview.preview_text or ""),
                })

        if jsonld.get("headline"):
            title = clean_text(str(jsonld["headline"]))
        elif preview.source == "ЛУКОЙЛ":
            title = extract_lukoil_title(soup, fallback_title=title)
        else:
            h1 = soup.find("h1")
            if h1:
                candidate = clean_text(h1.get_text(" ", strip=True))
                if len(candidate) >= 10:
                    title = candidate

        # Не удаляем scripts до чтения JSON-LD и метаданных даты.
        published_at = (
            published_at
            # Для Ростеха важно сначала объединить дату + время из
            # одного контейнера. _page_date иначе берёт первое span.date
            # и возвращает полночь вместо реального времени публикации.
            or _source_specific_published_date(soup, preview.source)
            or _page_date(soup, jsonld)
            or _source_article_headline_date(soup, preview.source, title)
        )
        text = _extract_main_text(soup, preview.source, title, jsonld)
        if text:
            text = _strip_source_boilerplate(text, preview.source)
            text_source = "html"

        # Роснефть: служебная подпись пресс-службы и дата рядом с ней
        # не являются частью основного текста новости.
        if preview.source == "Роснефть" and text:
            text = re.split(
                r"(?i)\bДепартамент информации и рекламы\b",
                text,
                maxsplit=1,
            )[0].strip()

        # Норникель добавляет в конец пресс-релизов постоянную справку,
        # которая не относится к конкретной новости.
        if preview.source == "Норильский никель" and text:
            text = re.split(
                r"(?i)\bИНФОРМАЦИЯ О КОМПАНИИ\b",
                text,
                maxsplit=1,
            )[0].strip()

        # EPRussia: дата в конце статьи часто записана обычным текстом.
        if preview.source == "Энергетика и промышленность России" and text:
            date_match = re.search(
                r"(?i)\bОпубликовано:\s*(\d{1,2}\s+[а-яё]+\s+\d{4}\s*,?\s*\d{1,2}:\d{2})",
                text,
            )
            if date_match:
                published_at = published_at or _parse_published(date_match.group(1))
                text = clean_text(
                    text[:date_match.start()] + text[date_match.end():]
                )
            text = re.sub(
                r"(?i)Все новости отрасли в канале.+?в МАХ",
                " ",
                text,
            ).strip()

    if not text and preview.preview_text:
        text = clean_text(preview.preview_text)
        text_source = "rss"
    if not text:
        if preview.source == "РИА Новости":
            audit = _FETCH_AUDIT.get()
            if audit is not None and len(audit.get("empty_article_examples", [])) < 5:
                audit.setdefault("empty_article_examples", []).append({
                    "url": preview.url,
                    "rss_title": preview.title,
                    "rss_summary_chars": len(preview.preview_text or ""),
                    "html_loaded": bool(html),
                })
        print(f"Нет достоверного текста, пропуск: {preview.url}")

    if preview.source == "Добывающая промышленность":
        title = re.sub(
            r"\s*\|\s*Добывающая промышленность\s*$",
            "",
            title,
            flags=re.IGNORECASE,
        )

    return Article(
        source=preview.source,
        title=normalize_title(title, preview.source),
        url=preview.url,
        published_at=published_at,
        text=text,
        text_source=text_source,
    )

def collect_news(
    sources: list,
    limit_per_source: int = 10,
    session=None,
    diagnostics: dict | None = None,
    allow_unverified_tls: bool = False,
    apply_filters: bool = True,
) -> list[Article]:
    """
    Собирает статьи (БД не изменяет). Если передан diagnostics, сохраняет
    статистику по каждому источнику для анализа ошибок.
    apply_filters=False: техническая проверка сбора без тематического отбора.
    """
    all_articles = []
    seen_urls = set()
    # У Mil.Press одна и та же публикация доступна на нескольких
    # доменах. Канонический ключ — путь статьи, а не домен-зеркало.
    seen_milpress_article_paths = set()

    for source in sources:
        stat = {
            "candidates": 0, "accepted": 0, "short_or_empty": 0,
            "errors": [], "rejected_urls": [],
            "fetch_failures": [], "status": "not_started",
        }
        if diagnostics is not None:
            diagnostics[source.name] = stat

        audit_token = _FETCH_AUDIT.set(stat)
        unverified_token = _ALLOW_UNVERIFIED_TLS.set(
            bool(allow_unverified_tls) and source.name in ("Ростех", "Роснефть")
        )
        try:
            try:
                if source.source_type == "rss":
                    # RSS-кандидаты тоже нужно проверять глубже: первая
                    # подходящая по ключевым словам запись может оказаться
                    # непрофильной или не иметь полного текста.
                    rss_preview_limit = (
                        min(max(20, limit_per_source * 10), 60)
                        if source.name in {"РИА Новости", "ТАСС", "Интерфакс"}
                        else limit_per_source
                    )
                    previews = collect_rss_previews(
                        source, limit=rss_preview_limit,
                        apply_filters=apply_filters,
                    )
                elif source.source_type == "html":
                    collectors = {
                        "Росатом": collect_rosatom_previews,
                        "Вестник промышленности": collect_vestnikprom_previews,
                        "MASHNEWS": collect_mashnews_previews,
                        "Коммерсантъ — Промышленность": collect_kommersant_industry_previews,
                        "DixiNews — Промышленность": collect_dixinews_industry_previews,
                        "НедраДВ": collect_nedradv_previews,
                        "ТПП РФ": collect_tpprf_previews,
                        "Добывающая промышленность": collect_dprom_previews,
                        "Ростех": collect_rostec_previews,
                        "Роснефть": collect_rosneft_previews,
                        **{name: collect_milpress_previews for name in MILPRESS_SOURCES},
                    }
                    fn = collectors.get(source.name)
                    if fn is None:
                        fn = (
                            collect_html_previews
                            if source.name in _SOURCE_ARTICLE_PATHS
                            else _legacy_collect_html_previews
                        )
                    preview_limit = (
                        min(max(24, limit_per_source * 12), 40)
                        if source.name in MILPRESS_SOURCES
                        else min(limit_per_source * 3, 30)
                    )
                    previews = fn(source, limit=preview_limit)
                else:
                    message = f"Неизвестный тип источника: {source.source_type}"
                    stat["errors"].append(message)
                    print(f"{source.name}: {message}")
                    continue
            except Exception as error:
                message = f"Ошибка списка: {type(error).__name__}: {error}"
                stat["errors"].append(message)
                print(f"{source.name}: {message}")
                continue

            stat["candidates"] = len(previews)
            print(f"Кандидатов из {source.name}: {len(previews)}")
            if not previews:
                print(f"ВНИМАНИЕ: {source.name} не дал ссылок на статьи.")

            for preview in previews:
                if not preview.url or preview.url in seen_urls:
                    continue
                seen_urls.add(preview.url)

                if session is not None:
                    try:
                        from database.repository import news_url_exists
                        if news_url_exists(session, preview.url):
                            continue
                    except Exception as error:
                        message = f"Ошибка проверки БД: {type(error).__name__}: {error}"
                        stat["errors"].append(message)
                        print(message)
                        continue

                # У общих СМИ одна и та же CDN-защита обычно блокирует
                # все статьи. После подтверждённой пустой страницы
                # не тратим десятки браузерных запросов на остальные.
                if source.name in {"РИА Новости", "ТАСС"} and any(
                    item.get("kind") == "site_shell_or_access_denied"
                    for item in stat["fetch_failures"]
                ):
                    print(f"{source.name}: ограничение доступа к статьям; останавливаем запросы")
                    break

                try:
                    article = build_full_article(preview)
                except Exception as error:
                    message = f"{preview.url}: {type(error).__name__}: {error}"
                    stat["errors"].append(message)
                    print(f"Ошибка парсинга: {message}")
                    continue

                # Ограничиваем частоту запросов и для отсеянных публикаций.
                # Ранее continue в фильтрах пропускал паузу.
                time.sleep(1)

                min_text_length = (
                    350 if source.name == "Коммерсантъ — Промышленность"
                    else 250 if source.name in REACTIVATED_SOURCES
                    else 120
                )
                # При чистой проверке сбора не скрываем короткие статьи:
                # длина текста остаётся видна в JSON, а решение принимает человек.
                # Пустая страница не является статьёй даже в raw-режиме.
                if not article.text or not article.text.strip() or (
                    apply_filters and len(article.text.strip()) < min_text_length
                ):
                    stat["short_or_empty"] += 1
                    stat["rejected_urls"].append(preview.url)
                    print(f"Не извлечён достаточный основной текст: {preview.url}")
                    continue

                # У новых источников нельзя прерывать поиск на дайджесте
                # или нерелевантной статье: она всё равно не пройдёт
                # последующий фильтр и займёт слот лимита.
                if apply_filters and source.name in REACTIVATED_SOURCES:
                    from processing.topic_filter import classify_article
                    decision = classify_article(article)
                    if not decision.accepted:
                        stat.setdefault("filtered_out", []).append({
                            "url": article.url, "reason": decision.reason,
                        })
                        continue

                if apply_filters and source.name in MILPRESS_SOURCES and is_milpress_consumer_news(
                    article.title
                ):
                    stat.setdefault("filtered_out", []).append({
                        "url": article.url, "reason": "consumer_price_story",
                    })
                    continue

                if apply_filters and source.name in MILPRESS_SOURCES and clearly_foreign_milpress_article(
                    article.title, article.text
                ):
                    stat.setdefault("filtered_out", []).append({
                        "url": article.url, "reason": "foreign_without_russian_connection",
                    })
                    continue

                # При чистом сборе показываем даже зеркальные публикации
                # разных источников: дедупликация относится к обработке.
                if apply_filters and source.name in MILPRESS_SOURCES:
                    from urllib.parse import unquote
                    mirror_key = unquote(urlparse(article.url).path).casefold().rstrip("/")
                    if mirror_key in seen_milpress_article_paths:
                        stat["duplicate_milpress"] = (
                            stat.get("duplicate_milpress", 0) + 1
                        )
                        continue
                    seen_milpress_article_paths.add(mirror_key)

                all_articles.append(article)
                stat["accepted"] += 1
                if stat["accepted"] >= limit_per_source:
                    break

            print(
                f"{source.name}: успешно {stat['accepted']}/"
                f"{stat['candidates']}, не извлечено {stat['short_or_empty']}"
            )
        finally:
            _ALLOW_UNVERIFIED_TLS.reset(unverified_token)
            _FETCH_AUDIT.reset(audit_token)
            if stat["accepted"]:
                stat["status"] = "ok"
            elif stat["fetch_failures"]:
                stat["status"] = "fetch_failed"
                # Сохраняем краткие причины в привычном поле errors,
                # даже если fetch_html() возвратил пустую строку.
                reasons = dict.fromkeys(
                    f"{e['kind']}: {e['url']}" for e in stat["fetch_failures"]
                )
                stat["errors"].extend(list(reasons)[:15])
            elif stat.get("duplicate_milpress"):
                stat["status"] = "duplicates_only"
            elif stat["candidates"]:
                stat["status"] = "article_extraction_failed"
            else:
                stat["status"] = "no_article_links"

    return all_articles

def debug_collect_news(
    sources: list,
    limit_per_source: int = 3,
    text_preview_length: int = 1200,
):
    articles = collect_news(
        sources=sources,
        limit_per_source=limit_per_source
    )

    print("\n")
    print("=" * 140)
    print(f"СОБРАНО НОВОСТЕЙ: {len(articles)}")
    print("=" * 140)

    for index, article in enumerate(articles, start=1):
        print("\n" + "-" * 140)
        print(f"[{index}] {article.title}")
        print(f"\nИсточник: {article.source}")
        print(f"\nURL:")
        print(article.url)
        print(f"\nДата:")
        print(article.published_at)
        print(f"\nИсточник текста:")
        print(article.text_source)
        print(f"\nДлина текста:")
        print(len(article.text))

        preview = article.text[:text_preview_length]

        print(f"\nТекст:")
        print(preview)

        if len(article.text) > text_preview_length:
            print("...")

        print("-" * 140)


if __name__ == "__main__":
    from database.db import get_session
    from database.models import Source

    session = get_session()

    sources = (
        session.query(Source)
        .filter(Source.is_active == True)
        .all()
    )

    debug_collect_news(
        sources=sources,
        limit_per_source=2
    )

    session.close()