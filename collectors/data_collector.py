import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import time
import feedparser
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


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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

def fetch_html_with_requests(url: str) -> str:
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=40,
            verify=False
        )

        if response.status_code == 403:
            print(f"Доступ запрещён: {url}")
            return ""

        response.raise_for_status()

        if response.encoding is None or response.encoding.lower() in ["iso-8859-1", "latin-1"]:
            response.encoding = response.apparent_encoding

        return response.text

    except Exception as error:
        print(f"Ошибка загрузки страницы {url}: {error}")
        return ""


def fetch_html_with_browser(url: str) -> str:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)

            page = browser.new_page(
                user_agent=HEADERS["User-Agent"],
                viewport={"width": 1280, "height": 900},
            )

            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000)

            html = page.content()
            browser.close()

            return html

    except Exception as error:
        print(f"Ошибка браузерной загрузки {url}: {error}")
        return ""


def fetch_html(url: str) -> str:
    domain = get_domain(url)

    browser_domains = [
        "nornickel.ru",
        "заводы.рф",
        "xn--80az8a.xn--p1ai",
        "производства.рф",
        "xn--80adahnf5bdekrm.xn--p1ai",

        # ТПП РФ часто не отдаёт страницы новостей через обычный requests
        "tpprf.ru",
        "news.tpprf.ru",
    ]

    if any(domain_name in domain for domain_name in browser_domains):
        return fetch_html_with_browser(url)

    html = fetch_html_with_requests(url)

    # запасной вариант: если requests ничего не вернул,
    # пробуем открыть страницу через браузер
    if not html:
        return fetch_html_with_browser(url)

    return html

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
    for selector in selectors:
        blocks = soup.select(selector)

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
    Извлекает текст статьи с сайта НедраДВ.
    """

    candidates = []

    if title:
        candidates.append(clean_text(title))

    for tag in soup.find_all(["p", "blockquote"]):
        text = clean_text(tag.get_text(" ", strip=True))

        if not text:
            continue

        if len(text) < 40:
            continue

        lower = text.lower()

        bad_fragments = [
            "поделиться",
            "читайте также",
            "последние публикации",
            "последние аукционы",
            "наши контакты",
            "все права защищены",
            "при полном или частичном использовании",
            "настоящий ресурс содержит",
            "подписаться",
            "комментарии для сайта",
        ]

        if any(fragment in lower for fragment in bad_fragments):
            continue

        candidates.append(text)

    return clean_text("\n".join(dict.fromkeys(candidates)))

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

def collect_rss_previews(source, limit: int) -> list[ArticlePreview]:
    previews = []

    if not source.rss_url:
        return previews

    print("=" * 100)
    print(f"RSS-источник: {source.name}")
    print(f"RSS: {source.rss_url}")

    feed = feedparser.parse(source.rss_url)

    if feed.bozo:
        print("RSS прочитан с ошибкой или предупреждением:")
        print(feed.bozo_exception)

    print(f"Найдено записей: {len(feed.entries)}")

    for entry in feed.entries[:limit]:
        title = clean_text(entry.get("title", ""))
        url = entry.get("link", "").strip()
        published_raw = entry.get("published") or entry.get("updated")
        published_at = parse_datetime(published_raw)
        preview_text = extract_rss_text(entry)

        if not title or not url:
            continue

        previews.append(
            ArticlePreview(
                source=source.name,
                title=normalize_title(title, source.name),
                url=url,
                published_at=published_at,
                preview_text=preview_text,
            )
        )

    return previews


# =========================
# Специальные сборщики ссылок
# =========================

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

        if not title or not href or len(title) < 10:
            continue

        absolute_url = urljoin(source.url, href)

        if not same_domain(absolute_url, source.url):
            continue

        if "/doc/" not in absolute_url:
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

def collect_html_previews(source, limit: int) -> list[ArticlePreview]:
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


# =========================
# Сбор полной новости
# =========================

def build_full_article(preview: ArticlePreview) -> Article:
    html = fetch_html(preview.url)

    full_title = preview.title
    full_text = ""
    text_source = "html"

    if html:
        soup = BeautifulSoup(html, "html.parser")
        domain = get_domain(preview.url)

        if "lukoil.ru" in domain:
            page_title = extract_lukoil_title(
                soup=soup,
                fallback_title=preview.title
            )

            if page_title:
                full_title = page_title

        else:
            h1 = soup.find("h1")

            if h1:
                page_title = clean_text(h1.get_text(" ", strip=True))

                if page_title:
                    full_title = page_title

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
            full_text = extract_text_proizvodstva(soup, title=full_title)

        elif "заводы.рф" in domain or "xn--80az8a.xn--p1ai" in domain:
            full_text = extract_text_zavody(soup, title=full_title)

        elif "rosatom.ru" in domain:
            full_text = extract_text_rosatom(soup, title=full_title)

        elif "lukoil.ru" in domain:
            full_text = extract_text_lukoil(soup, title=full_title)

        elif (
            "военное.рф" in domain
            or "xn--b1aga5aadd.xn--p1ai" in domain
            or "flotprom.ru" in domain
            or "отраслевое.рф" in domain
            or "xn--80aegqufhcjg6b.xn--p1ai" in domain
            or "flot.com" in domain
        ):
            full_text = extract_text_military_media(soup, title=full_title)

        elif "nedradv.ru" in domain:
            full_text = extract_text_nedradv(soup, title=full_title)

        elif "tpprf.ru" in domain:
            full_text = extract_text_tpprf(soup, title=full_title)

        elif "dprom.online" in domain:
            full_text = extract_text_dprom(soup, title=full_title)

        else:
            full_text = extract_text_generic(soup, title=full_title)

    if not full_text and preview.preview_text:
        full_text = normalize_article_text(preview.preview_text, title=full_title)
        text_source = "rss"

    elif not full_text:
        text_source = "none"
        print(f"Не удалось извлечь текст: {preview.url}")

    return Article(
        source=preview.source,
        title=normalize_title(full_title, preview.source),
        url=preview.url,
        published_at=preview.published_at,
        text=full_text,
        text_source=text_source,
    )

def collect_news(
    sources: list,
    limit_per_source: int = 10,
    session=None,
) -> list[Article]:
    all_articles = []

    for source in sources:
        if source.source_type == "rss":
            previews = collect_rss_previews(source, limit=limit_per_source)

        elif source.source_type == "html":

            if source.name == "Росатом":
                previews = collect_rosatom_previews(source, limit=limit_per_source)

            elif source.name == "Вестник промышленности":
                previews = collect_vestnikprom_previews(source, limit=limit_per_source)

            elif source.name == "MASHNEWS":
                previews = collect_mashnews_previews(source, limit=limit_per_source)

            elif source.name == "Коммерсантъ — Промышленность":
                previews = collect_kommersant_industry_previews(source, limit=limit_per_source)

            elif source.name == "DixiNews — Промышленность":
                previews = collect_dixinews_industry_previews(source, limit=limit_per_source)

            elif source.name == "НедраДВ":
                previews = collect_nedradv_previews(source, limit=limit_per_source)

            elif source.name == "ТПП РФ":
                previews = collect_tpprf_previews(source, limit=limit_per_source)

            elif source.name == "Добывающая промышленность":
                previews = collect_dprom_previews(source, limit=limit_per_source)

            else:
                previews = collect_html_previews(source, limit=limit_per_source)

        else:
            continue

        print(f"Кандидатов из источника {source.name}: {len(previews)}")

        for preview in previews:
            if session is not None:
                from database.repository import news_url_exists

                if news_url_exists(session, preview.url):
                    print(f"Уже есть в БД, пропускаем парсинг: {preview.url}")
                    continue

            article = build_full_article(preview)

            if article.text:
                all_articles.append(article)

            time.sleep(1)

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