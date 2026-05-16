import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import time
import feedparser
import requests
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
            timeout=20,
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

            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(2500)

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
    ]

    if any(domain_name in domain for domain_name in browser_domains):
        return fetch_html_with_browser(url)

    return fetch_html_with_requests(url)


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
        "div[class*='content'] p",
        "div[class*='text'] p",
        "div[class*='article'] p",
    ]

    return extract_by_selectors(soup, selectors, min_len=250, title=title)

def extract_text_proizvodstva(soup: BeautifulSoup, title: str | None = None) -> str:
    selectors = [
        "article p",
        ".post-content p",
        ".entry-content p",
        ".wp-block-post-content p",
        "main p",
        "div[class*='content'] p",
    ]

    return extract_by_selectors(soup, selectors, min_len=200, title=title)

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

        if len(text) >= 250:
            return text

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
            or "flotprom.ru" in domain
            or "отраслевое.рф" in domain
            or "flot.com" in domain
    ):
        return extract_text_military_media(soup, title=title)

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
        title = clean_text(a.get_text(" ", strip=True))
        href = a.get("href")

        if not title or not href or len(title) < 15:
            continue

        absolute_url = urljoin(source.url, href)

        if not same_domain(absolute_url, source.url):
            continue

        path = get_path(absolute_url)

        if not path.startswith(required_prefix.rstrip("/")):
            continue

        if exclude_exact and path == exclude_exact.rstrip("/"):
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
    full_text = extract_article_text(preview.url, title=preview.title)
    text_source = "html"

    if not full_text and preview.preview_text:
        full_text = normalize_article_text(preview.preview_text, title=preview.title)
        text_source = "rss"
    elif not full_text:
        text_source = "none"
        print(f"Не удалось извлечь текст: {preview.url}")

    return Article(
        source=preview.source,
        title=normalize_title(preview.title, preview.source),
        url=preview.url,
        published_at=preview.published_at,
        text=full_text,
        text_source=text_source,
    )


def collect_news(
    sources: list,
    limit_per_source: int = 10,
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


            else:

                previews = collect_html_previews(source, limit=limit_per_source)
        else:
            continue

        print(f"Кандидатов из источника {source.name}: {len(previews)}")

        for preview in previews:
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
    """
    Временная отладочная функция.
    Показывает:
    - заголовок
    - ссылку
    - дату
    - источник текста
    - длину текста
    - начало текста
    """

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

def cut_related_news(text: str) -> str:
    markers = [
        "Главное в отраслевых СМИ",
        "Читайте также",
        "Другие материалы",
        "Материалы по теме",
        "Новости по теме",
        "Еще по теме",
        "Смотрите также",
        "Популярное",
        "Последние новости",
        "Ранее сообщалось",
        "Подписывайтесь",
        "Печатная версия статьи",
        "Печатная версия",
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

def extract_text_proizvodstva(soup: BeautifulSoup, title: str | None = None) -> str:
    selectors = [
        "article p",
        ".post-content p",
        ".entry-content p",
        ".wp-block-post-content p",
        "main p",
        "div[class*='content'] p",
    ]

    return extract_by_selectors(soup, selectors, min_len=200, title=title)


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

    return extract_by_selectors(soup, selectors, min_len=180, title=title)


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

        if len(text) >= 250:
            return text

    return ""

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
        lambda tag: tag.name in ["h2", "h3"]
        and "Промышленные новости" in tag.get_text(" ", strip=True)
    )

    if not heading:
        print("Блок 'Промышленные новости' не найден")
        return previews

    current = heading.find_next()

    while current and len(previews) < limit:
        text = current.get_text(" ", strip=True)

        # Останавливаемся, когда дошли до следующего крупного раздела
        if current.name in ["h2", "h3"] and "Промышленные новости" not in text:
            break

        links = current.select("a[href]")

        for a in links:
            title = clean_text(a.get_text(" ", strip=True))
            href = a.get("href")

            if not title or not href or len(title) < 15:
                continue

            absolute_url = urljoin(source.url, href)

            if not same_domain(absolute_url, source.url):
                continue

            if absolute_url in seen_urls:
                continue

            # Берём только записи из категории "Новости"
            parent_text = current.get_text(" ", strip=True)
            if "Новости" not in parent_text and "/category/news" not in absolute_url:
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

        current = current.find_next_sibling()

    print(f"Найдено HTML-кандидатов: {len(previews)}")

    return previews

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

        # запасной вариант: берём все p, но фильтруем мусор
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
        ]

        paragraphs = filter_paragraphs(paragraphs, title=title)
        text = normalize_article_text(" ".join(paragraphs), title=title)

        if len(text) >= 120:
            return text

        return ""