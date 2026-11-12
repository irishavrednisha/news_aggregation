"""Find main ARTICLE image candidates from already fetched HTML.

No image downloads here. Do not use list-page thumbnails or unrelated assets.
Rights are never presumed from the ability to download an image.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup


@dataclass(frozen=True)
class ImageCandidate:
    url: str
    caption: str = ""
    credit: str = ""
    license_url: str = ""
    origin: str = ""


_BAD_TOKENS = (
    "logo", "avatar", "sprite", "icon", "placeholder", "banner",
    "advert", "tracking", "pixel", "favicon", "counter", "blank",
)
_CONTENT_SELECTORS = (
    "article", "[itemprop='articleBody']", ".article-content",
    ".article__content", ".article__body", ".entry-content",
    ".post-content", ".news-detail", "main",
)


def _absolute(url: str | None, base_url: str) -> str:
    candidate = (url or "").strip()
    if not candidate or candidate.startswith(("data:", "blob:", "#")):
        return ""
    full = urljoin(base_url, candidate)
    parsed = urlsplit(full)
    if parsed.scheme not in ("https", "http") or not parsed.hostname:
        return ""
    if any(token in parsed.path.casefold() for token in _BAD_TOKENS):
        return ""
    if parsed.path.casefold().endswith((".svg", ".gif", ".ico")):
        return ""
    return full


def _attr(node, name: str) -> str:
    if not node:
        return ""
    return str(node.get(name, "") or "").strip()


def _source_from_img(img) -> str:
    for name in ("data-src", "data-original", "data-lazy-src", "src"):
        value = _attr(img, name)
        if value and not value.startswith(("data:", "blob:")):
            return value
    srcset = _attr(img, "data-srcset") or _attr(img, "srcset")
    if srcset:
        # Responsive image candidates are normally small -> large; choose
        # the largest descriptor, not a tiny thumbnail.
        choices = []
        for item in srcset.split(","):
            bits = item.strip().split()
            if not bits:
                continue
            width = 0
            if len(bits) > 1:
                try:
                    width = int(float(bits[-1].rstrip("wx")))
                except ValueError:
                    pass
            choices.append((width, bits[0]))
        if choices:
            return max(choices, key=lambda pair: pair[0])[1]
    return ""


def extract_image_candidates(
    soup: BeautifulSoup,
    article_url: str,
    max_candidates: int = 6,
) -> list[ImageCandidate]:
    """Return distinct, ordered candidates. Caller validates file contents."""
    found: list[ImageCandidate] = []
    seen: set[str] = set()

    def add(raw_url: str, *, origin: str, caption: str = "", credit: str = "",
            license_url: str = "") -> None:
        full = _absolute(raw_url, article_url)
        if not full or full in seen or len(found) >= max_candidates:
            return
        seen.add(full)
        found.append(ImageCandidate(
            url=full,
            caption=caption[:500].strip(),
            credit=credit[:300].strip(),
            license_url=license_url[:1000].strip(),
            origin=origin,
        ))

    # Main preview image selected by article author/CMS.
    for value in ("og:image:secure_url", "og:image", "twitter:image", "twitter:image:src"):
        for tag in soup.select("meta[property], meta[name]"):
            if _attr(tag, "property").casefold() == value or _attr(tag, "name").casefold() == value:
                add(_attr(tag, "content"), origin=value)

    # Article figures normally outrank scattered decorative images.
    container = next((soup.select_one(selector) for selector in _CONTENT_SELECTORS
                      if soup.select_one(selector) is not None), None)
    if container is None:
        return found

    for figure in container.select("figure")[:12]:
        img = figure.select_one("img")
        if img is None:
            continue
        caption_node = figure.select_one("figcaption")
        caption = caption_node.get_text(" ", strip=True) if caption_node else ""
        credit = _attr(img, "data-credit") or _attr(img, "data-author")
        add(_source_from_img(img), origin="figure", caption=caption, credit=credit)

    for img in container.select("img")[:25]:
        alt = _attr(img, "alt").casefold()
        if any(token in alt for token in _BAD_TOKENS):
            continue
        try:
            width = int(_attr(img, "width") or "0")
            height = int(_attr(img, "height") or "0")
        except ValueError:
            width = height = 0
        if (width and width < 300) or (height and height < 180):
            continue
        add(_source_from_img(img), origin="article", caption=_attr(img, "alt"))

    return found
