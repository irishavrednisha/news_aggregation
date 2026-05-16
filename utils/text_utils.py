import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import re
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def fix_encoding(text: str) -> str:
    if not text:
        return ""

    if "Р" in text or "С" in text:
        try:
            fixed = text.encode("latin1").decode("utf-8")
            if fixed:
                return fixed
        except Exception:
            pass

    bad_symbols = ["Ì", "Î", "Á", "Â", "Ã", "Ä", "Å", "Æ", "Ç", "È", "É"]
    if any(symbol in text for symbol in bad_symbols):
        try:
            fixed = text.encode("latin1").decode("cp1251")
            if fixed:
                return fixed
        except Exception:
            pass

    return text


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = fix_encoding(text)
    text = BeautifulSoup(text, "html.parser").get_text(" ", strip=True)

    replacements = {
        "\xa0": " ",
        "\u200b": "",
        "\ufeff": "",
        "…": "...",
        "« ": "«",
        " »": "»",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


def normalize_title(title: str, source: str) -> str:
    if not title:
        return ""

    title = clean_text(title)

    title = re.sub(r"^Заголовок:\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^\d{2}\.\d{2}\.\d{4}\s*[-–—]\s*", "", title)

    important_sources = [
        "Военное.рф",
        "ФлотПром",
        "Отраслевое.рф",
        "ФлотКом",
    ]

    if source in important_sources and title.startswith("Важное "):
        title = title[len("Важное "):].strip()

    letters = [c for c in title if c.isalpha()]
    if letters:
        upper_ratio = sum(c.isupper() for c in letters) / len(letters)
        if upper_ratio > 0.8:
            title = title.capitalize()

    return title.strip()


def remove_dateline(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"^[А-ЯЁA-Z][а-яёА-ЯЁA-Z\-\s]+\.?\s*\d{1,2}\s+[а-яё]+\.\s*INTERFAX\.RU\s*[-—]?\s*",
        r"^[А-ЯЁA-Z][а-яёА-ЯЁA-Z\-\s]+,\s*\d{1,2}\s+[а-яё]+\s*[-–—]\s*РИА Новости\.?\s*",
        r"^РИА Новости\.?\s*",
        r"^МОСКВА,\s*\d{1,2}\s+[а-яё]+\s*[-–—]\s*РИА Новости\.?\s*",
        r"^[А-ЯЁA-Z][а-яёА-ЯЁA-Z\-\s]+,\s*\d{1,2}\s+[а-яё]+\.?\s*/ТАСС/\.?\s*",
        r"^[А-ЯЁA-Z][а-яёА-ЯЁA-Z\-\s]+,\s*\d{1,2}\s+[а-яё]+\.?\s*/Корр\. ТАСС[^/]*/\.?\s*",
        r"^/\s*[^.]+?\.\s*",
    ]

    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", text).strip()


def remove_boilerplate(text: str) -> str:
    if not text:
        return ""

    stop_phrases = [
        "Читайте также",
        "Материалы по теме",
        "Другие материалы",
        "Главное в отраслевых СМИ",
        "Новости по теме",
        "Популярное",
        "Последние новости",
        "Еще по теме",
        "Смотрите также",
        "Ранее сообщалось",
        "Подписывайтесь",
        "Подписаться",
        "Печатная версия статьи",
        "Печатная версия",
        "Источник:",
        "Фото:",
        "Реклама",
        "На правах рекламы",
        "Свидетельство о регистрации СМИ",
        "Политика конфиденциальности",
        "Согласие на обработку",
        "На информационном ресурсе применяются рекомендательные технологии",
        "Шрифт Guildenstern",
        "Оригинал: FontSpace",
    ]

    lowered = text.lower()

    cut_positions = []

    for phrase in stop_phrases:
        pos = lowered.find(phrase.lower())
        if pos != -1:
            cut_positions.append(pos)

    if cut_positions:
        text = text[:min(cut_positions)]

    return re.sub(r"\s+", " ", text).strip()


def remove_contacts_and_service_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\+?\d[\d\s().-]{7,}\d", " ", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", " ", text)
    text = re.sub(r"\bwww\.[^\s]+", " ", text)
    text = re.sub(r"\bhttps?://[^\s]+", " ", text)

    text = re.sub(r"\b\d+\s*тел\.?:?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bтел\.?:?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\be-mail:?", " ", text, flags=re.IGNORECASE)

    return re.sub(r"\s+", " ", text).strip()


def remove_repeated_title(text: str, title: str | None = None) -> str:
    if not text:
        return ""

    text = re.sub(r"^(Заголовок:\s*)+", "", text, flags=re.IGNORECASE).strip()

    if title:
        clean_title = normalize_title(title, "")
        escaped = re.escape(clean_title)

        text = re.sub(
            rf"^({escaped})\s+",
            "",
            text,
            flags=re.IGNORECASE
        ).strip()

    return text


def split_sentences(text: str) -> list[str]:
    if not text:
        return []

    return re.split(r"(?<=[.!?])\s+", text)


def remove_duplicate_sentences(text: str) -> str:
    if not text:
        return ""

    sentences = split_sentences(text)
    result = []
    seen = set()

    for sentence in sentences:
        sentence = sentence.strip()

        if len(sentence) < 20:
            continue

        key = re.sub(r"[^а-яёa-z0-9]+", " ", sentence.lower()).strip()

        if key in seen:
            continue

        seen.add(key)
        result.append(sentence)

    return " ".join(result)


def is_bad_paragraph(text: str) -> bool:
    if not text:
        return True

    lowered = text.lower()

    bad_phrases = [
        "читайте также",
        "материалы по теме",
        "другие материалы",
        "главное в отраслевых сми",
        "последние новости",
        "популярное",
        "подписывайтесь",
        "подписаться",
        "реклама",
        "на правах рекламы",
        "свидетельство о регистрации",
        "политика конфиденциальности",
        "согласие на обработку",
        "рекомендательные технологии",
        "fontspace",
        "guildenstern",
        "cookie",
        "cookies",
        "печатная версия статьи",
        "печатная версия",
        "источник новости",
    ]

    if any(phrase in lowered for phrase in bad_phrases):
        return True

    if len(text) < 35:
        return True

    if len(text.split()) < 5:
        return True

    # слишком много ссылок/контактов
    if text.count("@") > 0 or text.count("www.") > 0:
        return True

    # меню и навигация часто состоят из коротких фраз через |
    if text.count("|") >= 2:
        return True

    return False


def normalize_article_text(text: str, title: str | None = None) -> str:
    text = clean_text(text)
    text = remove_urls(text)
    text = remove_repeated_title(text, title)
    text = remove_boilerplate(text)
    text = remove_dateline(text)
    text = remove_contacts_and_service_text(text)
    text = cut_related_titles_tail(text)
    text = remove_duplicate_sentences(text)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    return text.strip()


def filter_paragraphs(paragraphs: list[str], title: str | None = None) -> list[str]:
    result = []
    seen = set()

    for paragraph in paragraphs:
        text = normalize_article_text(paragraph, title=title)

        if is_bad_paragraph(text):
            continue

        key = re.sub(r"[^а-яёa-z0-9]+", " ", text.lower()).strip()

        if key in seen:
            continue

        seen.add(key)
        result.append(text)

    return result

def cut_related_titles_tail(text: str) -> str:
    if not text:
        return ""

    markers = [
        "Другие новости",
        "Еще новости",
        "Ещё новости",
        "Похожие новости",
        "Читайте также",
        "Читайте по теме",
        "Новости по теме",
        "Материалы по теме",
        "Рекомендуем",
    ]

    lowered = text.lower()

    for marker in markers:
        pos = lowered.find(marker.lower())
        if pos != -1:
            return text[:pos].strip()

    # Частый случай: после нормального текста подряд идут 3+ заголовка без точек.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    clean_sentences = []

    for sentence in sentences:
        words = sentence.split()

        # Заголовки похожих статей обычно короткие, без точки внутри,
        # начинаются с заглавной буквы и идут подряд.
        if len(words) <= 10 and not sentence.endswith((".", "!", "?")):
            continue

        clean_sentences.append(sentence)

    return " ".join(clean_sentences).strip()

def remove_urls(text: str) -> str:
    if not text:
        return ""

    # http:// https://
    text = re.sub(r"https?://\S+", " ", text)

    # www.
    text = re.sub(r"www\.\S+", " ", text)

    # ссылки без протокола
    text = re.sub(
        r"\b[a-zA-Z0-9.-]+\.(ru|com|net|org|рф|info|io|biz)\S*",
        " ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()