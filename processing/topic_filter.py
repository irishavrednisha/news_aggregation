BAD_TITLE_KEYWORDS = [
    "дайджест",
    "главное в отраслевых сми",
    "поздравляем",
    "поздравление",
    "с днем",
    "с днём",
    "итоги недели",
    "за неделю",
    "обзор сми",
    "подборка",
    "анонс",
    "фоторепортаж",
    "видео",
    "новости и медиа",
]

BAD_CONTENT_KEYWORDS = [
    "свидетельство о регистрации сми",
    "политика конфиденциальности",
    "согласие на обработку",
    "персональных данных",
    "юридическая информация",
    "внутренние документы",
    "корпоративное управление",
    "личный кабинет",
    "тендеры",
    "вакансии",
]

BAD_URL_PARTS = [
    "/contacts",
    "/contact",
    "/about",
    "/privacy",
    "/policy",
    "/personal",
    "/governance",
    "/career",
    "/personnel-policy",
    "/tenders",
    "/legal",
    "/documents",
    "/internaldocuments",
    "/contractor",
    "/sustainability",
]


def normalize(text: str) -> str:
    if not text:
        return ""
    return text.lower().replace("ё", "е")


def has_bad_title(title: str) -> bool:
    title = normalize(title)
    return any(word in title for word in BAD_TITLE_KEYWORDS)


def has_bad_url(url: str) -> bool:
    url = normalize(url)
    return any(part in url for part in BAD_URL_PARTS)


def has_bad_content(text: str) -> bool:
    text = normalize(text)
    return any(word in text for word in BAD_CONTENT_KEYWORDS)


def is_valid_article(article, min_text_len: int = 250) -> bool:
    """
    Проверяет, можно ли считать материал нормальной новостью.
    """
    if not article.title or not article.url:
        return False

    if has_bad_title(article.title):
        return False

    if has_bad_url(article.url):
        return False

    if not article.text or len(article.text) < min_text_len:
        return False

    if has_bad_content(article.text):
        return False

    return True


def filter_articles(articles: list) -> list:
    """
    Оставляет только нормальные новости.
    """
    return [article for article in articles if is_valid_article(article)]


def print_filter_report(articles: list, filtered: list):
    print("\n" + "=" * 100)
    print("Результаты фильтрации")
    print(f"Всего собрано: {len(articles)}")
    print(f"Оставлено: {len(filtered)}")
    print(f"Удалено: {len(articles) - len(filtered)}")

    print("\nОСТАВЛЕННЫЕ НОВОСТИ:")
    print("-" * 100)
    for article in filtered:
        print(f"[OK] {article.source}: {article.title}")
        print(article.url)

    print("\nУДАЛЁННЫЕ МАТЕРИАЛЫ:")
    print("-" * 100)
    for article in articles:
        if article not in filtered:
            print(f"[NO] {article.source}: {article.title}")
            print(article.url)