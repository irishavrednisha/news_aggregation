import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from sqlalchemy.orm import Session

from database.models import Source, News


def get_or_create_source(
    session: Session,
    name: str,
    url: str,
    rss_url: str | None = None,
    source_type: str = "html",
    is_active: bool = True
) -> Source:
    source = session.query(Source).filter(Source.name == name).first()

    if source:
        source.url = url
        source.rss_url = rss_url
        source.source_type = source_type
        source.is_active = is_active
        session.commit()
        session.refresh(source)
        return source

    source = Source(
        name=name,
        url=url,
        rss_url=rss_url,
        source_type=source_type,
        is_active=is_active
    )

    session.add(source)
    session.commit()
    session.refresh(source)

    return source


def seed_sources(session: Session):
    default_sources = [
        {
            "name": "ИАПН",
            "url": "https://iapn.ru",
            "rss_url": "https://iapn.ru/feed/",
            "source_type": "rss",
            "is_active": True,
        },
        {
            "name": "Промышленный вестник / ПВ.РФ",
            "url": "https://promvest.info/ru/novosti-promyishlennosti/",
            "rss_url": None,
            "source_type": "html",
            "is_active": True,
        },
        {
            "name": "Производства.рф",
            "url": "https://производства.рф/",
            "rss_url": None,
            "source_type": "html",
            "is_active": True,
        },

        {
            "name": "Ростех",
            "url": "https://rostec.ru/media/news/",
            "rss_url": None,
            "source_type": "html",
            "is_active": True,
        },
        {
            "name": "ОСК",
            "url": "https://www.aoosk.ru/press-center/news/",
            "rss_url": None,
            "source_type": "html",
            "is_active": True,
        },
        {
            "name": "Роснефть",
            "url": "https://www.rosneft.ru/press/news/",
            "rss_url": None,
            "source_type": "html",
            "is_active": True,
        },
        {
            "name": "Заводы.рф",
            "url": "https://заводы.рф/publications",
            "rss_url": None,
            "source_type": "html",
            "is_active": True,
        },
        {
            "name": "Газпром",
            "url": "https://www.gazprom.ru/press/news/",
            "rss_url": None,
            "source_type": "html",
            "is_active": False,
        },
        {
            "name": "ЛУКОЙЛ",
            "url": "https://lukoil.ru/PressCenter/Pressreleases",
            "rss_url": None,
            "source_type": "html",
            "is_active": True,
        },
        {
            "name": "Норильский никель",
            "url": "https://www.nornickel.ru/news-and-media/press-releases-and-news/",
            "rss_url": None,
            "source_type": "html",
            "is_active": True,
        },

        {
            "name": "Военное.рф",
            "url": "https://военное.рф/2026/",
            "rss_url": None,
            "source_type": "html",
            "is_active": True,
        },
        {
            "name": "ФлотПром",
            "url": "https://flotprom.ru/2026/",
            "rss_url": None,
            "source_type": "html",
            "is_active": True,
        },
        {
            "name": "Отраслевое.рф",
            "url": "https://отраслевое.рф",
            "rss_url": None,
            "source_type": "html",
            "is_active": True,
        },
        {
            "name": "ФлотКом",
            "url": "https://flot.com",
            "rss_url": None,
            "source_type": "html",
            "is_active": True,
        },

        {
            "name": "Энергетика и промышленность России",
            "url": "https://www.eprussia.ru",
            "rss_url": None,
            "source_type": "html",
            "is_active": True,
        },
        {
            "name": "Вестник промышленности",
            "url": "https://vestnikprom.ru",
            "rss_url": None,
            "source_type": "html",
            "is_active": True,
        },

        {
            "name": "РИА Новости",
            "url": "https://ria.ru",
            "rss_url": "https://ria.ru/export/rss2/index.xml",
            "source_type": "rss",
            "is_active": False,
        },
        {
            "name": "ТАСС",
            "url": "https://tass.ru",
            "rss_url": "https://tass.ru/rss/v2.xml",
            "source_type": "rss",
            "is_active": False,
        },
        {
            "name": "Интерфакс",
            "url": "https://www.interfax.ru",
            "rss_url": "https://www.interfax.ru/rss.asp",
            "source_type": "rss",
            "is_active": False,
        },
    ]

    for source in default_sources:
        get_or_create_source(
            session=session,
            name=source["name"],
            url=source["url"],
            rss_url=source["rss_url"],
            source_type=source["source_type"],
            is_active=source["is_active"]
        )


def get_active_sources(session: Session) -> list[Source]:
    return (
        session.query(Source)
        .filter(Source.is_active == True)
        .all()
    )


def news_exists(session: Session, url: str) -> bool:
    return session.query(News).filter(News.url == url).first() is not None


def save_news(session: Session, article) -> News | None:
    if news_exists(session, article.url):
        return None

    source = session.query(Source).filter(Source.name == article.source).first()

    if not source:
        return None

    news = News(
        source_id=source.id,
        title=article.title,
        text=article.text,
        url=article.url,
        published_at=article.published_at,
        cluster_id=None,
        is_primary=False
    )

    session.add(news)
    session.commit()
    session.refresh(news)

    return news


def save_articles(session: Session, articles: list) -> int:
    saved_count = 0

    for article in articles:
        saved = save_news(session, article)

        if saved is not None:
            saved_count += 1

    return saved_count