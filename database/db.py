import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from config import DATABASE_URL


engine = create_engine(
    DATABASE_URL,
    echo=False
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

Base = declarative_base()


def create_tables():
    """
    Создает таблицы в базе данных.
    """
    from database.models import Source, News, Cluster, Post

    # Legacy tables only: do NOT create new moderation/image tables on startup.
    # They must be installed through the validated Alembic migration, after backup.
    Base.metadata.create_all(
        bind=engine,
        tables=[Source.__table__, News.__table__, Cluster.__table__, Post.__table__],
    )


def get_session():
    """
    Создает сессию для работы с БД.
    """
    return SessionLocal()