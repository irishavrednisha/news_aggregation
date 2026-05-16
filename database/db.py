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

    Base.metadata.create_all(bind=engine)


def get_session():
    """
    Создает сессию для работы с БД.
    """
    return SessionLocal()