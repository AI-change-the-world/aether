from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

sql_url: Optional[str] = None
engine = None
SessionLocal = None
Base = declarative_base()


def _create_engine(url: str):
    return create_engine(url, echo=False)


def get_engine():
    global engine, sql_url

    if engine:
        return engine

    # 如果没设置 URL，fallback 到 SQLite
    if not sql_url:
        sql_url = "sqlite:///./aether.db"

    engine = _create_engine(sql_url)
    return engine


def get_session():
    global SessionLocal
    if SessionLocal is None:
        SessionLocal = sessionmaker(
            bind=get_engine(), autocommit=False, autoflush=False
        )
    return SessionLocal()
