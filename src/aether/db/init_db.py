from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url

from aether.db import Base, engine, get_engine, sql_url


def create_database_if_not_exists(db_url: str):
    url = make_url(db_url)
    if url.drivername.startswith("mysql"):
        db_name = url.database
        tmp_url = url.set(database="mysql")
        tmp_engine = create_engine(tmp_url)
        with tmp_engine.connect() as conn:
            conn.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{db_name}` DEFAULT CHARACTER SET utf8mb4;"
                )
            )


def init_db(url: str = None):
    from aether import db

    if url:
        db.sql_url = url
        db.engine = None  # 重新初始化

    # 触发创建
    engine = get_engine()
    create_database_if_not_exists(str(engine.url))
    Base.metadata.create_all(bind=engine)
