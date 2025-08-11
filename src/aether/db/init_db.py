from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url

from aether.common.logger import logger
from aether.db.basic import Base, engine, get_engine, sql_url


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
    import aether.models
    from aether.db import basic

    if url:
        basic.sql_url = url
        basic.engine = None  # 重新初始化

    # 触发创建
    engine = get_engine()
    logger.info(f"init db, url: {engine.url}")
    create_database_if_not_exists(str(engine.url))
    Base.metadata.create_all(bind=engine)
    logger.info(f"注册的表：{Base.metadata.tables.keys()}",)
