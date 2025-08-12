import json

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
    import aether.models as models
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

    from sqlalchemy.orm import Session

    from aether.api.tool_type import ToolType
    from aether.models.tool.tool_crud import AetherToolCRUD

    with Session(engine) as session:
        # 检查是否已经有数据
        exists = session.query(models.AetherTool).count()
        if exists == 0:
            logger.info("插入默认Task数据...")
            AetherToolCRUD.create(
                session,
                {
                    "tool_name": ToolType.FETCH_RESULT,
                    "tool_config": json.dumps(
                        {"tool_type": ToolType.FETCH_RESULT, "description": "获取异步任务结果"},
                        ensure_ascii=False,
                    ),
                },
            )
            AetherToolCRUD.create(
                session,
                {
                    "tool_name": ToolType.REGISTER_MODEL,
                    "tool_config": json.dumps(
                        {
                            "tool_type": ToolType.REGISTER_MODEL,
                            "description": "注册新模型到系统",
                        },
                        ensure_ascii=False,
                    ),
                },
            )
            AetherToolCRUD.create(
                session,
                {
                    "tool_name": ToolType.GET_TOOLS,
                    "tool_config": json.dumps(
                        {
                            "tool_type": ToolType.GET_TOOLS,
                            "description": "获取当前系统中所有可用工具",
                        },
                        ensure_ascii=False,
                    ),
                    "req": json.dumps({"like": {"type": "string", "default": "..."}}),
                },
            )
        else:
            logger.info("用户表已有数据，跳过初始化。")
