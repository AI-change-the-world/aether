from sqlalchemy import TIMESTAMP, Column, Integer, SmallInteger, String, func

from aether.db import Base


class AetherTask(Base):
    __tablename__ = "aether_task"
    __table_args__ = {"comment": "task table"}

    aether_task_id = Column(
        Integer, primary_key=True, autoincrement=True, comment="aether task id"
    )
    # req_task_id = Column(Integer, nullable=True, comment="request task id")
    task_type = Column(String, default="", nullable=True, comment="must in __TASKS__")
    created_at = Column(TIMESTAMP, server_default=func.now(), comment="创建时间")
    updated_at = Column(
        TIMESTAMP, server_default=func.now(), onupdate=func.now(), comment="更新时间"
    )
    is_deleted = Column(SmallInteger, default=0, comment="逻辑删除标记")
    status = Column(
        SmallInteger,
        default=0,
        comment="任务状态, 0 pre task, 1 on task,2 post task,3 done, 4 other",
    )
    req = Column(String, comment="请求参数")
