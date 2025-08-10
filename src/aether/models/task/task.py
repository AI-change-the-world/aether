import time
from datetime import datetime

from sqlalchemy import Column, Integer, SmallInteger, String, event, inspect

from aether.db.basic import Base


class AetherTask(Base):
    __tablename__ = "aether_task"
    __table_args__ = {"comment": "task table"}

    aether_task_id = Column(
        Integer, primary_key=True, autoincrement=True, comment="aether task id"
    )
    # req_task_id = Column(Integer, nullable=True, comment="request task id")
    task_type = Column(String, default="", nullable=True, comment="must in __TASKS__")
    created_at = Column(
        Integer, default=lambda: int(time.time()), comment="创建时间(秒级时间戳)"
    )
    updated_at = Column(
        Integer,
        default=lambda: int(time.time()),
        onupdate=lambda: int(time.time()),
        comment="更新时间(秒级时间戳)",
    )
    is_deleted = Column(SmallInteger, default=0, comment="逻辑删除标记")
    status = Column(
        SmallInteger,
        default=0,
        comment="任务状态, 0 pre task, 1 on task,2 post task,3 done, 4 other",
    )
    req = Column(String, comment="请求参数")

    def to_dict(self):
        """将 ORM 实例转为 JSON 可用的 dict"""
        result = {}
        for c in inspect(self).mapper.column_attrs:
            value = getattr(self, c.key)
            if isinstance(value, datetime):
                value = value.isoformat()  # 转成字符串
            result[c.key] = value
        return result


# 事件钩子：在 SQLite 下手动更新 updated_at
@event.listens_for(AetherTask, "before_update", propagate=True)
def update_timestamp_before_update(mapper, connection, target):
    target.updated_at = int(time.time())
