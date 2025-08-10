import time

from sqlalchemy import Column, Integer, SmallInteger, String, event

from aether.db.basic import Base


class AetherTool(Base):
    __tablename__ = "aether_tool"
    __table_args__ = {"comment": "tool table"}

    aether_tool_id = Column(
        Integer, primary_key=True, autoincrement=True, comment="aether tool id"
    )
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
    tool_name = Column(String, comment="tool name")
    # json config, such as {"base_url": "http://xxx", "api_key": "sk-1234567890","model_name": "gpt-3.5-turbo"}
    tool_config = Column(String, comment="tool config")
    # tool model request definition, corresponding to api.request.AetherRequest.extra
    # if req == "", then extra is plain text
    req = Column(String, nullable=False, comment="请求参数定义")
    # tool model response definition, corresponding to api.response.AetherResponse.output
    # if resp == "", then output is plain text
    resp = Column(String, nullable=False, comment="响应参数定义")


# 事件钩子：在 SQLite 下手动更新 updated_at
@event.listens_for(AetherTool, "before_update", propagate=True)
def update_timestamp_before_update(mapper, connection, target):
    target.updated_at = int(time.time())
