from sqlalchemy import TIMESTAMP, Column, Integer, SmallInteger, String, func

from aether.db import Base


class AetherToolModel(Base):
    __tablename__ = "aether_tool_model"
    __table_args__ = {"comment": "task table"}

    aether_tool_model_id = Column(
        Integer, primary_key=True, autoincrement=True, comment="aether tool model id"
    )
    created_at = Column(TIMESTAMP, server_default=func.now(), comment="创建时间")
    updated_at = Column(
        TIMESTAMP, server_default=func.now(), onupdate=func.now(), comment="更新时间"
    )
    is_deleted = Column(SmallInteger, default=0, comment="逻辑删除标记")
    tool_model_name = Column(String, comment="tool model name")
    # json config
    tool_model_config = Column(String, comment="tool model config")
    # tool model request json, corresponding to models.request.AetherRequest.extra
    req = Column(String, comment="请求参数")
    # tool model response json, corresponding to models.response.AetherResponse.output
    resp = Column(String, comment="响应参数")
