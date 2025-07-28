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
    # json config, such as {"base_url": "http://xxx", "api_key": "sk-1234567890","model_name": "gpt-3.5-turbo"}
    tool_model_config = Column(String, comment="tool model config")
    # tool model request definition, corresponding to api.request.AetherRequest.extra
    # if req == "", then extra is plain text
    req = Column(String, nullable=False, comment="请求参数定义")
    # tool model response definition, corresponding to api.response.AetherResponse.output
    # if resp == "", then output is plain text
    resp = Column(String, nullable=False, comment="响应参数定义")
