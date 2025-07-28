from typing import Generic, Optional

from pydantic import BaseModel, Field

from aether.api.generic import T


class ResponseMeta(BaseModel):
    """
    响应元数据模型类

    用于存储API响应的元数据信息

    Attributes:
        time_cost_ms (int): 请求处理耗时，单位为毫秒
        task_id (Optional[int]): 任务ID，可选字段
    """

    time_cost_ms: int = Field(..., alias="time_cost_ms")
    task_id: Optional[int] = Field(None, alias="task_id")


class AetherResponse(BaseModel, Generic[T]):
    """
    通用响应模型类

    用于封装API的统一响应格式，支持泛型输出类型

    Attributes:
        success (bool): 请求是否成功
        output (Optional[T]): 响应数据内容，类型为泛型T，可选字段
        meta (ResponseMeta): 响应元数据信息
        error (Optional[str]): 错误信息，当请求失败时提供详细错误描述，可选字段
    """

    success: bool
    output: Optional[T] = None
    meta: ResponseMeta
    error: Optional[str] = None
