import time
from typing import Generic, Optional

from pydantic import BaseModel, Field

from aether.models.generic import T
from aether.models.input import Input


class RequestMeta(BaseModel):
    """
    请求元数据模型类

    用于定义请求的元数据信息，继承自BaseModel

    Attributes:
        task_id (Optional[int]): 任务ID，可选参数，默认为当前时间
        sync (bool): 是否同步执行，必填参数
    """

    task_id: Optional[int] = Field(
        default_factory=lambda: int(time.time()), alias="task_id"
    )
    sync: bool


class AetherRequest(BaseModel, Generic[T]):
    """
    AetherRequest类用于定义请求模型，继承自BaseModel并支持泛型

    Attributes:
        task (str): 任务描述字符串
        model_id (int): 模型ID，使用Field定义别名
        input (Input): 输入数据对象
        meta (RequestMeta): 请求元数据对象
        extra (Optional[T]): 可选的额外数据，类型为泛型T，默认为None
    """

    task: str
    model_id: int = Field(..., alias="model_id")
    input: Input
    meta: RequestMeta
    extra: Optional[T] = None
