import time
from functools import wraps
import traceback
from typing import Callable, Generic, Optional

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


def with_timing_response(func: Callable[..., T]) -> Callable[..., AetherResponse[T]]:
    @wraps(func)
    def wrapper(*args, **kwargs) -> AetherResponse[T]:
        start_time = time.time()
        try:
            output = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            traceback.print_exc()
            output = None
            success = False
            error = str(e)
        end_time = time.time()

        time_cost_ms = int((end_time - start_time) * 1000)

        # 尝试获取 task_id（从返回值或 kwargs 中提取）
        task_id = None
        if success and isinstance(output, dict) and "task_id" in output:
            task_id = output["task_id"]
            try:
                output.pop("task_id")
            except:
                pass
        elif "aether_task" in kwargs and hasattr(
            kwargs["aether_task"], "aether_task_id"
        ):
            task_id = kwargs["aether_task"].aether_task_id

        meta = ResponseMeta(time_cost_ms=time_cost_ms, task_id=task_id)

        return AetherResponse[T](
            success=success, output=output if success else None, meta=meta, error=error
        )

    return wrapper
