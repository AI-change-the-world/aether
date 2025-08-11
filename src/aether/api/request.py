from enum import Enum
from typing import Generic, Optional

from pydantic import BaseModel, Field, field_validator

from aether.api.generic import T
from aether.api.input import Input

__TASKS__ = [
    # for model train and predict
    "train",
    "predict",
    "yolo_detection",
    # for llm
    "chat",
    # for data augment
    "augment",
    # finetune
    "finetune_lora",
    # gpu status
    "gpu_status",
    # register model
    "register_model",
    # fetch result ; if Execution.ASYNC, use {"task_id": xxx} to fetch result
    "fetch_result",
    # get model inormation by {"tool_id": xxx}
    "fetch_model_info",
]


class Execution(str, Enum):
    SYNC = "sync"
    STREAM = "stream"
    ASYNC = "async"


class RequestMeta(BaseModel):
    """
    请求元数据模型类

    用于定义请求的元数据信息，继承自BaseModel

    Attributes:
        execution (str): 执行方式， 包括 sync, stream, async
    """

    execution: Execution = Field(default=Execution.SYNC, alias="execution")
    auto_dispose: bool = Field(default=True, alias="auto_dispose")


class RegisterModelRequest(BaseModel):
    """
    模型注册请求数据模型类

    用于定义模型注册接口的请求参数结构，继承自BaseModel基类

    Attributes:
        model_name (str): 模型名称，用于唯一标识一个模型
        tool_type (str): 模型类型，表示模型的分类或用途
        model_path (Optional[str]): 模型文件路径，可选参数，后续将支持s3路径，目前仅支持本地路径
        base_url (Optional[str]): 模型服务的基础URL，可选参数
        api_key (Optional[str]): 访问模型服务所需的API密钥，可选参数
        request_definition (dict): 模型请求参数定义，描述调用模型时需要的参数结构
                                 : TODO 待完善, yolo example : {}
        response_definition (dict): 模型响应结果定义，描述模型返回结果的结构格式
                                 : TODO 待完善, yolo example :
                                    {
                                                "image": {
                                                    "type": "string",
                                                    "default": "*.png"
                                                },
                                                "bbox": {
                                                    "type": "list",
                                                    "default": "...",
                                                    "items": {
                                                    "type": "object",
                                                    "fields": {
                                                        "x": {
                                                        "type": "int",
                                                        "default": "..."
                                                        },
                                                        "y": {
                                                        "type": "int",
                                                        "default": "..."
                                                        },
                                                        "w": {
                                                        "type": "int",
                                                        "default": "..."
                                                        },
                                                        "h": {
                                                        "type": "int",
                                                        "default": "..."
                                                        },
                                                        "label": {
                                                        "type": "string",
                                                        "default": "..."
                                                        }
                                                    }
                                                    }
                                                }
                                                }
    """

    model_name: str
    tool_type: str
    # 后续支持s3路径，暂时只支持本地路径
    model_path: Optional[str]
    base_url: Optional[str]
    api_key: Optional[str]
    request_definition: dict
    response_definition: dict


class AetherRequest(BaseModel, Generic[T]):
    """
    AetherRequest类用于定义请求模型，继承自BaseModel并支持泛型

    Attributes:
        task (str): 任务描述字符串
        tool_id (int): 模型ID，使用Field定义别名
        input (Input): 输入数据对象, 默认为空
        meta (RequestMeta): 请求元数据对象
        extra (Optional[T]): 可选的额外数据，类型为泛型T，默认为None
    """

    task: str
    tool_id: int = Field(default=0, alias="tool_id")
    input: Optional[Input] = Field(default=None, alias="input")
    meta: RequestMeta = Field(default=RequestMeta(), alias="meta")
    extra: Optional[T] = None

    @field_validator("task", mode="before")
    def task_validate(cls, v):
        v = v.lower()
        assert v in __TASKS__, f"Invalid task: {v}"
        return v
