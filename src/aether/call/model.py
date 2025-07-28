import gc
from enum import Enum
from typing import Optional

import torch
from pydantic import BaseModel, model_validator

from aether.api.request import AetherRequest


class ModelType(str, Enum):
    """
    模型类型枚举类

    定义了系统支持的模型类型常量
    """

    OPENAI = "openai"
    YOLO = "yolo"
    GAN = "gan"
    GROUNDING_DINO = "grounding_dino"  # support huggingface model now


class ModelConfig(BaseModel):
    """
    模型配置类

    用于存储和验证模型配置信息的Pydantic模型

    Attributes:
        base_url (Optional[str]): 模型服务的基础URL地址
        api_key (Optional[str]): 访问模型服务所需的API密钥
        model_name (Optional[str]): 模型名称
        save_path (Optional[str]): 模型文件的本地保存路径
        model_type (ModelType): 模型类型，必须指定
    """

    base_url: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None
    save_path: Optional[str] = None
    model_type: ModelType

    @model_validator
    def check_url_or_path(cls, values):
        """
        验证器函数，确保base_url和save_path至少有一个被指定

        Args:
            values: 包含所有字段值的字典

        Returns:
            values: 验证通过的字段值字典

        Raises:
            ValueError: 当base_url和save_path都为空时抛出异常
        """
        # 获取base_url和save_path的值
        a, b = values.get("base_url"), values.get("save_path")
        # 验证base_url和save_path不能同时为空
        if not a and not b:
            raise ValueError("参数base_url和参数save_path不能同时为空")
        return values


class __BasicClient:

    def __init__(self):
        self.model = None

    def dispose(self):
        if self.model is not None:
            # 假设 model 有个 close 或 cleanup 方法，调用它
            if hasattr(self.model, "close"):
                try:
                    self.model.close()
                except Exception:
                    pass

            # 如果是 PyTorch 模型，释放 GPU 显存
            if hasattr(self.model, "to"):
                try:
                    self.model.cpu()  # 把模型移回 CPU
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 断开引用
            self.model = None

            # 手动触发垃圾回收，释放内存
            gc.collect()

    def re_activate(self):
        pass

    def call(self, input: AetherRequest, **kwargs): ...


class ProvidedModelClient(__BasicClient):
    """
    模型客户端类，用于初始化和管理模型配置

    Args:
        config (ModelConfig): 模型配置对象，包含模型的相关配置信息
        auto_dispose (bool, optional): 是否自动释放资源，默认为False
    """

    def __init__(self, config: ModelConfig, auto_dispose: bool = False):
        self.config = config
        self.auto_dispose = auto_dispose

    def call(self, input, **kwargs):
        return super().call(input, **kwargs)

    def re_activate(self):
        return super().re_activate()

    @staticmethod
    def from_config(config: ModelConfig, auto_dispose: bool = False):
        pass


global_activated_model = {}
