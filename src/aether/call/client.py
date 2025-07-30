import gc

import torch

from aether.api.request import AetherRequest
from aether.call.model import ModelConfig
from aether.common.logger import logger


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


class Client(__BasicClient):
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
