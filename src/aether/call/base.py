import gc
import inspect
from abc import ABC
from typing import Optional, Type

import torch

from aether.api.generic import T
from aether.api.request import AetherRequest
from aether.api.response import AetherResponse, with_timing_response
from aether.call.config import BaseToolConfig
from aether.common.logger import logger
from aether.db.basic import get_session
from aether.models.task.task import AetherTask
from aether.models.task.task_crud import AetherTaskCRUD
from aether.models.tool.tool import AetherTool
from aether.models.tool.tool_crud import AetherToolCRUD


# --- Client Interface and Implementations ---
class BaseClient(ABC):
    """Abstract base class for all Aether clients."""

    def __init__(self, config: Optional[BaseToolConfig] = None):
        # 检查调用栈，避免用户直接实例化
        stack = inspect.stack()
        if not any(
            "create_client" in frame.function or "create_client" in frame.filename
            for frame in stack
        ):
            logger.warning(
                f"⚠️ {self.__class__.__name__} 建议通过 ClientManager 创建，以便正确缓存和管理资源。"
            )

        self.config = config
        self.session = get_session()
        self.tool = None
        self.tool_model = getattr(self.config, "tool_model", None)

    def create_task(self, req: AetherRequest) -> Optional[AetherTask]:
        try:
            task_json = {
                "task_type": self.__task_name__,
                "status": 0,
                "req": req.model_dump_json(),
            }
            aether_task = AetherTaskCRUD.create(self.session, task_json)
            return aether_task
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            return None

    def get_tool(self, req: AetherRequest) -> Optional[AetherTool]:
        if req.tool_id is not None and req.tool_id > 0:
            tool = AetherToolCRUD.get_by_id(self.session, req.tool_id)
            return tool
        return None

    @with_timing_response
    def threaded_task_wrapper(self, task_id: int) -> dict:
        return {"task_id": task_id}

    def dispose(self):
        if self.tool is not None:
            if hasattr(self.tool, "close"):
                try:
                    self.tool.close()
                except Exception:
                    pass
            if hasattr(self.tool, "to"):
                try:
                    self.tool.cpu()
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.tool = None
            gc.collect()

    def call(self, req: AetherRequest, **kwargs) -> AetherResponse[T]:
        ...
