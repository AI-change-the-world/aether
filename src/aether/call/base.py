import gc
from abc import ABC
from typing import Optional

import torch

from aether.api.generic import T
from aether.api.request import AetherRequest
from aether.api.response import AetherResponse, with_timing_response
from aether.call.config import BaseModelConfig
from aether.db.basic import get_session


# --- Client Interface and Implementations ---
class BaseClient(ABC):
    """Abstract base class for all Aether clients."""

    def __init__(
        self, config: Optional[BaseModelConfig] = None, auto_dispose: bool = False
    ):
        self.config = config
        self.session = get_session()
        self.model = None
        self.auto_dispose = auto_dispose

    @with_timing_response
    def threaded_task_wrapper(self, task_id: int) -> dict:
        return {"task_id": task_id}

    def dispose(self):
        if self.model is not None and self.auto_dispose:
            if hasattr(self.model, "close"):
                try:
                    self.model.close()
                except Exception:
                    pass
            if hasattr(self.model, "to"):
                try:
                    self.model.cpu()
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            gc.collect()

    def re_activate(self):
        pass

    @staticmethod
    def from_config(
        config: BaseModelConfig, auto_dispose: bool = False
    ) -> "BaseClient":
        pass

    def call(self, req: AetherRequest, **kwargs) -> AetherResponse[T]: ...
