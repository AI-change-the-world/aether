import gc
import json

import torch

from aether.api.request import AetherRequest, RegisterModelRequest
from aether.call.model import ModelConfig
from aether.common.logger import logger
from aether.db.basic import get_session
from aether.db.task.task_crud import AetherTaskCRUD
from aether.db.tool_model.tool_model_crud import AetherToolModelCRUD
from aether.utils.object_match import is_object_match


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

    def call(self, req: AetherRequest, **kwargs): ...


class Client(__BasicClient):
    """
    模型客户端类，用于初始化和管理模型配置

    Args:
        config (ModelConfig): 模型配置对象，包含模型的相关配置信息
        auto_dispose (bool, optional): 是否自动释放资源，默认为False
    """

    def __init__(self, auto_dispose: bool = False):
        self.auto_dispose = auto_dispose
        self.session = get_session()

    # TODO: implement async call
    def __call_register_model(self, req: AetherRequest[RegisterModelRequest], **kwargs):
        __task_name__ = "register_model"
        assert req.task == "register_model", "task must be register_model"
        if req.model_id != 0:
            logger.warning(f"[{__task_name__}] model_id is not 0, will be ignored")
        logger.info(f"[{__task_name__}] create register model task ...")
        task_json = {
            "task_type": "register_model",
            "status": 0,
            "req": req.model_dump_json(),
        }
        aether_task = AetherTaskCRUD.create(self.session, task_json)
        if aether_task is None:
            logger.error(f"[{__task_name__}] create task failed")
            raise ValueError("create task failed")
        if not is_object_match(req.extra, RegisterModelRequest):
            logger.error(f"[{__task_name__}] extra is not RegisterModelRequest")
            task_json["status"] = 4
            aether_task = AetherTaskCRUD.update(
                self.session, aether_task.aether_task_id, task_json
            )
            raise ValueError("extra is not RegisterModelRequest")
        logger.info(f"[{__task_name__}] create tool model ...")
        model_config = {
            "api_key": req.extra.api_key,
            "base_url": req.extra.base_url,
            "model_name": req.extra.model_name,
            "model_type": req.extra.model_type,
            "model_path": req.extra.model_path,
        }
        aether_tool_json = {
            "tool_model_name": req.extra.model_name,
            "tool_model_config": json.dumps(model_config),
            "req": json.dumps(req.extra.request_definition),
            "resp": json.dumps(req.extra.response_definition),
        }
        aether_tool_model = AetherToolModelCRUD.create(self.session, aether_tool_json)
        if aether_tool_model is None:
            logger.error(f"[{__task_name__}] create tool model failed")
            task_json["status"] = 4
            aether_task = AetherTaskCRUD.update(
                self.session, aether_task.aether_task_id, task_json
            )
            raise ValueError("create tool model failed")
        logger.info(f"[{__task_name__}] update task status to 1 ...")
        task_json["status"] = 1
        AetherTaskCRUD.update(self.session, aether_task.aether_task_id, task_json)

    def __call_openai_model(self, req: AetherRequest, **kwargs):
        __task_name__ = "call_openai_model"
        assert req.task == "chat", "task must be chat"
        if req.model_id == 0:
            logger.error(f"[{__task_name__}] model_id is 0")
            raise ValueError("model_id is 0")
        pass

    def call(self, req, **kwargs):
        if req.task == "register_model":
            return self.__call_register_model(req, **kwargs)
        return super().call(input, **kwargs)

    def re_activate(self):
        return super().re_activate()

    @staticmethod
    def from_config(config: ModelConfig, auto_dispose: bool = False):
        pass
