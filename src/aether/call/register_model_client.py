import json

from aether.api.request import AetherRequest, RegisterModelRequest
from aether.api.response import with_timing_response
from aether.call.base import BaseClient
from aether.call.model import ModelConfig
from aether.common.logger import logger
from aether.models.task.task_crud import AetherTaskCRUD
from aether.models.tool_model.tool_model_crud import AetherToolModelCRUD
from aether.utils.object_match import is_object_match


class RegisterModelClient(BaseClient):
    """
    专门处理模型注册任务的客户端。
    A client specialized in handling model registration tasks.
    """

    def __init__(self):
        super().__init__(auto_dispose=True)

    @with_timing_response
    def __call(self, req: AetherRequest[RegisterModelRequest]) -> dict:
        __task_name__ = "register_model"
        assert req.task == "register_model", "task must be register_model"
        if req.model_id != 0:
            logger.warning(f"[{__task_name__}] model_id is not 0, will be ignored")
        logger.info(f"[{__task_name__}] create register model task ...")

        task_json = {
            "task_type": __task_name__,
            "status": 0,
            "req": req.model_dump_json(),
        }
        aether_task = AetherTaskCRUD.create(self.session, task_json)

        if not is_object_match(req.extra, RegisterModelRequest):
            logger.error(f"[{__task_name__}] extra is not RegisterModelRequest")
            task_json["status"] = 4
            AetherTaskCRUD.update(self.session, aether_task.aether_task_id, task_json)
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
            AetherTaskCRUD.update(self.session, aether_task.aether_task_id, task_json)
            raise ValueError("create tool model failed")

        logger.info(f"[{__task_name__}] update task status to 1 ...")
        task_json["status"] = 1
        AetherTaskCRUD.update(self.session, aether_task.aether_task_id, task_json)
        return {"task_id": aether_task.aether_task_id}

    def call(self, req, **kwargs):
        return self.__call(req)
