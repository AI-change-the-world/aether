import json

from aether.api.request import AetherRequest, Execution, RegisterModelRequest
from aether.api.response import with_timing_response
from aether.call.base import BaseClient
from aether.call.threaded_call import GLOBAL_THREAD_TASK_MANAGER
from aether.common.logger import logger
from aether.models.task.task_crud import AetherTaskCRUD
from aether.models.tool.tool_crud import AetherToolCRUD
from aether.utils.object_match import is_object_match


class RegisterModelClient(BaseClient):
    """
    专门处理模型注册任务的客户端。
    A client specialized in handling model registration tasks.
    """

    __task_name__ = "register_model"

    def __init__(self, tool_model=None):
        super().__init__()
        self.tool_model = tool_model

    @with_timing_response
    def __call(self, req: AetherRequest[RegisterModelRequest], **kwargs) -> dict:
        """
        处理模型注册请求，创建任务并保存模型配置信息。

        参数:
            req (AetherRequest[RegisterModelRequest]): 包含模型注册请求数据的封装对象。

        返回:
            dict: 包含创建的任务ID的字典，格式为 {"task_id": int}。

        异常:
            ValueError: 当 req.extra 不是 RegisterModelRequest 类型或创建工具模型失败时抛出。
        """

        # 构造任务数据并创建任务记录
        task_json = kwargs.get("task_json", {})
        task_id = kwargs.get("task_id", None)
        if task_id is None:
            logger.error(f"[{self.__task_name__}] task_id is None")
            raise ValueError("task_id is None")

        # 校验 extra 字段是否符合 RegisterModelRequest 结构
        if not is_object_match(req.extra, RegisterModelRequest):
            logger.error(f"[{self.__task_name__}] extra is not RegisterModelRequest")
            task_json["status"] = 4
            AetherTaskCRUD.update(self.session, task_id, task_json)
            raise ValueError("extra is not RegisterModelRequest")

        logger.info(f"[{self.__task_name__}] create tool model ...")
        # 构造模型配置和工具模型数据
        model_config = {
            "api_key": req.extra.api_key,
            "base_url": req.extra.base_url,
            "model_name": req.extra.model_name,
            "tool_type": req.extra.tool_type,
            "model_path": req.extra.model_path,
        }
        aether_tool_json = {
            "tool_name": req.extra.model_name,
            "tool_config": json.dumps(model_config),
            "req": json.dumps(req.extra.request_definition),
            "resp": json.dumps(req.extra.response_definition),
        }
        # 创建工具模型记录
        aether_tool_model = AetherToolCRUD.create(self.session, aether_tool_json)

        # 检查工具模型是否创建成功
        if aether_tool_model is None:
            logger.error(f"[{self.__task_name__}] create tool model failed")
            task_json["status"] = 4
            AetherTaskCRUD.update(self.session, task_id, task_json)
            raise ValueError("create tool model failed")

        # 更新任务状态为完成（status=1）
        logger.info(f"[{self.__task_name__}] update task status to 1 ...")
        task_json["status"] = 1
        AetherTaskCRUD.update(self.session, task_id, task_json)
        return {"task_id": task_id}

    def call(self, req, **kwargs):

        assert req.task == "register_model", "task must be register_model"
        if req.tool_id != 0:
            logger.warning(f"[{self.__task_name__}] tool_id is not 0, will be ignored")
        logger.info(f"[{self.__task_name__}] create register model task ...")
        aether_task = self.create_task(req)
        task_json = aether_task.to_dict()
        if req.meta.execution == Execution.SYNC:
            return self.__call(
                req, task_id=aether_task.aether_task_id, task_json=task_json,
            )
        elif req.meta.execution == Execution.ASYNC:
            GLOBAL_THREAD_TASK_MANAGER.submit(
                self.__call,
                req,
                task_id=aether_task.aether_task_id,
                task_json=task_json,
            )
            return self.threaded_task_wrapper(aether_task.aether_task_id)
        else:
            logger.info(
                f"[{self.__task_name__}] this client only support sync or async execution, using sync instead"
            )
            return self.__call(req)
