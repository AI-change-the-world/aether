from aether.api.generic import T
from aether.api.request import AetherRequest
from aether.api.response import AetherResponse, with_timing_response
from aether.call.base import BaseClient
from aether.common.logger import logger
from aether.models.task.task_crud import AetherTaskCRUD
from aether.models.tool.tool import AetherTool
from aether.utils.object_match import validate


class GetToolsClient(BaseClient):
    __task_name__ = "get_tools"

    def __init__(self, config=None, tool_model=None):
        super().__init__(config)
        self.tool_model = tool_model

    @with_timing_response
    def __call(self, req: AetherRequest, **kwargs) -> dict:
        aether_task = self.create_task(req)

        if aether_task is None:
            logger.error(f"[{self.__task_name__}] create task failed")
            raise ValueError("create task failed")

        task_json = aether_task.to_dict()
        like = None
        if self.tool_model.req is not None and req.extra is not None:
            if not validate(req.extra, self.tool_model.req):
                raise Exception(
                    f"[{self.__task_name__}] extra not match req\nExpected: {self.tool_model.req}\nGot: {req.extra}"
                )
            like = req.extra.get("like", None)

        tools = (
            self.session.query(AetherTool)
            .filter(AetherTool.tool_name.like(f"%{like}%") if like else True)
            .all()
        )

        task_json["status"] = 1
        AetherTaskCRUD.update(self.session, aether_task.aether_task_id, task_json)
        return {
            "output": [tool.to_dict() for tool in tools],
            "task_id": aether_task.aether_task_id,
        }

    def call(self, req: AetherRequest, **kwargs) -> AetherResponse[T]:
        return self.__call(req)
