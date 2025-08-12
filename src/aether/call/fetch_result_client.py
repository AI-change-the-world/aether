from aether.api.request import AetherRequest
from aether.api.response import AetherResponse, with_timing_response
from aether.call.base import BaseClient
from aether.common.logger import logger
from aether.models.task.task_crud import AetherTaskCRUD


class FetchResultClient(BaseClient):
    __task_name__ = "fetch_result"

    @with_timing_response
    def __call(self, req: AetherRequest) -> dict:

        try:
            task = self.create_task(req)
            task_json = task.to_dict()
            task_id = int(req.input.data)
            log = AetherTaskCRUD.get_by_id(self.session, task_id)
            if log is None:
                raise Exception(f"Task {task_id} not found")
            return {
                "task_id": task.aether_task_id,
                "output": log.to_dict(),
            }

        except Exception as e:
            logger.error(f"[{self.__task_name__}] Invalid task_id: {e}")
            task_json["status"] = 4
            AetherTaskCRUD.update(self.session, task.aether_task_id, task_json)
            raise e

    def __init__(self, config=None):
        super().__init__(config=config)

    def call(self, req, **kwargs) -> AetherResponse:
        return self.__call(req)
