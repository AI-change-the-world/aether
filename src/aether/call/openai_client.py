import json
import uuid

from aether.api.request import AetherRequest
from aether.api.response import with_timing_response
from aether.call.base import BaseClient
from aether.call.model import ModelConfig
from aether.common import logger
from aether.db.basic import get_session
from aether.models.task.task_crud import AetherTaskCRUD
from aether.models.tool_model.tool_model_crud import AetherToolModelCRUD


class OpenAIClient(BaseClient):
    """
    专门处理 OpenAI 模型的客户端，实现了 chat 任务。
    A client specialized in handling OpenAI models and the chat task.
    """

    def __init__(self, auto_dispose: bool = False):
        super().__init__(auto_dispose=auto_dispose)

    @with_timing_response
    def __call(self, req: AetherRequest) -> dict:
        __task_name__ = "call_openai_model"
        assert req.task == "chat", "task must be chat"

        task_json = {
            "task_type": __task_name__,
            "status": 0,
            "req": req.model_dump_json(),
        }
        aether_task = AetherTaskCRUD.create(self.session, task_json)

        if req.model_id == 0:
            logger.error(f"[{__task_name__}] model_id is 0")
            raise ValueError("model_id is 0")

        if not self.auto_dispose:
            # TODO compose model config from req
            ...

        try:
            from openai import OpenAI

            tool_model = AetherToolModelCRUD.get_by_id(self.session, req.model_id)
            if tool_model is None:
                logger.error(f"[{__task_name__}] tool model not found")
                raise ValueError("tool model not found")

            tool_model_config = json.loads(tool_model.tool_model_config)
            self.model = OpenAI(
                api_key=tool_model_config["api_key"],
                base_url=tool_model_config["base_url"],
            )
            model_name = tool_model.tool_model_name
            history = getattr(req.extra, "history", []) if req.extra else []
            temperature = getattr(req.extra, "temperature", 0.7) if req.extra else 0.7
            history.append({"role": "user", "content": req.input.data})

            response = self.model.chat.completions.create(
                model=model_name, messages=history, temperature=temperature
            )
            task_json["status"] = 1
            AetherTaskCRUD.update(self.session, aether_task.aether_task_id, task_json)
            return {
                "uuid": str(uuid.uuid4()),
                "output": response.choices[0].message.content,
                "done": True,
                "task_id": aether_task.aether_task_id,
            }
        except json.JSONDecodeError:
            logger.error(f"[{__task_name__}] tool model config is not valid json")
            task_json["status"] = 4
            AetherTaskCRUD.update(self.session, aether_task.aether_task_id, task_json)
            raise ValueError("tool model config is not valid json")
        except Exception as e:
            logger.error(f"[{__task_name__}] error: {e}")
            task_json["status"] = 4
            AetherTaskCRUD.update(self.session, aether_task.aether_task_id, task_json)
            raise e

    def call(self, req, **kwargs):
        return self.__call(req)
