import json
import uuid
from typing import Optional

from aether.api.request import AetherRequest
from aether.api.response import with_timing_response
from aether.call.base import BaseClient
from aether.call.config import OpenAIChatConfig
from aether.common.logger import logger
from aether.models.task.task_crud import AetherTaskCRUD
from aether.models.tool.tool_crud import AetherToolCRUD
from aether.utils.object_match import validate


class OpenAIClient(BaseClient):
    """
    专门处理 OpenAI 模型的客户端，实现了 chat 任务。
    A client specialized in handling OpenAI models and the chat task.
    """

    __task_name__ = "call_openai_model"

    def __init__(self, config: OpenAIChatConfig):
        super().__init__(config)
        self.config = config
        try:
            from openai import OpenAI

            self.tool = OpenAI(
                api_key=self.config.api_key, base_url=self.config.base_url,
            )
        except ImportError:
            raise ImportError("The 'openai' package is not installed.")

    @with_timing_response
    def __call(self, req: AetherRequest) -> dict:
        """
        调用OpenAI模型处理聊天任务。

        参数:
            req (AetherRequest): 包含任务信息的请求对象，必须是聊天类型任务。

        返回:
            dict: 包含模型输出结果和任务状态的字典，包含以下键：
                - uuid: 唯一标识符
                - output: 模型返回的内容
                - done: 任务是否完成的标志
                - task_id: 任务ID
        """

        assert req.task == "chat", "task must be chat"

        # 创建任务记录
        aether_task = self.create_task(req)

        if aether_task is None:
            logger.error(f"[{self.__task_name__}] create task failed")
            raise ValueError("create task failed")

        task_json = aether_task.to_dict()

        # 检查模型ID是否有效
        if req.tool_id == 0:
            logger.error(f"[{self.__task_name__}] tool_id is 0")
            raise ValueError("tool_id is 0")

        try:
            if self.tool is None:
                logger.error(f"[{self.__task_name__}] model is null")
                raise ValueError("model is null")

            model_name = self.config.model_name

            if self.tool_model.req is not None and req.extra is not None:
                if not validate(req.extra, self.tool_model.req):
                    raise Exception(
                        f"[{self.__task_name__}] extra not match req\nExpected: {self.tool_model.req}\nGot: {req.extra}"
                    )

            # 构造对话历史并调用模型
            history = getattr(req.extra, "history", []) if req.extra else []
            temperature = getattr(req.extra, "temperature", 0.7) if req.extra else 0.7
            history.append({"role": "user", "content": req.input.data})

            response = self.tool.chat.completions.create(
                model=model_name, messages=history, temperature=temperature
            )

            # 更新任务状态为成功
            task_json["status"] = 1
            AetherTaskCRUD.update(self.session, aether_task.aether_task_id, task_json)

            return {
                "uuid": str(uuid.uuid4()),
                "output": response.choices[0].message.content,
                "done": True,
                "task_id": aether_task.aether_task_id,
            }
        except json.JSONDecodeError:
            # 处理模型配置JSON解析错误
            logger.error(f"[{self.__task_name__}] tool model config is not valid json")
            task_json["status"] = 4
            AetherTaskCRUD.update(self.session, aether_task.aether_task_id, task_json)
            raise ValueError("tool model config is not valid json")
        except Exception as e:
            # 处理其他异常情况
            logger.error(f"[{self.__task_name__}] error: {e}")
            task_json["status"] = 4
            AetherTaskCRUD.update(self.session, aether_task.aether_task_id, task_json)
            raise e
        finally:
            if req.meta.auto_dispose:
                self.dispose()

    def call(self, req, **kwargs):
        return self.__call(req)
