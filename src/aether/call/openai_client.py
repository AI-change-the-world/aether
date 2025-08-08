import json
import uuid

from aether.api.request import AetherRequest
from aether.api.response import with_timing_response
from aether.call.base import BaseClient
from aether.call.config import OpenAIChatConfig
from aether.common.logger import logger
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
        __task_name__ = "call_openai_model"
        assert req.task == "chat", "task must be chat"

        # 创建任务记录
        task_json = {
            "task_type": __task_name__,
            "status": 0,
            "req": req.model_dump_json(),
        }
        aether_task = AetherTaskCRUD.create(self.session, task_json)

        # 检查模型ID是否有效
        if req.model_id == 0:
            logger.error(f"[{__task_name__}] model_id is 0")
            raise ValueError("model_id is 0")

        # 如果不自动释放资源，则需要保存模型配置（TODO）
        if not self.auto_dispose:
            # TODO save model config from req
            ...

        try:
            from openai import OpenAI

            # 获取工具模型配置
            tool_model = AetherToolModelCRUD.get_by_id(self.session, req.model_id)
            if tool_model is None:
                logger.error(f"[{__task_name__}] tool model not found")
                raise ValueError("tool model not found")

            tool_model_config = json.loads(tool_model.tool_model_config)
            config = OpenAIChatConfig(**tool_model_config)
            self.model = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
            )
            model_name = config.model_name

            # 构造对话历史并调用模型
            history = getattr(req.extra, "history", []) if req.extra else []
            temperature = getattr(req.extra, "temperature", 0.7) if req.extra else 0.7
            history.append({"role": "user", "content": req.input.data})

            response = self.model.chat.completions.create(
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
            logger.error(f"[{__task_name__}] tool model config is not valid json")
            task_json["status"] = 4
            AetherTaskCRUD.update(self.session, aether_task.aether_task_id, task_json)
            raise ValueError("tool model config is not valid json")
        except Exception as e:
            # 处理其他异常情况
            logger.error(f"[{__task_name__}] error: {e}")
            task_json["status"] = 4
            AetherTaskCRUD.update(self.session, aether_task.aether_task_id, task_json)
            raise e

    def call(self, req, **kwargs):
        return self.__call(req)
