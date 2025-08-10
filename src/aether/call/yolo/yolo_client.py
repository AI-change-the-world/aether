import json
import os

import cv2

from aether.api.request import AetherRequest
from aether.api.response import with_timing_response
from aether.call.base import BaseClient
from aether.call.config import YOLOConfig
from aether.common.logger import logger
from aether.models.task.task_crud import AetherTaskCRUD
from aether.models.tool.tool_crud import AetherToolCRUD
from aether.utils.object_match import validate


class YoloClient(BaseClient):
    def __init__(self, auto_dispose: bool = False):
        super().__init__(auto_dispose=auto_dispose)

    @with_timing_response
    def __call(self, req: AetherRequest) -> dict:
        """
        执行YOLO推理任务的调用函数

        Args:
            req (AetherRequest): 包含任务请求信息的对象，必须包含task字段且值为"yolo_detection"

        Returns:
            dict: 包含任务执行结果的字典

        Raises:
            AssertionError: 当req.task不等于"yolo_detection"时抛出断言错误
        """
        __task_name__ = "call_yolo_inference"
        # 验证任务类型必须为yolo_detection
        assert req.task == "yolo_detection", "task must be yolo_detection"

        # 构造任务JSON数据
        task_json = {
            "task_type": __task_name__,
            "status": 0,
            "req": req.model_dump_json(),
        }

        try:

            # 创建Aether任务记录
            aether_task = AetherTaskCRUD.create(self.session, task_json)

            if req.tool_id == 0:
                logger.error(f"[{__task_name__}] tool_id is 0")
                raise ValueError("tool_id is 0")

            tool_model = AetherToolCRUD.get_by_id(self.session, req.tool_id)
            if tool_model is None:
                logger.error(f"[{__task_name__}] tool model not found")
                raise ValueError("tool model not found")
            tool_config = json.loads(tool_model.tool_config)
            config = YOLOConfig(**tool_config)
            if req.input is None:
                logger.error(f"[{__task_name__}] input is None")
                raise ValueError("input is None")

            # 导入YOLO检测模块
            from aether.call.yolo.inference import yolo_detect

            img = req.input.data
            # TODO get data from input
            # default local path
            image = cv2.imread(img)
            thres = 0.25
            if req.extra and tool_model.req is not None:
                if not validate(req.extra, tool_model.req):
                    raise Exception(
                        f"[{__task_name__}] Input extra is not valid\nExpected: {tool_model.req}\nGot: {req.extra}"
                    )

            thres = req.extra.get("conf_thres", 0.25)
            model, result = yolo_detect(config.model_path, image, thres)
            if result is None:
                raise ValueError("result is None")

            self.model = model
            result["task_id"] = aether_task.aether_task_id
            result["image"] = img
            if tool_model.resp is not None:
                if not validate(result, tool_model.resp):
                    raise Exception(
                        f"[{__task_name__}] result is not valid\nExpected: {tool_model.resp}\nActual: {result}"
                    )
            return result

        except Exception as e:
            logger.error(f"[{__task_name__}] error: {e}")
            task_json["status"] = 4
            AetherTaskCRUD.update(self.session, aether_task.aether_task_id, task_json)
            raise e

    def call(self, req, **kwargs) -> dict:
        return self.__call(req)
