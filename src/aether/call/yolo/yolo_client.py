import json
import os

import cv2

from aether.api.request import AetherRequest
from aether.api.response import with_timing_response
from aether.call.base import BaseClient
from aether.call.client import ActivatedToolRegistry, register_on_success
from aether.call.config import YOLOConfig
from aether.common.logger import logger
from aether.models.task.task_crud import AetherTaskCRUD
from aether.models.tool.tool_crud import AetherToolCRUD
from aether.utils.object_match import validate


class YoloClient(BaseClient):
    def __init__(self, auto_dispose: bool = False):
        super().__init__(auto_dispose=auto_dispose)

    @register_on_success(
        lambda self, args, kwargs: getattr(self, "_last_tool_id", None)
    )
    @with_timing_response
    def __call(self, req: AetherRequest) -> dict:
        __task_name__ = "call_yolo_inference"

        # 校验任务类型
        assert req.task == "yolo_detection", "task must be yolo_detection"

        aether_task = self._create_task(req, __task_name__)

        try:
            tool_model = AetherToolCRUD.get_by_id(self.session, req.tool_id)
            if not tool_model:
                raise ValueError("tool model not found")

            self._validate_input(req.extra, tool_model.req, __task_name__)

            self.model = self._load_or_get_model(req, tool_model)
            result = self._run_inference(self.model, req)

            self._validate_output(result, tool_model.resp, __task_name__)

            result["task_id"] = aether_task.aether_task_id
            result["image"] = req.input.data

            self.finalize(req)

            return result

        except Exception as e:
            logger.error(f"[{__task_name__}] error: {e}")
            if aether_task:
                AetherTaskCRUD.update(
                    self.session, aether_task.aether_task_id, {"status": 4}
                )
            raise

    # --- 辅助方法 ---

    def _create_task(self, req, task_name):
        task_json = {
            "task_type": task_name,
            "status": 0,
            "req": req.model_dump_json(),
        }
        return AetherTaskCRUD.create(self.session, task_json)

    def _validate_input(self, extra, expected_schema, task_name):
        if extra and expected_schema is not None:
            if not validate(extra, expected_schema):
                raise ValueError(f"[{task_name}] Input extra is not valid")

    def _load_or_get_model(self, req, tool_model):
        from aether.call.yolo.inference import load_model

        model = ActivatedToolRegistry.instance().get(req.tool_id)
        if model:
            return model

        logger.info(f"[call_yolo_inference] tool not found, loading...")
        tool_config = json.loads(tool_model.tool_config)
        config = YOLOConfig(**tool_config)
        return load_model(config.model_path)

    def _run_inference(self, model, req):
        from aether.call.yolo.inference import yolo_detect

        image = cv2.imread(req.input.data)
        _, result = yolo_detect(model, image, req.extra.get("conf_thres", 0.25))
        return result or {}

    def _validate_output(self, result, expected_schema, task_name):
        if expected_schema is not None:
            if not validate(result, expected_schema):
                raise ValueError(f"[{task_name}] result is not valid")

    def call(self, req, **kwargs) -> dict:
        return self.__call(req)
