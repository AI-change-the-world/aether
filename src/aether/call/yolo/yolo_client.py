
import cv2

from aether.api.request import AetherRequest
from aether.api.response import with_timing_response
from aether.call.base import BaseClient
from aether.call.config import YOLOConfig
from aether.common.logger import logger
from aether.models.task.task_crud import AetherTaskCRUD
from aether.utils.object_match import validate


class YoloClient(BaseClient):

    __task_name__ = "call_yolo_inference"

    def __init__(self, config: YOLOConfig, auto_dispose: bool = False):
        self.auto_dispose = auto_dispose
        self.config = config
        self.model = None

        try:
            self.model = self._load_or_get_model()
        except Exception as e:
            logger.error(f"[{self.__task_name__}] load model error: {e}")
            raise

    @with_timing_response
    def __call(self, req: AetherRequest) -> dict:
        # 校验任务类型
        assert req.task == "yolo_detection", "task must be yolo_detection"

        aether_task = self._create_task(req, self.__task_name__)

        try:
            if self.tool_model is not None:
                self._validate_input(req.extra, self.tool_model.req, self.__task_name__)

            result = self._run_inference(self.model, req)
            if self.tool_model is not None:
                self._validate_output(result, self.tool_model.resp, self.__task_name__)

            result["task_id"] = aether_task.aether_task_id
            result["image"] = req.input.data

            self.finalize(req)

            return result

        except Exception as e:
            logger.error(f"[{self.__task_name__}] error: {e}")
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

    def _load_or_get_model(
        self,
    ):
        from aether.call.yolo.inference import load_model

        return load_model(self.config.model_path)

    def _run_inference(self, model, req):
        from aether.call.yolo.inference import yolo_detect

        # TODO data source implementation
        image = cv2.imread(req.input.data)
        _, result = yolo_detect(model, image, req.extra.get("conf_thres", 0.25))
        return result or {}

    def _validate_output(self, result, expected_schema, task_name):
        if expected_schema is not None:
            if not validate(result, expected_schema):
                raise ValueError(f"[{task_name}] result is not valid")

    def call(self, req, **kwargs) -> dict:
        return self.__call(req)
