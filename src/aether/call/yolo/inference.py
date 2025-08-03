from typing import Any, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.ops import nms

from aether.call.yolo import __IS_SUPERVISION_INSTALLED__, __IS_ULTRALYTICS_INSTALLED__
from aether.common.logger import logger


class YoloInstance:
    def __init__(self, instance: Any, is_ultralytics: bool):
        self.instance = instance
        self.is_ultralytics = is_ultralytics


def yolo_detect(
    model_path_or_instance: str | YoloInstance, img: Union[np.ndarray, Image.Image, str]
) -> Tuple[Any, dict]:
    """If **img** is str, must be a s3 object"""

    if img is str:
        # TODO download from s3
        pass

    if isinstance(model_path_or_instance, YoloInstance):
        model = model_path_or_instance.instance
        is_ultralytics = model_path_or_instance.is_ultralytics
    else:
        if __IS_ULTRALYTICS_INSTALLED__:
            logger.info("[yolo inference] Using ultralytics")
            from ultralytics import YOLO

            model = YOLO(model_path_or_instance)
            is_ultralytics = True
        else:

            is_cuda = torch.cuda.is_available()
            model = torch.load(
                model_path_or_instance,
                map_location=torch.device("cpu" if not is_cuda else "cuda"),
            )
            net = model.model if hasattr(model, "model") else model
            net.eval()

            transform = T.Compose(
                [
                    T.Resize((640, 640)),  # 注意：YOLOv8 默认是 640x640
                    T.ToTensor(),
                ]
            )
            img_tensor = transform(img).unsqueeze(0)
            if is_cuda:
                img_tensor = img_tensor.to()
                net = net.to()
            with torch.no_grad():
                pred = net(img_tensor)
