from typing import Any, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

from aether.call.yolo import __IS_SUPERVISION_INSTALLED__, __IS_ULTRALYTICS_INSTALLED__
from aether.common.logger import logger


def letterbox_resize(image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    将图像按比例缩放并填充到新的形状，保持长宽比不变。
    返回：填充后的图像，缩放比例，以及填充的偏移量 (pad_left, pad_top)。
    """
    h, w = image.shape[:2]
    new_w, new_h = new_shape

    # 计算缩放比例
    scale = min(new_w / w, new_h / h)
    resized_w, resized_h = int(round(w * scale)), int(round(h * scale))

    # 缩放图像
    image_resized = cv2.resize(
        image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
    )

    # 计算填充大小
    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    # 添加填充
    image_padded = cv2.copyMakeBorder(
        image_resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    return image_padded, scale, left, top


def scale_coords(xyxy, scale, pad_left, pad_top, original_shape):
    """
    将缩放和填充后的坐标映射回原始图像尺寸。
    """
    # 减去填充
    xyxy[:, [0, 2]] -= pad_left
    xyxy[:, [1, 3]] -= pad_top
    # 应用缩放
    xyxy /= scale
    # 裁剪坐标以确保它们在原始图像边界内
    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, original_shape[1])  # x1
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, original_shape[0])  # y1
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, original_shape[1])  # x2
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, original_shape[0])  # y2
    return xyxy


def xywh2xyxy(x):
    """
    将 [x_center, y_center, width, height] 格式的边界框转换为 [x1, y1, x2, y2] 格式。
    (x1, y1) 是左上角, (x2, y2) 是右下角。
    """
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.45, iou_thres=0.5):
    """
    对 YOLO 模型输出执行非极大值抑制 (NMS)。

    参数:
    - prediction: 模型的原始输出，形状为 [num_boxes, 4 + num_classes], e.g., [8400, 84]。
    - conf_thres: 置信度阈值。
    - iou_thres: IOU阈值。

    返回:
    - detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    # 1. 过滤掉置信度低的预测
    # 对于 YOLOv8, 类别分数本身就代表了置信度
    # prediction[:, 4:] 获取所有类别的分数, .max(1) 找到每个框的最高类别分数
    max_scores = prediction[:, 4:].max(1)[0]
    mask = max_scores > conf_thres
    prediction = prediction[mask]

    if not prediction.shape[0]:
        return []

    # 2. 将框从 (center_x, center_y, width, height) 格式转换为 (x1, y1, x2, y2) 格式
    boxes_xywh = prediction[:, :4]
    boxes_xyxy = xywh2xyxy(boxes_xywh)

    # 3. 找到每个框的最佳类别和其对应的分数
    max_class_scores, max_class_idxs = prediction[:, 4:].max(1)

    # 4. 准备用于 NMS 的数据 [x1, y1, x2, y2, conf, cls]
    # unsqueeze(1) 用于将维度从 [N] 变为 [N, 1] 以便拼接
    dets = torch.cat(
        [
            boxes_xyxy,
            max_class_scores.unsqueeze(1),
            max_class_idxs.unsqueeze(1).float(),
        ],
        1,
    )

    # 5. 执行 NMS
    keep = nms(dets[:, :4], dets[:, 4], iou_thres)

    return dets[keep]


def infer_with_pt(
    model_path_or_instance: str | Any,
    img: Union[np.ndarray, Image.Image],
    device: str = "cpu",
    conf_thres: float = 0.25,
) -> Tuple[Any, Optional[dict]]:
    """
    使用PyTorch模型对图像进行推理，返回检测到的边界框信息。

    参数:
        model_path_or_instance (str | Any): 模型路径（字符串）或已加载的模型实例
        img (Union[np.ndarray, Image.Image]): 输入图像，支持numpy数组或PIL图像
        device (str, optional): 模型运行设备，默认为"cpu"

    返回:
        Tuple[Any, Optional[dict]]: 第一个元素是模型数据（加载的模型或原始输入），第二个元素是包含检测结果的字典，格式如下：
            {
                "bbox": [
                    {
                        "x": int,         # 边界框左上角x坐标
                        "y": int,         # 边界框左上角y坐标
                        "w": int,         # 边界框宽度
                        "h": int,         # 边界框高度
                        "conf": float,    # 置信度分数
                        "class": int,     # 类别索引
                        "class_name": str # 类别名称（如果模型提供）
                    },
                    ...
                ]
            }
            如果发生错误则两个返回值均为None
    """
    class_names = None
    try:
        # 加载模型：根据输入类型决定是从路径加载还是直接使用模型实例
        if isinstance(model_path_or_instance, str):
            model_data = torch.load(
                model_path_or_instance, map_location=device, weights_only=False
            )
        else:
            model_data = model_path_or_instance
        net = (
            model_data["model"]
            if isinstance(model_data, dict) and "model" in model_data
            else model_data
        )
        # 获取类别名称（如果模型中包含）
        if hasattr(net, "names"):
            class_names = net.names
        net = net.float().eval()

        # 图像预处理：统一图像格式并调整尺寸以适应模型输入要求
        if isinstance(img, Image.Image):
            img = np.array(img)
        img_input, scale, pad_left, pad_top = letterbox_resize(img, (640, 640))

        # 转换为模型输入格式 (HWC -> CHW, BGR -> RGB)
        img = img_input[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)  # 转换为连续内存布局
        img = torch.from_numpy(img).float() / 255.0
        img_tensor = img.unsqueeze(0).to(device)

        # 执行模型推理：通过前向传播获取预测结果
        with torch.no_grad():
            # 原始输出形状为 [1, 84, 8400]
            preds = net(img_tensor)[0]
        preds = preds.permute(0, 2, 1)

        results = {"bbox": []}

        # 应用非极大值抑制(NMS)过滤重复检测框
        detections = non_max_suppression(preds[0], conf_thres=0.45, iou_thres=0.5)

        # 处理检测结果：将边界框映射回原图空间，并提取置信度和类别信息
        if detections is not None and len(detections):
            # 将坐标映射回原图
            xyxy_boxes = detections[:, :4].cpu().numpy()
            xyxy_boxes = scale_coords(xyxy_boxes, scale, pad_left, pad_top, img.shape)

            scores = detections[:, 4].cpu().numpy()
            classes = detections[:, 5].cpu().numpy().astype(int)

            logger.info(f"检测到 {len(xyxy_boxes)} 个物体。")

            for i in range(len(xyxy_boxes)):
                box = xyxy_boxes[i].astype(int)

                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                label = (
                    f"Class_{classes[i]}"
                    if class_names is None
                    else class_names[classes[i]]
                )
                score = scores[i]

                if score > conf_thres:

                    results["bbox"].append(
                        {
                            "x": int(x1),
                            "y": int(y1),
                            "w": int(w),
                            "h": int(h),
                            "conf": score,
                            "class": classes[i],
                            "class_name": label,
                        }
                    )
        return model_data, results

    except Exception as e:
        logger.error(e)
        return None, None


def infer_with_ultralytics(
    model_path_or_instance: str | Any,
    img: Union[np.ndarray, Image.Image, str],
    conf_thres: float = 0.25,
) -> Tuple[Any, Optional[dict]]:
    """
    使用Ultralytics YOLO模型对图像进行推理检测

    参数:
        model_path_or_instance (str | Any): 模型路径字符串或已加载的模型实例
        img (Union[np.ndarray, Image.Image, str]): 输入图像，可以是numpy数组、PIL图像或图像路径字符串

    返回:
        Tuple[Any, Optional[dict]]: 包含模型实例和检测结果的元组。检测结果为字典格式，键为"bbox"，值为边界框信息列表，每个边界框包含以下字段：
            - x (int): 边界框左上角x坐标
            - y (int): 边界框左上角y坐标
            - w (int): 边界框宽度
            - h (int): 边界框高度
            - label (str): 类别名称
            - confidence (float): 置信度
            - class_id (int): 类别ID
            如果推理失败则返回None作为第二个元素
    """
    import supervision as sv
    from ultralytics import YOLO

    # 加载模型：如果传入的是路径则加载模型，否则直接使用传入的模型实例
    if isinstance(model_path_or_instance, str):
        model = YOLO(model_path_or_instance)
    else:
        model = model_path_or_instance

    # 执行模型推理
    res = model([img])
    res0 = res[0]

    # 将Ultralytics结果转换为supervision的Detections格式
    d = sv.Detections.from_ultralytics(res0)

    # 解析检测结果，提取每个边界框的信息
    result = []

    for i in range(len(d.xyxy)):
        x1, y1, x2, y2 = d.xyxy[i]
        w = x2 - x1
        h = y2 - y1
        label = d.data["class_name"][i]
        confidence = float(d.confidence[i])
        class_id = int(d.class_id[i])
        if confidence > conf_thres:
            result.append(
                {
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(w),
                    "h": int(h),
                    "label": label,
                    "confidence": confidence,
                    "class_id": class_id,
                }
            )
    return model, {"bbox": result}


def load_model(model_path_or_instance: str | Any) -> Any:
    """
    加载模型

    参数:
        model_path_or_instance (str | Any): 模型路径字符串或已加载的模型实例

    返回:
        Any: 加载的模型实例
    """
    if isinstance(model_path_or_instance, str):
        if __IS_ULTRALYTICS_INSTALLED__ and __IS_SUPERVISION_INSTALLED__:
            from ultralytics import YOLO

            model = YOLO(model_path_or_instance)
            return model
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_data = torch.load(
                model_path_or_instance, map_location=device, weights_only=False
            )
            return model_data
    return model_path_or_instance


def yolo_detect(
    model_path_or_instance: str | Any,
    img: Union[np.ndarray, Image.Image],
    conf_thres=0.25,
) -> Tuple[Any, Optional[dict]]:
    """
    使用YOLO模型对图像进行目标检测

    参数:
        model_path_or_instance (str | Any): 模型路径字符串或已加载的模型实例
        img (Union[np.ndarray, Image.Image]): 待检测的图像，可以是numpy数组或PIL图像对象

    返回:
        Tuple[Any, Optional[dict]]: 检测结果元组，第一个元素为模型，第二个元素为检测结果
    """
    # 根据CUDA可用性选择运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 根据依赖库的安装情况选择推理方式
    if __IS_ULTRALYTICS_INSTALLED__ and __IS_SUPERVISION_INSTALLED__:
        return infer_with_ultralytics(model_path_or_instance, img, conf_thres)
    else:
        return infer_with_pt(model_path_or_instance, img, device, conf_thres)
