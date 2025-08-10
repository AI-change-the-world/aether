import os
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def save_grounding_dino_results_cv2(
    image: Union[str, np.ndarray],
    detections: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    box_color: tuple = (0, 0, 255),  # BGR, 默认红色
    box_thickness: int = 2,
    font_scale: float = 0.6,
    font_thickness: int = 1,
) -> np.ndarray:
    """
    使用 OpenCV 在原图上绘制 Grounding DINO 的检测结果并保存/返回图像。

    Args:
        image: 图像路径（str）或 numpy.ndarray（BGR）.
        detections: Grounding DINO 输出的 list，格式示例：
            [{'scores': tensor([..]), 'boxes': tensor([[x1,y1,x2,y2]]), 'text_labels': [...], 'labels': [...]}, ...]
        output_path: 如果提供则保存到该路径。
        box_color: 方框颜色 (B, G, R)。
        box_thickness: 方框线宽。
        font_scale: 文本缩放（影响文字大小）。
        font_thickness: 文本线宽。

    Returns:
        numpy.ndarray: 绘制后的 BGR 图像（dtype=uint8）。
    """
    # --- load image ---
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"image not found: {image}")
    elif isinstance(image, np.ndarray):
        img = image.copy()
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
    else:
        raise TypeError("image must be a file path or a numpy.ndarray")

    H, W = img.shape[:2]

    # --- helper to convert tensors/arrays ---
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    # iterate detections
    for det in detections:
        boxes = det.get("boxes")
        if boxes is None:
            continue
        boxes = to_numpy(boxes)  # shape (N,4)
        scores = det.get("scores", None)
        if scores is not None:
            scores = to_numpy(scores)
        else:
            scores = np.ones((boxes.shape[0],), dtype=float)

        # prefer textual labels: text_labels > labels > class_name
        labels = (
            det.get("text_labels") or det.get("labels") or det.get("class_name") or []
        )
        # if labels is numpy array, convert to list of str
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()
        # ensure labels length
        if isinstance(labels, (list, tuple)):
            labels_list = [str(x) for x in labels]
        else:
            labels_list = []

        for i, box in enumerate(boxes):
            # box might be tensor/list of floats
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

            # clamp to image bounds and convert to int
            x1_i = int(round(max(0.0, min(x1, W - 1))))
            y1_i = int(round(max(0.0, min(y1, H - 1))))
            x2_i = int(round(max(0.0, min(x2, W - 1))))
            y2_i = int(round(max(0.0, min(y2, H - 1))))

            # draw rectangle
            cv2.rectangle(
                img, (x1_i, y1_i), (x2_i, y2_i), box_color, thickness=box_thickness
            )

            # label + score
            label = labels_list[i] if i < len(labels_list) else ""
            score = float(scores[i]) if i < len(scores) else None
            text = f"{label} {score:.2f}" if score is not None else label

            # compute text size and background rectangle
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            # default place above box; if not enough space, place inside box top
            text_x = x1_i
            text_y_top = y1_i - 5
            if text_y_top - text_h - baseline < 0:
                # put inside box (below top)
                bg_tl = (text_x, y1_i)
                bg_br = (text_x + text_w, y1_i + text_h + baseline)
                text_org = (text_x, y1_i + text_h)
            else:
                # put above box
                bg_tl = (text_x, text_y_top - text_h - baseline)
                bg_br = (text_x + text_w, text_y_top + baseline)
                text_org = (text_x, text_y_top)

            # draw filled rectangle and put text (white text on colored bg)
            cv2.rectangle(img, bg_tl, bg_br, box_color, thickness=-1)
            cv2.putText(
                img,
                text,
                text_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )

    # save if requested
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"[save_grounding_dino_results_cv2] saved -> {output_path}")

    return img


tool_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(tool_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(tool_id).to(device)

image_path = "1.webp"
image = Image.open(image_path)
# Check for cats and remote controls
# VERY important: text queries need to be lowercased + end with a dot
text = "a person."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]],
)

print(results)

save_grounding_dino_results_cv2(image_path, results, "result.jpg")


def grounding_dino_to_standard(
    detections: List[Dict[str, Any]], label_to_id: Dict[str, int] = None
) -> List[Dict[str, Any]]:
    """
    将 Grounding DINO 输出转换为标准检测格式:
    {
        "x": float,       # 左上角 x
        "y": float,       # 左上角 y
        "w": float,       # 宽度
        "h": float,       # 高度
        "label": str,     # 标签名
        "confidence": float,
        "class_id": int
    }

    Args:
        detections: Grounding DINO 原始输出列表
        label_to_id: 标签到 class_id 的映射，不提供则自动从 0 开始编号

    Returns:
        list[dict]: 标准化检测结果
    """

    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)

    results = []
    label_id_map = label_to_id or {}
    next_id = max(label_id_map.values(), default=-1) + 1

    for det in detections:
        boxes = to_numpy(det.get("boxes", []))
        scores = to_numpy(det.get("scores", []))
        labels = det.get("text_labels") or det.get("labels") or []

        if isinstance(labels, np.ndarray):
            labels = labels.tolist()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(float, box)
            w = x2 - x1
            h = y2 - y1
            label = str(labels[i]) if i < len(labels) else "unknown"
            conf = float(scores[i]) if i < len(scores) else 1.0

            # 分配 class_id
            if label not in label_id_map:
                label_id_map[label] = next_id
                next_id += 1

            results.append(
                {
                    "x": x1,
                    "y": y1,
                    "w": w,
                    "h": h,
                    "label": label,
                    "confidence": conf,
                    "class_id": label_id_map[label],
                }
            )

    return results


print(grounding_dino_to_standard(results))
