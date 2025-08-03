import cv2
import numpy as np
import torch
from torchvision.ops import nms


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


# -------- 1. 加载模型 --------
model_path = "yolo11n.pt"  # 请确保这是一个 YOLOv8 格式的模型
class_names = None
try:
    # 加载模型时，通常需要加载 'model' 键
    model_data = torch.load(model_path, map_location="cpu", weights_only=False)
    print(model_data.keys())
    print(model_data["train_args"])
    print(model_data["train_metrics"])

    net = (
        model_data["model"]
        if isinstance(model_data, dict) and "model" in model_data
        else model_data
    )
    print(type(net))
    if hasattr(net, "names"):
        class_names = net.names
    print(class_names)
    net = net.float().eval()
except Exception as e:
    print(f"加载模型失败，请确保 '{model_path}' 是一个有效的 YOLO 文件。错误: {e}")
    exit()

# -------- 2. 加载原图并 letterbox 处理 --------
img_path = "1.webp"
ori_img = cv2.imread(img_path)
if ori_img is None:
    print(f"无法读取图片: {img_path}")
    exit()

img_input, scale, pad_left, pad_top = letterbox_resize(ori_img, (640, 640))

# 转为模型输入格式 (HWC -> CHW, BGR -> RGB)
img = img_input[:, :, ::-1].transpose(2, 0, 1)
img = np.ascontiguousarray(img)  # 转换为连续内存布局
img = torch.from_numpy(img).float() / 255.0
img_tensor = img.unsqueeze(0)

# -------- 3. 模型推理 --------
with torch.no_grad():
    # 原始输出形状为 [1, 84, 8400]
    preds = net(img_tensor)[0]

# -------- 4. 调整维度并进行后处理 --------
# YOLOv8 输出格式为 [batch, channels, boxes], e.g., [1, 84, 8400]
# 我们需要将其转置为 [batch, boxes, channels], e.g., [1, 8400, 84]
preds = preds.permute(0, 2, 1)

# preds[0] 是因为我们只有一个 batch
detections = non_max_suppression(preds[0], conf_thres=0.45, iou_thres=0.5)

# -------- 5. 坐标映射回原图并可视化 --------
if detections is not None and len(detections):
    # 将坐标映射回原图
    xyxy_boxes = detections[:, :4].cpu().numpy()
    xyxy_boxes = scale_coords(xyxy_boxes, scale, pad_left, pad_top, ori_img.shape)

    scores = detections[:, 4].cpu().numpy()
    classes = detections[:, 5].cpu().numpy().astype(int)

    print(f"检测到 {len(xyxy_boxes)} 个物体。")

    for i in range(len(xyxy_boxes)):
        box = xyxy_boxes[i].astype(int)
        label = (
            f"Class_{classes[i]}" if class_names is None else class_names[classes[i]]
        )
        score = scores[i]

        # 绘制矩形框
        cv2.rectangle(ori_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # 准备标签文本
        label_text = f"{label}: {score:.2f}"

        # 绘制标签背景
        (text_w, text_h), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(
            ori_img,
            (box[0], box[1] - text_h - 5),
            (box[0] + text_w, box[1]),
            (0, 255, 0),
            -1,
        )

        # 绘制标签文本
        cv2.putText(
            ori_img,
            label_text,
            (box[0], box[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    cv2.imwrite("result.jpg", ori_img)
    print("检测完成，结果保存在 result.jpg")
else:
    print("没有检测到任何物体。")
