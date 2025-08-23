import cv2
import numpy as np
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m-pose.pt")  # load an official model

# Predict with the model
results = model(r"D:\github_repo\aether\tests\logs\15.jpg")  # predict on an image

print(len(results))

# # Access the results
# for result in results:
#     xy = result.keypoints.xy  # x and y coordinates
#     xyn = result.keypoints.xyn  # normalized
#     kpts = result.keypoints.data  # x, y, visibility (if available)

#     print(result.keypoints)


def detect_and_crop(image: np.ndarray, min_keypoints=5, conf_thresh=0.5, padding=0.2):
    """
    使用YOLO Pose检测并裁剪单个人员图像
    - image: 输入图像 np.ndarray (BGR格式)
    - min_keypoints: 最少关键点数
    - conf_thresh: 置信度阈值
    - padding: 边界框扩展比例
    返回: 单个人员的裁剪图像 np.ndarray 或 None
    """
    results = model(image)

    print(f"boxes: {len(results)}")

    if len(results) == 0 or len(results[0].keypoints) == 0:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores = results[0].boxes.conf.cpu().numpy()  # 置信度
    kps = results[0].keypoints.data.cpu().numpy()  # 关键点 [n, 17, 3]

    print(f"kps: {kps}")

    annotated_img = image.copy()

    for kp in kps:
        for x, y, _ in kp:
            if x > 0 and y > 0:  # 关键点存在
                cv2.circle(annotated_img, (int(x), int(y)), 3, (0, 255, 0), -1)

    cv2.imwrite(f"xxx.jpg", annotated_img)

    # 只保留置信度高的
    valid_indices = [i for i, s in enumerate(scores) if s >= conf_thresh]
    print(f"{len(valid_indices)} valid boxes")
    if len(valid_indices) != 1:
        return None  # 过滤：没检测到人 或 多个人

    idx = valid_indices[0]
    x1, y1, x2, y2 = boxes[idx]
    keypoints = kps[idx]

    # 过滤关键点数不足的情况
    valid_kps = [kp for kp in keypoints if kp[2] > 0.3]
    if len(valid_kps) < min_keypoints:
        return None

    # 扩展边框
    h, w, _ = image.shape
    box_w, box_h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - padding * box_w))
    y1 = max(0, int(y1 - padding * box_h))
    x2 = min(w, int(x2 + padding * box_w))
    y2 = min(h, int(y2 + padding * box_h))

    # 裁剪
    cropped = image[y1:y2, x1:x2]
    return cropped


if __name__ == "__main__":
    img = r"D:\github_repo\aether\tests\logs\15.jpg"
    image = cv2.imread(img)
    cropped = detect_and_crop(image)
    print(cropped.shape)
    cv2.imwrite("Cropped.png", cropped)
