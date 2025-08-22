import base64
import pickle
import random
import traceback

import cv2
import numpy as np
import supervision as sv
from full_track_script import (
    GLOBAL_INFO,
    ClassTrackerObject,
    crop,
    executor,
    frame_rate,
)
from openai import OpenAI
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

video = cv2.VideoCapture("test.mp4")

frame_id = 0

prompt = """
你是一个专业的视频图像分析专家。你将收到多张图片（大概率为同一人，不同尺寸/角度/光照）。请综合所有图片，只输出一个最终的人物外观描述，严格遵循以下规则与格式。

【任务规则】
1) 只分析人物外观，不推测身份或情绪。
2) 若出现多人：默认选择“在人像框中面积最大且更居中”的那个人。若无法稳定定位到同一人，所有字段填“未知”。
3) 多图融合：先分别在心中评估每张图的属性，再进行融合。
   - 若某属性在部分图片可见、其他图片不可见：以“可见”的结论为准。
   - 若不同图片结论冲突：优先采用“多数一致”的值；若仍冲突，则以“更清晰/更大/遮挡更少”的图片为准（由你判断）。
4) 可见性与外层优先：
   - 仅根据可见区域判断；看不清/被遮挡则该字段填“未知”。
   - 服装描述优先“最外层、最主要、覆盖面积最大”的衣物。
5) 颜色与类型表述：
   - 颜色优先使用常见色词：黑色、白色、灰色、蓝色、红色、绿色、黄色、棕色、粉色。难以判断时填“其他”或“未知”。
   - 类型示例（不限于）：短袖、长袖、T恤、衬衫、毛衣、夹克、外套、连帽衫、背心、连衣裙；裤装如牛仔裤、长裤、短裤、运动裤；鞋类如运动鞋、皮鞋、凉鞋、靴子。
6) 配饰仅列可见者；多项用“、”分隔；没有则写“无”。
7) 特殊动作或状态仅在明显时填写（如：抽烟、打电话、背包、推车、跑步、骑车、拿伞、抱娃、拉箱子等）；不明显则写“无”。
8) 无法判断的一律写“未知”。不要输出除下列字段以外的任何文字、解释或标点。

【输出格式（严格按顺序与字段名；每行一个字段；不得省略字段）】
性别：男 / 女 / 未知
发型：长发 / 短发 / 光头 / 其他 / 未知
发色：黑色 / 棕色 / 黄色 / 白色 / 灰色 / 其他 / 未知
上身：<颜色><类型> 或 未知
下身：<颜色><类型> 或 未知
鞋子：<颜色><类型> 或 未知
配饰：<按“颜色+类型”列举；无则写“无”>
特殊动作或状态：<列举；无则写“无”>

"""


prompt2 = """
你是一个专业的视频图像分析专家。你将收到多张工地场景下的图片（大概率为同一人，不同尺寸/角度/光照）。请综合所有图片，只输出一个最终的人物外观描述，严格遵循以下规则与格式。

【任务规则】
1) 只分析人物外观，不推测身份或情绪。
2) 若出现多人：默认选择“在人像框中面积最大且更居中”的那个人。若无法稳定定位到同一人，所有字段填“未知”。
3) 多图融合：先分别在心中评估每张图的属性，再进行融合。  
   - 若某属性在部分图片可见、其他图片不可见：以“可见”的结论为准。  
   - 若不同图片结论冲突：优先采用“多数一致”的值；若仍冲突，则以“更清晰/更大/遮挡更少”的图片为准（由你判断）。  
4) 可见性与外层优先：  
   - 仅根据可见区域判断；看不清/被遮挡则该字段填“未知”。  
   - 服装描述优先“最外层、最主要、覆盖面积最大”的衣物。  
5) 颜色与类型表述：  
   - 颜色优先使用常见色词：黑色、白色、灰色、蓝色、红色、绿色、黄色、棕色、粉色。难以判断时填“其他”或“未知”。  
   - 类型示例（不限于）：短袖、长袖、T恤、衬衫、毛衣、夹克、外套、连帽衫、背心、连衣裙；裤装如牛仔裤、长裤、短裤、运动裤；鞋类如运动鞋、皮鞋、凉鞋、靴子。  
6) 工地专属物品与配饰：  
   - 优先识别安全帽（颜色+安全帽）、反光背心（颜色+反光背心）、工地手套、工鞋等。  
   - 其他配饰按“颜色+类型”列举；没有则写“无”。  
7) 特殊动作或状态仅在明显时填写（如：搬运物料、戴手套、戴口罩、打电话、抽烟、推车、骑车、拿伞、抱娃、拉箱子等）；不明显则写“无”。  
8) 无法判断的一律写“未知”。不要输出除下列字段以外的任何文字、解释或标点。  

【输出格式（严格按顺序与字段名；每行一个字段；不得省略字段）】
性别：男 / 女 / 未知  
发型：长发 / 短发 / 光头 / 其他 / 未知  
发色：黑色 / 棕色 / 黄色 / 白色 / 灰色 / 其他 / 未知  
上身：<颜色><类型> 或 未知  
下身：<颜色><类型> 或 未知  
鞋子：<颜色><类型> 或 未知  
配饰：<按“颜色+类型”列举；无则写“无”>  
特殊动作或状态：<列举；无则写“无”>  

"""


def numpy_to_base64(frame: np.ndarray) -> str:
    """
    将 NumPy 数组转换为 base64 编码的字符串。
    """
    _, buffer = cv2.imencode(".png", frame)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"


vl_client = OpenAI(
    api_key="sk-x",
    base_url="http://localhost:9997/v1",
)

while True:
    success, frame = video.read()
    if not success:
        print("Video processing complete.")
        break

    frame_id += 1

    # 模型推理
    results = model(frame)[0]

    # 转换为 supervision 检测对象
    detections = sv.Detections.from_ultralytics(results)

    person_class_id = 0
    mask = detections.class_id == person_class_id
    detections = detections[mask]

    # 追踪更新
    detections = tracker.update_with_detections(detections)

    if len(detections) == 0:
        continue

    for bbox, tracker_id in zip(detections.xyxy, detections.tracker_id):
        if tracker_id not in GLOBAL_INFO:
            # 新对象
            obj = ClassTrackerObject(
                tracker_id, start_frame=frame_id, bounding_box=bbox
            )
            GLOBAL_INFO[tracker_id] = obj
            # 更新裁剪图像
            # obj.update_image(crop(frame, bbox, id=tracker_id, save_image=False))
            obj.image = crop(frame, bbox, id=tracker_id, save_image=False)
            obj.image_base64 = numpy_to_base64(obj.image)

            # 异步更新 feature
            # if not obj.is_updating:
            #     obj.is_updating = True
            #     executor.submit(async_update_feature, obj, frame)
        else:
            # 已存在对象，更新最后一次 bbox
            GLOBAL_INFO[tracker_id].update_bounding_box(bbox)
            GLOBAL_INFO[tracker_id].update_end_frame(frame_id)
            if frame_id % frame_rate == 0:
                # 每隔 frame_rate 帧更新一次图像
                executor.submit(GLOBAL_INFO[tracker_id].update_image, crop(frame, bbox))

valid_items = [
    obj for obj in GLOBAL_INFO.values() if obj.end_frame - obj.start_frame >= 24
]


def get_features(obj: ClassTrackerObject):
    if obj.image_base64 is None:
        return
    print(f"Processing object ID: {obj.object_id}  cache size: {len(obj.cache_images)}")
    try:
        if len(obj.cache_images) > 1:
            selected_images = random.sample(
                obj.cache_images, min(3, len(obj.cache_images))
            )

            d = [
                {"type": "image_url", "image_url": {"url": numpy_to_base64(img)}}
                for img in selected_images
            ]

            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}, *d]}
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt2},
                        {"type": "image_url", "image_url": {"url": obj.image_base64}},
                    ],
                }
            ]
        response = vl_client.chat.completions.create(
            model="qwen2.5-vl-instruct",
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        obj.features = response.choices[0].message.content
    except:
        traceback.print_exc()


for obj in valid_items:
    if obj.image_base64 is None:
        continue

    executor.submit(get_features, obj)

executor.shutdown(wait=True)

for obj in valid_items:
    if obj.features is None:
        continue
    print(f"Object ID: {obj.object_id} :")
    print(obj.features)
    print("-----")


with open("full_track_openai.pkl", "wb") as f:
    pickle.dump(valid_items, f, protocol=pickle.HIGHEST_PROTOCOL)
