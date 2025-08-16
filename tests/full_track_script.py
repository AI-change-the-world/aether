import base64
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import supervision as sv
import torch
from openai import OpenAI
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from ultralytics import YOLO
from ultralytics.utils import LOGGER

LOGGER.setLevel(logging.WARNING)  # 只输出 warning 以上的日志

# CLIP section
# 初始化模型和处理器（全局使用，避免重复加载）
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("/root/models/clip").to(device)
processor = CLIPProcessor.from_pretrained("/root/models/clip")


def get_clip_vector(img: np.ndarray) -> np.ndarray:
    """
    获取图像的 CLIP 向量
    参数:
        img: np.ndarray，形状 H x W x C，像素值范围 0-255
    返回:
        np.ndarray，归一化后的 CLIP 向量
    """
    pil_img = Image.fromarray(img.astype(np.uint8))
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    # 归一化
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    res = features.cpu().numpy().flatten()
    # print(res.shape)
    return res


def clip_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    计算两个 CLIP 向量的余弦相似度
    参数:
        vec1, vec2: np.ndarray，归一化后的 CLIP 向量
    返回:
        float，相似度 [-1, 1]，越接近 1 越相似
    """
    vec1 = torch.tensor(vec1)
    vec2 = torch.tensor(vec2)
    similarity = torch.nn.functional.cosine_similarity(
        vec1.unsqueeze(0), vec2.unsqueeze(0)
    )
    return similarity.item()


# MLLM section

prompt = """
你是一个专业的视频图像分析专家，请仔细观察输入的图像或视频帧中的人物，并严格按照以下格式输出人物特征（不允许输出多余信息，不允许省略字段）：

性别：男 / 女 / 未知
发型：长发 / 短发 / 光头 / 其他  
发色：黑色 / 棕色 / 黄色 / 白色 / 灰色 / 其他  
上身：颜色 + 类型（如 红色短袖、黑色夹克、白色衬衫、蓝色毛衣 等）  
下身：颜色 + 类型（如 蓝色牛仔裤、黑色长裤、灰色短裙、白色运动裤 等）  
鞋子：颜色 + 类型（如 白色运动鞋、黑色皮鞋、灰色凉鞋 等）  
配饰：如帽子（颜色+类型）、眼镜、耳机、口罩、包等（如果没有则写“无”）  
特殊动作或状态：如抽烟、打电话、背包、推车、跑步等（如果没有则写“无”）  

注意：
1. 只分析人物外观，不推测身份。  
2. 如果无法判断，请写“未知”。  
3. 保证输出严格按照上述字段顺序和格式。
"""


def numpy_to_base64(frame: np.ndarray) -> str:
    """
    将 NumPy 数组转换为 base64 编码的字符串。
    """
    _, buffer = cv2.imencode(".png", frame)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"


vl_client = OpenAI(api_key="sk-x", base_url="http://localhost:9997/",)


def parse_person_features(
    llm_output: str, start_step: int, end_step: int, person_id: int
):
    """
    将 LLM 输出的特征文本解析为 JSON 结构。
    输入:
        llm_output: str  —— LLM 按 prompt 格式输出的人物特征
        start_step: int  —— 起始帧
        end_step: int    —— 结束帧
        person_id: int   —— 人物 ID
    返回:
        dict —— {"start_step": int, "end_step": int, "person_id": int, "features": {...}}
    """
    # 用正则匹配每个字段
    patterns = {
        "性别": r"性别：(.+)",
        "发型": r"发型：(.+)",
        "发色": r"发色：(.+)",
        "上身": r"上身：(.+)",
        "下身": r"下身：(.+)",
        "鞋子": r"鞋子：(.+)",
        "配饰": r"配饰：(.+)",
        "特殊动作或状态": r"特殊动作或状态：(.+)",
    }

    features = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, llm_output)
        if match:
            features[key] = match.group(1).strip()
        else:
            features[key] = "未知"  # 如果没匹配到，就写未知

    return {
        "start_step": start_step,
        "end_step": end_step,
        "person_id": person_id,
        "features": features,
    }


# YOLO section


model = YOLO("yolo11n.pt", verbose=False)
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

video = cv2.VideoCapture("test.mp4")


class ClassTrackerObject:
    def __init__(
        self,
        object_id: str,
        start_frame: int,
        bounding_box: Tuple[int, int, int, int],
        current_bounding_box: Tuple[int, int, int, int] = None,
        end_frame: int = None,
        features: str = None,
        embed_vector=None,
        min_image_size=(10, 10),
    ):
        self.object_id = object_id
        self.start_frame = start_frame
        self.end_frame = end_frame if end_frame else start_frame
        self.features = features
        self.bounding_box = bounding_box
        self.embed_vector = embed_vector  # CLIP 向量
        self.lock = threading.Lock()  # 避免并发写冲突
        self.is_updating = False  # 标记是否正在更新

        self.image = None  # 用于存储裁剪后的图像, 可能需要存储多个对象
        self.image_base64 = None  # 用于存储裁剪图像的 base64 编码
        self.min_image_size = min_image_size  # 最小图像尺寸，避免过小的图像
        self.current_bounding_box = current_bounding_box or bounding_box

        self.cache_images = []

    def update_image(self, image: np.ndarray):
        if (
            image.shape[0] < self.min_image_size[0]
            or image.shape[1] < self.min_image_size[1]
        ):
            return
        with self.lock:
            self.image = image
            self.update_embed_vector(get_clip_vector(image))
            if self.image_base64 is None:  # 确保 image_base64 不为 None
                self.image_base64 = numpy_to_base64(image)
            self.cache_images.append(image)

    def update_bounding_box(self, bounding_box: Tuple[int, int, int, int]):
        with self.lock:
            self.current_bounding_box = bounding_box

    def update_end_frame(self, frame: int):
        with self.lock:
            self.end_frame = frame

    def update_features(self, features):
        with self.lock:
            self.features = features

    def update_embed_vector(self, new_vec, alpha=0.7):
        """
        融合新的向量：指数滑动平均
        alpha 越大，越依赖历史；越小，越依赖新特征
        """
        new_vec = np.asarray(new_vec, dtype=np.float32)
        new_vec = new_vec / (np.linalg.norm(new_vec) + 1e-6)

        if getattr(self, "embed_vector", None) is None:
            self.embed_vector = new_vec
        else:
            old_vec = self.embed_vector
            fused = alpha * old_vec + (1 - alpha) * new_vec
            fused /= np.linalg.norm(fused) + 1e-6
            self.embed_vector = fused

    def __str__(self):
        return f"id: {self.object_id} initial bbox: {self.bounding_box} current bbox: {self.current_bounding_box} start: {self.start_frame} end: {self.end_frame} image: {self.image.shape if self.image is not None else 'None'}   features: {self.features} vector: {self.embed_vector.shape if self.embed_vector is not None else 'None'} "


GLOBAL_INFO: dict[str, ClassTrackerObject] = {}
executor = ThreadPoolExecutor(max_workers=8)  # 控制并发线程数

# === 工具方法 ===
def crop_and_encode(frame, bbox):
    """裁剪bbox并转为base64"""
    x1, y1, x2, y2 = map(int, bbox)
    crop_img = frame[y1:y2, x1:x2]
    return numpy_to_base64(crop_img)


def crop(frame, bbox, id: str = "", save_image: bool = False):
    """裁剪bbox"""
    x1, y1, x2, y2 = map(int, bbox)
    crop_img = frame[y1:y2, x1:x2] if x1 < x2 and y1 < y2 else None
    if save_image:
        cv2.imwrite(f"{id}.jpg", crop_img)
    return crop_img


bndbox_threshold = 0.5  # 重叠阈值


def bndbox_overlap(
    bndbox1: Tuple[float, float, float, float],
    bndbox2: Tuple[float, float, float, float],
) -> float:
    """计算两个边界框的重叠面积比例"""
    x1, y1, x2, y2 = bndbox1
    x1_p, y1_p, x2_p, y2_p = bndbox2

    # 先快速排除不可能相交的情况
    if x2 <= x1_p or x2_p <= x1 or y2 <= y1_p or y2_p <= y1:
        return 0.0

    # 计算重叠区域
    overlap_x1 = max(x1, x1_p)
    overlap_y1 = max(y1, y1_p)
    overlap_x2 = min(x2, x2_p)
    overlap_y2 = min(y2, y2_p)

    overlap_area = max(0, overlap_x2 - overlap_x1) * max(0, overlap_y2 - overlap_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_p - x1_p) * (y2_p - y1_p)

    return overlap_area / min(area1, area2)


frame_rate = 24


class ToBeMergedCadidate:
    def __init__(self, object_id: str, target_object_id: str):
        self.object_id = object_id
        self.target_object_id = target_object_id
        self.similarity = 0.0
        self.dist = 0.0  # 用于存储 bbox 中心点距离

    def __eq__(self, value):
        if not isinstance(value, ToBeMergedCadidate):
            return False
        return (
            self.object_id == value.object_id
            and self.target_object_id == value.target_object_id
        )

    def __hash__(self):
        return hash((self.object_id, self.target_object_id))

    def __str__(self):
        return f"{self.object_id} -> {self.target_object_id}, similarity: {self.similarity:.4f}, dist: {self.dist:.2f}"


candidates_to_merge = set()


def merge_candidates_by_similarity_and_bbox(
    global_info: Dict[str, "ClassTrackerObject"],
    sim_threshold: float = 0.8,
    # max_bbox_move: float = 50.0,  # bbox中心最大移动像素阈值
) -> Set["ToBeMergedCadidate"]:
    """
    循环遍历 global_info 中的对象，按 start_frame 排序两两比对，
    如果 bbox 移动小于 max_bbox_move 且 embedding 相似度大于 sim_threshold，
    就加入 ToBeMergedCadidate 集合。
    """
    from itertools import combinations

    global candidates_to_merge

    objs = list(global_info.values())
    # 按 start_frame 排序
    objs.sort(key=lambda o: o.start_frame)

    def bbox_center(bbox):
        x1, y1, x2, y2 = bbox
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    for obj_a, obj_b in combinations(objs, 2):
        # 获取 obj_a 的结束帧和 obj_b 的起始帧
        end_a = obj_a.end_frame if obj_a.end_frame is not None else obj_a.start_frame
        start_b = (
            obj_b.start_frame if obj_b.start_frame is not None else obj_b.end_frame
        )

        # 严格时间顺序：后一个对象第一帧 >= 前一个对象最后一帧
        if start_b <= end_a:
            continue

        # 计算 bbox 中心移动距离
        center_a = bbox_center(obj_a.bounding_box)
        center_b = bbox_center(obj_b.bounding_box)
        dist = (
            (center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2
        ) ** 0.5
        # if dist > 5 * (start_b - end_a):
        #     continue
        if dist > 50:  # 这里可以调整阈值，50 是一个经验值
            continue

        # 计算 embedding 相似度
        v_a = getattr(obj_a, "embed_vector", None)
        v_b = getattr(obj_b, "embed_vector", None)
        if v_a is None or v_b is None:
            continue
        cos_sim = np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b) + 1e-6)
        if cos_sim < sim_threshold:
            continue

        tbm = ToBeMergedCadidate(obj_b.object_id, obj_a.object_id)
        tbm.similarity = cos_sim
        tbm.dist = dist
        # 同时满足条件，加入候选集合
        candidates_to_merge.add(tbm)


def build_time_ordered_chains_with_position_and_similarity(
    global_info: Dict[str, "ClassTrackerObject"],
    candidates: Set["ToBeMergedCadidate"],
    base_dist: float = 5,
    sim_threshold: float = 0.8,
) -> List[List[str]]:
    """
    构建按时间顺序排列的链条，同时根据位置和向量相似度判断是否可以合并。
    - base_dist: 时间跨度为1帧时允许的最大中心点距离
    - sim_threshold: 融合向量的最小相似度要求
    """
    from collections import defaultdict

    def cosine_similarity(vec1, vec2):
        if vec1 is None or vec2 is None:
            return 0.0
        v1 = np.asarray(vec1, dtype=np.float32).reshape(-1)
        v2 = np.asarray(vec2, dtype=np.float32).reshape(-1)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    # 构建邻接表
    graph = defaultdict(list)
    indegree = defaultdict(int)
    for pair in candidates:
        graph[pair.target_object_id].append(pair.object_id)
        indegree[pair.object_id] += 1
        indegree.setdefault(pair.target_object_id, 0)

    start_nodes = [oid for oid, deg in indegree.items() if deg == 0]
    chains = []
    visited_global = set()

    for start in start_nodes:
        chain = [start]
        visited_local = {start}
        current = start

        while True:
            children = graph.get(current, [])
            if not children:
                break

            # 按 start_frame 排序
            children.sort(key=lambda x: global_info[x].start_frame)

            added = False
            for oid in children:
                if oid in visited_local:
                    continue

                a = global_info[current]
                b = global_info[oid]

                # 时间顺序检查
                end_a = a.end_frame if a.end_frame is not None else a.start_frame
                start_b = b.start_frame
                if start_b < end_a:
                    continue

                # 中心点距离检查
                x_a = (a.bounding_box[0] + a.bounding_box[2]) / 2
                y_a = (a.bounding_box[1] + a.bounding_box[3]) / 2
                x_b = (b.bounding_box[0] + b.bounding_box[2]) / 2
                y_b = (b.bounding_box[1] + b.bounding_box[3]) / 2
                dist = ((x_b - x_a) ** 2 + (y_b - y_a) ** 2) ** 0.5
                dt = start_b - end_a
                max_dist = base_dist * dt
                if dist > max_dist:
                    continue

                # 相似度检查：与链条中所有节点都要超过阈值
                all_sim_ok = True
                for prev_oid in chain:
                    sim = cosine_similarity(
                        global_info[prev_oid].embed_vector,
                        global_info[oid].embed_vector,
                    )
                    if sim < sim_threshold:
                        all_sim_ok = False
                        break
                if not all_sim_ok:
                    continue

                # 满足所有条件，加入链条
                chain.append(oid)
                visited_local.add(oid)
                current = oid
                added = True
                break  # 每次只加一个

            if not added:
                break

        chains.append(chain)
        visited_global.update(chain)

    # 去掉单节点链条
    chains = [chain for chain in chains if len(chain) > 1]
    chains = filter_chains_unique_nodes(chains)
    return chains


def filter_chains_unique_nodes(chains):
    """
    保证每个节点只在结果链条中出现一次。
    出现重复的节点，整条链条舍弃。
    """
    all_nodes = set()
    filtered = []

    for chain in chains:
        if any(node in all_nodes for node in chain):
            continue  # 这条链条有重复节点，舍弃
        filtered.append(chain)
        all_nodes.update(chain)

    return filtered


if __name__ == "__main__":
    frame_id = 0

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
                obj.update_image(crop(frame, bbox, id=tracker_id, save_image=False))

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
                    executor.submit(
                        GLOBAL_INFO[tracker_id].update_image, crop(frame, bbox)
                    )

        if frame_id % frame_rate == 0:
            # 每隔 frame_rate 帧检查一次
            executor.submit(merge_candidates_by_similarity_and_bbox, GLOBAL_INFO)

    executor.submit(merge_candidates_by_similarity_and_bbox, GLOBAL_INFO)
    executor.shutdown(wait=True)

    print("Tracking complete.")
    video.release()
    print(
        "============================================================================================"
    )
    valid_items = [
        obj for obj in GLOBAL_INFO.values() if obj.end_frame - obj.start_frame >= 24
    ]
    print("Tracked objects:")
    for obj in valid_items:
        print(obj)
    print(
        "============================================================================================"
    )
    print("all objects:")
    for obj in GLOBAL_INFO.values():
        print(obj)
    print(
        "============================================================================================"
    )
    print("Total frames:", frame_id)
    print("Total objects:", len(valid_items))

    print("Candidates to merge:")
    for candidate in candidates_to_merge:
        print(candidate)
    chains = build_time_ordered_chains_with_position_and_similarity(
        GLOBAL_INFO, candidates_to_merge
    )
    print("Chains of merged objects:")
    for chain in chains:
        print(chain)
