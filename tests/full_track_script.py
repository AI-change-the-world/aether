import base64
import logging
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, Set, Tuple

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
    print(res.shape)
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

    def update_image(self, image: np.ndarray):
        if (
            image.shape[0] < self.min_image_size[0]
            or image.shape[1] < self.min_image_size[1]
        ):
            return
        with self.lock:
            self.image = image
            if self.image_base64 is not None:  # 确保 image_base64 不为 None
                self.image_base64 = base64.b64encode(image).decode("utf-8")

    def update_bounding_box(self, bounding_box: Tuple[int, int, int, int]):
        with self.lock:
            self.current_bounding_box = bounding_box

    def update_end_frame(self, frame: int):
        with self.lock:
            self.end_frame = frame

    def update_features(self, features):
        with self.lock:
            self.features = features

    def update_embed_vector(self, embed_vector):
        with self.lock:
            self.embed_vector = embed_vector

    def __str__(self):
        return f"id: {self.object_id} initial bbox: {self.bounding_box} current bbox: {self.current_bounding_box} start: {self.start_frame} end: {self.end_frame} image: {self.image.shape if self.image is not None else 'None'}   features: {self.features} "


GLOBAL_INFO: dict[str, ClassTrackerObject] = {}
executor = ThreadPoolExecutor(max_workers=4)  # 控制并发线程数

# === 工具方法 ===
def crop_and_encode(frame, bbox):
    """裁剪bbox并转为base64"""
    x1, y1, x2, y2 = map(int, bbox)
    crop_img = frame[y1:y2, x1:x2]
    return numpy_to_base64(crop_img)


def crop(frame, bbox):
    """裁剪bbox"""
    x1, y1, x2, y2 = map(int, bbox)
    return frame[y1:y2, x1:x2] if x1 < x2 and y1 < y2 else None


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
        return f"{self.object_id} -> {self.target_object_id}"


TO_BE_MERGED: Set[ToBeMergedCadidate] = set()  # 全局缓存：等待 merge 的对象


def _build_graph_from_candidates(
    candidates: Iterable["ToBeMergedCadidate"],
) -> Dict[str, Set[str]]:
    """把候选边集合转成无向图：id -> {neighbors}"""
    g: Dict[str, Set[str]] = defaultdict(set)
    for c in candidates:
        a, b = c.object_id, c.target_object_id
        if a == b:
            continue
        g[a].add(b)
        g[b].add(a)
    return g


def build_merge_chains(
    global_info: Dict[str, "ClassTrackerObject"], candidates: Set["ToBeMergedCadidate"]
) -> List[List[str]]:
    """
    根据候选边构建连通分量，每个分量是一组 ID，按 start_frame 升序排序后返回。
    """
    graph = _build_graph_from_candidates(candidates)
    visited_ids: Set[str] = set()
    chains: List[List[str]] = []

    def dfs(start_id: str) -> Set[str]:
        stack = [start_id]
        comp: Set[str] = set()
        while stack:
            nid = stack.pop()
            if nid in comp:
                continue
            comp.add(nid)
            for nb in graph.get(nid, ()):
                if nb not in comp:
                    stack.append(nb)
        return comp

    for node in list(graph.keys()):
        if node in visited_ids:
            continue
        comp = dfs(node)
        visited_ids |= comp
        # 只保留 GLOBAL_INFO 中存在的 id
        comp = {oid for oid in comp if oid in global_info}
        if not comp:
            continue
        sorted_ids = sorted(comp, key=lambda oid: global_info[oid].start_frame)
        chains.append(sorted_ids)

    return chains


def _merge_two_objects_keep_first(
    target: "ClassTrackerObject", source: "ClassTrackerObject"
) -> None:
    """
    把 source 合并进 target。策略：
    - start_frame 取最小
    - end_frame 取最大（None 视为 start_frame）
    - bbox 取 end_frame 较晚的那个
    - features 若 target 无而 source 有，则带过去
    - embed_vector：若两者都有，做简单平均（归一化后）；否则取现有的那个
    """
    # 起止帧
    src_start = source.start_frame
    src_end = source.end_frame if source.end_frame is not None else source.start_frame
    tgt_start = target.start_frame
    tgt_end = target.end_frame if target.end_frame is not None else target.start_frame

    target.start_frame = min(tgt_start, src_start)
    target.update_end_frame(max(tgt_end, src_end))

    # bbox 用“谁的 end 更晚就用谁的”
    if src_end >= tgt_end:
        target.update_bounding_box(source.bounding_box)

    # features
    if getattr(target, "features", None) in (None, "", {}) and getattr(
        source, "features", None
    ) not in (None, "", {}):
        target.update_features(source.features)

    # 向量合并
    v_t = getattr(target, "embed_vector", None)
    v_s = getattr(source, "embed_vector", None)

    def _norm(v):
        if v is None:
            return None
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None

    nv_t = _norm(v_t)
    nv_s = _norm(v_s)
    if nv_t is not None and nv_s is not None:
        mixed = nv_t + nv_s
        n = np.linalg.norm(mixed)
        if n > 0:
            target.update_embed_vector(mixed / n)
        else:
            target.update_embed_vector(nv_t)  # 退化
    elif nv_t is None and nv_s is not None:
        target.update_embed_vector(nv_s)
    # 否则保留 target 的


def merge_by_similarity(
    global_info: Dict[str, "ClassTrackerObject"],
    chains: List[List[str]],
    candidates: Set["ToBeMergedCadidate"],
    sim_threshold: float = 0.8,
) -> Set[str]:
    """
    遍历每条轨迹链，按时间顺序两两比较：
    - 相似度 >= 阈值：把后者合并进前者（保留前者 ID）
    - 否则：前者切换为后者，继续往后比较
    返回：被删除（并入他人）的 ID 集合
    同时会从 GLOBAL_INFO 里删除这些 ID
    """
    removed_ids: Set[str] = set()

    for chain in chains:
        if not chain:
            continue
        merged_kept_id = chain[0]  # 当前保留者
        for next_id in chain[1:]:
            # 如果 next_id 在之前链处理中已被删，跳过
            if next_id in removed_ids or next_id not in global_info:
                continue
            if merged_kept_id not in global_info:
                # 极端情况：之前被别的链删除了，重置
                merged_kept_id = next_id
                continue

            obj_a = global_info[merged_kept_id]
            obj_b = global_info[next_id]

            if obj_a.embed_vector is None:
                obj_a.update_embed_vector(get_clip_vector(obj_a.image))
            if obj_b.embed_vector is None:
                obj_b.update_embed_vector(get_clip_vector(obj_b.image))

            sim = clip_similarity(
                np.asarray(obj_a.embed_vector), np.asarray(obj_b.embed_vector)
            )
            print(f" Comparing {merged_kept_id} and {next_id}, similarity: {sim}")
            if sim >= sim_threshold:
                # 合并：把 next_id 并入 merged_kept_id
                _merge_two_objects_keep_first(obj_a, obj_b)
                # 删除 next_id
                del global_info[next_id]
                removed_ids.add(next_id)
                # merged_kept_id 不变，继续尝试与下一个合并
            else:
                # 不合并，移动窗口
                merged_kept_id = next_id

    # 清理 TO_BE_MERGED：删除包含已移除 ID 的候选
    if candidates:
        to_drop = set()
        for c in candidates:
            if c.object_id in removed_ids or c.target_object_id in removed_ids:
                to_drop.add(c)
        candidates.difference_update(to_drop)

    return removed_ids


def merge(
    global_info: Dict[str, "ClassTrackerObject"],
    candidates: Set["ToBeMergedCadidate"],
    sim_threshold: float = 0.8,
) -> Set[str]:
    """
    执行一次合并流程：构建链 -> 合并 -> 清理候选
    返回本轮被合并删除的 ID 集合
    """
    chains = build_merge_chains(global_info, candidates)
    if not chains:
        print("No chains to merge.")
        return set()
    print(f"Found {len(chains)} chains to merge.")
    for i, chain in enumerate(chains):
        print(f"Chain {i+1}: {chain}")
    removed_ids = merge_by_similarity(
        global_info, chains, candidates, sim_threshold=sim_threshold
    )
    print(f"Merge completed, removed IDs: {removed_ids}")
    return removed_ids


if __name__ == "__main__":
    frame_id = 0

    while True:
        success, frame = video.read()
        if not success or frame_id > 480:  # 限制处理前 240 帧
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
                obj.update_image(crop(frame, bbox))
                # 找到GLOBAL_INFO中位置差不多的object,构建 ToBeMergedCadidate
                for obj in GLOBAL_INFO.values():
                    if obj.object_id == tracker_id:
                        continue
                    if (
                        bndbox_overlap(obj.bounding_box, bbox) > bndbox_threshold
                        or bndbox_overlap(obj.current_bounding_box, bbox)
                    ) and (
                        abs(obj.end_frame - obj.start_frame) <= 5
                        or abs(obj.start_frame - obj.end_frame) <= 5
                    ):
                        TO_BE_MERGED.add(ToBeMergedCadidate(tracker_id, obj.object_id))

                # 异步更新 feature
                # if not obj.is_updating:
                #     obj.is_updating = True
                #     executor.submit(async_update_feature, obj, frame)
            else:
                # 已存在对象，更新最后一次 bbox
                GLOBAL_INFO[tracker_id].update_bounding_box(bbox)
                GLOBAL_INFO[tracker_id].update_end_frame(frame_id)
                GLOBAL_INFO[tracker_id].update_image(crop(frame, bbox))

        if frame_id % frame_rate == 0:
            # 每隔 frame_rate 帧检查一次
            executor.submit(merge, GLOBAL_INFO, TO_BE_MERGED, sim_threshold=0.8)

    executor.submit(merge, GLOBAL_INFO, TO_BE_MERGED, sim_threshold=0.8)
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
