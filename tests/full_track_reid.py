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
from ultralytics import YOLO
from ultralytics.utils import LOGGER

LOGGER.setLevel(logging.WARNING)  # 只输出 warning 以上的日志


import onnxruntime as ort
import cv2
import numpy as np


class ReIDModel:
    def __init__(self, onnx_model_path: str):
        self.session = ort.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess_for_reid(
        self, img: np.ndarray, target_size=(128, 256)
    ) -> np.ndarray:
        """
        对图像进行padding和resize，保持原始高宽比
        """
        target_w, target_h = target_size
        h, w, _ = img.shape

        # 计算缩放比例和padding尺寸
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        # 创建一个目标尺寸的画布，并将resize后的图像粘贴到中心
        pad_img = np.full((target_h, target_w, 3), 128, dtype=np.uint8)  # 用灰色填充
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        pad_img[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = img_resized

        return pad_img

    def extract_feature(self, img: np.ndarray) -> np.ndarray:
        """
        输入: img (np.ndarray), shape=[H,W,3] (BGR 或 RGB 都可以)
        输出: feature 向量 (1, D)
        """
        if img is None:
            raise ValueError("输入图像为空")

        # 确保是 RGB
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # resize (W=128, H=256)
        img_resized = self.preprocess_for_reid(img, (128, 256))

        # float32 归一化
        img_resized = img_resized.astype(np.float32) / 255.0

        # [H,W,C] → [C,H,W]
        img_resized = np.transpose(img_resized, (2, 0, 1))

        # [1,C,H,W]
        img_resized = np.expand_dims(img_resized, axis=0)

        # 推理
        features = self.session.run([self.output_name], {self.input_name: img_resized})[
            0
        ]

        # 有些 ReID 模型会输出 (1, D)，直接返回即可
        return features.squeeze()


reid_model = ReIDModel("resnet50_market1501_aicity156.onnx")

# YOLO section


model = YOLO("yolo11n.pt", verbose=False)
tracker = sv.ByteTrack(
    track_activation_threshold=0.5,
    lost_track_buffer=48,
    frame_rate=24,
)
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

video = cv2.VideoCapture("test2.mp4")


def numpy_to_base64(frame: np.ndarray) -> str:
    """
    将 NumPy 数组转换为 base64 编码的字符串。
    """
    _, buffer = cv2.imencode(".png", frame)
    return f"data:image/png;base64,{base64.b64encode(buffer).decode('utf-8')}"


class ClassTrackerObject:
    """
    追踪对象类，用于管理视频中跟踪的目标对象

    该类封装了目标对象的所有属性和操作，包括：
    - 对象标识和时间信息
    - 边界框信息管理
    - 图像特征和向量管理
    - 线程安全的更新操作
    """

    def __init__(
        self,
        object_id: str,
        start_frame: int,
        bounding_box: Tuple[int, int, int, int],
        current_bounding_box: Tuple[int, int, int, int] = None,
        end_frame: int = None,
        features: str = None,
        embed_vector=None,
        min_image_size: Tuple[int, int] = (10, 10),
    ):
        """
        初始化追踪对象

        Args:
            object_id: 对象的唯一标识符
            start_frame: 对象首次出现的帧号
            bounding_box: 初始边界框 (x1, y1, x2, y2)
            current_bounding_box: 当前边界框，默认与初始边界框相同
            end_frame: 对象最后出现的帧号
            features: 对象特征描述字符串
            embed_vector: 对象的特征向量
            min_image_size: 最小图像尺寸阈值
        """
        # 基本标识信息
        self.object_id = object_id
        self.start_frame = start_frame
        self.end_frame = end_frame if end_frame is not None else start_frame
        self.features = features

        # 边界框信息
        self.bounding_box = bounding_box  # 初始边界框
        self.current_bounding_box = current_bounding_box or bounding_box  # 当前边界框

        # 特征向量信息
        self.embed_vector = embed_vector  # ReID特征向量

        # 图像信息
        self.image = None  # 当前对象图像
        self.image_base64 = None  # 图像的base64编码
        self.cache_images = []  # 缓存的历史图像
        self.min_image_size = min_image_size  # 最小图像尺寸阈值

        # 线程安全控制
        self.lock = threading.Lock()  # 防止并发修改冲突
        self.is_updating = False  # 更新状态标记

    def update_image(self, image: np.ndarray) -> bool:
        """
        更新对象图像和相关特征

        Args:
            image: 新的对象图像数组

        Returns:
            bool: 更新是否成功
        """
        # 检查图像尺寸是否满足最小要求
        if not self._is_valid_image_size(image):
            return False

        try:
            with self.lock:
                self.is_updating = True

                # 更新图像
                self.image = image.copy()  # 创建副本避免引用问题

                # 提取并更新特征向量
                feature_vector = reid_model.extract_feature(image)
                self.update_embed_vector(feature_vector)

                # 首次设置时生成base64编码
                if self.image_base64 is None:
                    self.image_base64 = numpy_to_base64(image)

                # 缓存图像（限制缓存数量）
                self._cache_image(image)

                self.is_updating = False
                return True

        except Exception as e:
            self.is_updating = False
            logging.warning(f"更新对象 {self.object_id} 图像时发生错误: {e}")
            return False

    def _is_valid_image_size(self, image: np.ndarray) -> bool:
        """
        检查图像尺寸是否有效

        Args:
            image: 待检查的图像

        Returns:
            bool: 图像尺寸是否满足要求
        """
        return (
            image.shape[0] >= self.min_image_size[0]
            and image.shape[1] >= self.min_image_size[1]
        )

    def _cache_image(self, image: np.ndarray, max_cache_size: int = 10) -> None:
        """
        缓存图像，维护固定大小的缓存

        Args:
            image: 要缓存的图像
            max_cache_size: 最大缓存数量
        """
        self.cache_images.append(image.copy())
        # 保持缓存大小在限制范围内
        if len(self.cache_images) > max_cache_size:
            self.cache_images.pop(0)  # 移除最旧的图像

    def update_bounding_box(self, bounding_box: Tuple[int, int, int, int]) -> bool:
        """
        更新当前边界框

        Args:
            bounding_box: 新的边界框坐标 (x1, y1, x2, y2)

        Returns:
            bool: 更新是否成功
        """
        if not self._is_valid_bounding_box(bounding_box):
            return False

        with self.lock:
            self.current_bounding_box = bounding_box
            return True

    def _is_valid_bounding_box(self, bbox: Tuple[int, int, int, int]) -> bool:
        """
        验证边界框的有效性

        Args:
            bbox: 边界框坐标

        Returns:
            bool: 边界框是否有效
        """
        x1, y1, x2, y2 = bbox
        return x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0

    def update_end_frame(self, frame: int) -> bool:
        """
        更新对象结束帧号

        Args:
            frame: 新的结束帧号

        Returns:
            bool: 更新是否成功
        """
        if frame < self.start_frame:
            logging.warning(f"结束帧号 {frame} 不能小于开始帧号 {self.start_frame}")
            return False

        with self.lock:
            self.end_frame = frame
            return True

    def update_features(self, features: str) -> None:
        """
        更新对象特征描述

        Args:
            features: 新的特征描述字符串
        """
        with self.lock:
            self.features = features

    def update_embed_vector(self, new_vec, alpha: float = 0.7) -> None:
        """
        使用指数滑动平均融合新的特征向量

        Args:
            new_vec: 新的特征向量
            alpha: 融合系数，取值范围[0,1]
                  - alpha 越大，越依赖历史特征
                  - alpha 越小，越依赖新特征
        """
        if new_vec is None:
            return

        # 规范化新向量
        new_vec = self._normalize_vector(new_vec)

        if self.embed_vector is None:
            # 首次设置向量
            self.embed_vector = new_vec
        else:
            # 融合历史向量和新向量
            old_vec = self.embed_vector
            fused_vec = alpha * old_vec + (1 - alpha) * new_vec
            self.embed_vector = self._normalize_vector(fused_vec)

    def _normalize_vector(self, vector) -> np.ndarray:
        """
        向量归一化

        Args:
            vector: 待归一化的向量

        Returns:
            np.ndarray: 归一化后的向量
        """
        vec = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-6) if norm > 1e-6 else vec

    def get_duration(self) -> int:
        """
        获取对象的持续时间（帧数）

        Returns:
            int: 对象存在的帧数
        """
        return self.end_frame - self.start_frame + 1

    def get_bbox_area(self) -> int:
        """
        获取当前边界框的面积

        Returns:
            int: 边界框面积
        """
        x1, y1, x2, y2 = self.current_bounding_box
        return (x2 - x1) * (y2 - y1)

    def is_active(self) -> bool:
        """
        检查对象是否处于活跃状态（非更新状态且有有效数据）

        Returns:
            bool: 对象是否活跃
        """
        return not self.is_updating and self.image is not None

    def get_cached_images_count(self) -> int:
        """
        获取缓存图像的数量

        Returns:
            int: 缓存图像数量
        """
        return len(self.cache_images)

    def __str__(self) -> str:
        """
        返回对象的字符串表示

        Returns:
            str: 对象的详细信息字符串
        """
        image_info = f"{self.image.shape}" if self.image is not None else "None"
        vector_info = (
            f"{self.embed_vector.shape}" if self.embed_vector is not None else "None"
        )

        return (
            f"TrackerObject(id={self.object_id}, "
            f"frames={self.start_frame}-{self.end_frame}, "
            f"duration={self.get_duration()}, "
            f"bbox_initial={self.bounding_box}, "
            f"bbox_current={self.current_bounding_box}, "
            f"area={self.get_bbox_area()}, "
            f"image_shape={image_info}, "
            f"vector_shape={vector_info}, "
            f"cached_images={self.get_cached_images_count()}, "
            f"features='{self.features}')"
        )

    def __repr__(self) -> str:
        """
        返回对象的开发者友好表示

        Returns:
            str: 对象的简洁表示
        """
        return f"TrackerObject(id={self.object_id}, frames={self.start_frame}-{self.end_frame})"


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
        cv2.imwrite(f"logs/{id}.jpg", crop_img)
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
    sim_threshold: float = 0.85,
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
                obj.update_image(crop(frame, bbox, id=tracker_id, save_image=True))

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
