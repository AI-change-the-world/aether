import numpy as np
import supervision as sv
from full_track_script import *
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

global_map: dict = {}


def __get_target_object_id(object_id: str) -> str:
    if object_id in global_map:
        target_tracker_id = global_map[object_id]
    else:
        target_tracker_id = object_id
    return target_tracker_id


def callback(frame: np.ndarray, frame_id: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    person_class_id = 0
    mask = detections.class_id == person_class_id
    detections = detections[mask]

    if len(detections) == 0:
        # 如果没有检测到目标，直接返回原始帧
        return frame

    for bbox, tracker_id in zip(detections.xyxy, detections.tracker_id):
        __tracker_id = str(tracker_id.item())
        if __tracker_id not in GLOBAL_INFO:
            # 新对象
            obj = ClassTrackerObject(
                __tracker_id, start_frame=frame_id, bounding_box=bbox
            )
            GLOBAL_INFO[__tracker_id] = obj
            # 更新裁剪图像
            obj.update_image(crop(frame, bbox, id=__tracker_id, save_image=False))

            # 异步更新 feature
            # if not obj.is_updating:
            #     obj.is_updating = True
            #     executor.submit(async_update_feature, obj, frame)
        else:
            # 已存在对象，更新最后一次 bbox
            GLOBAL_INFO[__tracker_id].update_bounding_box(bbox)
            GLOBAL_INFO[__tracker_id].update_end_frame(frame_id)
            if frame_id % frame_rate == 0:
                # 每隔 frame_rate 帧更新一次图像
                # executor.submit(
                #     GLOBAL_INFO[tracker_id].update_image, crop(frame, bbox)
                # )
                GLOBAL_INFO[__tracker_id].update_image(crop(frame, bbox))

    if frame_id % frame_rate == 0:
        merge_candidates_by_similarity_and_bbox(GLOBAL_INFO)

    chains = build_time_ordered_chains_with_position_and_similarity(
        GLOBAL_INFO, candidates_to_merge
    )

    for chain in chains:
        global_map[chain[0]] = chain[0]
        for i in range(1, len(chain) - 1):
            global_map[chain[i]] = chain[0]

    labels = [
        f"#{__get_target_object_id(str(tracker_id))} {class_name}"
        for class_name, tracker_id in zip(
            detections.data["class_name"], detections.tracker_id
        )
    ]

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels
    )


sv.process_video(
    source_path="test2.mp4", target_path="result_n2.mp4", callback=callback
)
