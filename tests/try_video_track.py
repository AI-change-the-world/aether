import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    try:

        labels = [
            f"#{tracker_id} {class_name}"
            for class_name, tracker_id in zip(
                detections.data["class_name"], detections.tracker_id
            )
        ]

        annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
        return label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels
        )
    except:
        return frame


sv.process_video(source_path="test.mp4", target_path="result.mp4", callback=callback)
