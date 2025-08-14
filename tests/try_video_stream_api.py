"""
ffmpeg -re -stream_loop -1 -i test.mp4  -c copy -f rtsp rtsp://localhost:8554/mystream
"""

import os

import cv2
import supervision as sv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

# 配置 RTSP 传输参数（UDP 可降低延迟，但可能丢包）
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

# 加载 YOLO 模型
model = YOLO("yolo11n.pt")

# 初始化 FastAPI
app = FastAPI()

# 打开 RTSP 视频流（替换为你的实际地址）
video = cv2.VideoCapture("rtsp://localhost:8554/mystream")

# 初始化 ByteTrack 追踪器
tracker = sv.ByteTrack()

# 初始化绘制器
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


def gen_frames():
    while True:
        success, frame = video.read()
        if not success:
            break

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
            # 如果没有检测到目标，直接编码并输出原始帧
            success, encoded_image = cv2.imencode(".jpg", frame)
            if not success:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encoded_image.tobytes() + b"\r\n"
            )
            continue

        # 为每个目标生成标签（带 tracker_id）
        labels = [
            f"#{tracker_id} {class_name}"
            for class_name, tracker_id in zip(
                detections.data["class_name"], detections.tracker_id
            )
        ]

        # 画框 + 画标签
        annotated_frame = box_annotator.annotate(
            scene=frame.copy(), detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        # 编码为 JPEG
        success, encoded_image = cv2.imencode(".jpg", annotated_frame)
        if not success:
            continue

        # 输出 MJPEG 帧
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + encoded_image.tobytes() + b"\r\n"
        )


@app.get("/video")
def video_feed():
    return StreamingResponse(
        gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=15234)
