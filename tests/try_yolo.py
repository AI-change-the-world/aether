import supervision as sv
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

# Run batched inference on a list of images
results = model(["1.webp"])  # return a list of Results objects

result = results[0]

detections = sv.Detections.from_ultralytics(result)

print(detections)
