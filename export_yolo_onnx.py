from ultralytics.models import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Export the model to ONNX format
model.export(format="onnx", device=0, nms=False)  # creates 'yolo11n.onnx'