from ultralytics import YOLO

# Load the YOLOv8n model
model = YOLO("yolov8n.pt").to('cuda')
# Export the model to TorchScript format
model.export(format="torchscript", half=True, device=0)  # creates 'yolov8n.torchscript'