from ultralytics import YOLO


model = YOLO("yolo11n.pt")

#model.export(format="torchscript", half=True, device=0, nms=False)  # creates 'yolo11n.engine'

#model.export(format="torchscript", half=True, device='cpu')  # creates 'yolo11n.engine'

model = model.export(format="onnx", nms=True, device=0)  # creates 'yolo11n.engine'

#model = model.export(format="onnx")  # creates 'yolo11n.engine'

#results = model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)


