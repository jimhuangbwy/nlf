import torch
from ultralytics import YOLO
import torch_tensorrt

# 1) load
m = YOLO("yolo11n.pt").model
m.eval().cuda()

# 2) export to ExportedProgram
example = torch.randn(1,3,640,640).cuda()
ep = torch.export.export(m, (example,))

# 3) compile ExportedProgram to TRT
trt_ep = torch_tensorrt.compile(
    m,
    inputs=[torch_tensorrt.Input((1,3,640,640))],
    enabled_precisions={torch.float16},
)

# 4) run trt_ep (its output is torch tensor)
out = trt_ep(example.half().cuda())
print(out[0].shape)