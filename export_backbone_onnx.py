import torch
import torchvision
from torch._export.converter import TS2EPConverter

# Load full TorchScript model
#backbone = torch.jit.load("models/backbone.torchscript").float().eval()

model = torch.jit.load("models/nlf_s_multi_0.2.2.torchscript").eval()
backbone = model.crop_model.backbone.float().eval()

# Extract the core module
#dummy_input = torch.randn(1, 3, 384, 384).cuda()  # for eff-L

dummy_input = torch.randn(1, 3, 256, 256).cuda()  # for eff-S

backbone = TS2EPConverter(backbone, (dummy_input,), {}).convert()

# 3. Export to ONNX
torch.onnx.export(
    backbone,
    dummy_input,
    "backbone_s.onnx",
    input_names=["input"],         # name of model input
    output_names=["output"],       # name of model output
    dynamo=True,
    do_constant_folding=True,
    dynamic_shapes=None,
)

print("✅ Exported TorchScript -> ONNX: backbone.onnx")