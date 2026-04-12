import torch
import torch_tensorrt

# ------------------------------
# Step 1: Load TorchScript backbone
# ------------------------------
# Load full TorchScript model
model = torch.jit.load("models/nlf_l_multi_0.3.2.torchscript").eval().cuda()

# Extract the core module
backbone = model.crop_model.backbone.half()
print("✅ Loaded backbone TorchScript")

inputs = [torch.randn((1, 3, 384, 384)).cuda().half()] # define a list of representative inputs here

trt_gm = torch_tensorrt.compile(backbone, ir="torchscript", inputs=inputs, enabled_precisions={torch.float16})
torch.jit.save(trt_gm, "models/backbone.ts") # PyTorch only supports Python runtime for an ExportedProgram. For C++ deployment, use a TorchScript file