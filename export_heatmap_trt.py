import torch

class Wrapper(torch.nn.Module):
    def __init__(self, ts_model):
        super().__init__()
        self.ts_model = ts_model
        self.aug_should_flip = (torch.arange(0, 0, device='cuda') - 0 // 2) % 2 != 0
        self.aug_should_flip_flat = torch.repeat_interleave(self.aug_should_flip, n_cases, dim=0)

    def forward(self, x):
        # Call the real inference method
        return self.ts_model.predict_multi_same_weights(
            x, new_intrinsic_matrix_flat=None, weights, self.aug_should_flip_flat
        )

# Load full TorchScript model
model = torch.jit.load("models/nlf_l_multi_0.3.2.torchscript").eval().cuda()

# Extract the core module
heatmap_head = model.crop_model

# Load TorchScript
ts_model = torch.jit.load("model.torchscript")

# Wrap it
model = Wrapper(ts_model).eval().cuda()

dummy_input = torch.rand(1, 3, 480, 640).cuda()

# 3. Export to ONNX
torch.onnx.export(
    heatmap_head,
    dummy_input,
    "heatmap_head.onnx",
    input_names=["input"],         # name of model input
    output_names=["output"],       # name of model output
    dynamic_axes={                 # allow dynamic batch size
        "input": {0: "batch"},
        "output": {0: "batch"}
    },
    opset_version=12               # ONNX opset (12–13 recommended for TensorRT)
)

print("✅ Exported TorchScript -> ONNX: model.onnx")

# import torch_tensorrt
#
# inputs = [torch_tensorrt.Input(
#             shape=(1, 3, 480, 640),   # fixed input shape
#             dtype=torch.half          # FP16 for speed
#             )]
#
# heatmap_trt = torch_tensorrt.compile(
#     heatmap,
#     inputs=inputs,
#     enabled_precisions={torch.half},  # FP16
# )
#
# torch.jit.save(heatmap_trt, "models/heatmap_trt.ts")
# torch.jit.save(heatmap_trt, "models/heatmap_trt.ep") # PyTorch only supports Python runtime for an ExportedProgram. For C++ deployment, use a TorchScript filet