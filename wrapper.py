import torch
import torchvision  # Must import this for the model to load without error
import torch_tensorrt

class NLFWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model.detect_smpl_batched(x)
        # Flatten dict into a tuple of tensors

        def ensure_tensor(val):
            if isinstance(val, torch.Tensor):
                return val
            elif isinstance(val, (list, tuple)):
                return torch.stack(val, dim=0)  # merge list of tensors
            else:
                raise TypeError(f"Unsupported type: {type(val)}")

        return ensure_tensor(out["joints3d"]
        )


# Load your original TorchScript
ts_model = torch.jit.load("models/nlf_l_multi_0.3.2.torchscript").cuda().eval()

# Wrap it
input = torch.randn(1, 3, 480, 640).cuda()
traced = torch.jit.trace(NLFWrapper(ts_model), input)

trt_module = torch_tensorrt.compile(
    traced,
    inputs=[torch_tensorrt.Input(
        shape=(1, 3, 480, 640),   # 🔹 fixed shape
        dtype=torch.half          # FP16 for speed (if GPU supports)
    )],
    enabled_precisions={torch.half}  # or {torch.float} for FP32
)

torch.jit.save(trt_module, "model/nlf_trt.ts")



# # Script it again so it has a forward()
# scripted = torch.jit.script(wrapped)
# torch.jit.save(scripted, "models/nlf_with_forward.ts")