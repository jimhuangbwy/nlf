import torch
import torch_tensorrt

model = torch.jit.load("models/backbone.torchscript").half().eval().cuda()

# model = torch.jit.load("models/nlf_l_multi_0.3.2.torchscript").eval()
# backbone= model.crop_model.backbone


inputs = [torch.randn((1, 3, 384, 384)).half().cuda()] # define a list of representative inputs here

enabled_precisions = {torch.half}  # Run with fp16

trt_ts_module = torch_tensorrt.compile(
    model, inputs=inputs, enabled_precisions=enabled_precisions, ir='torchscript', require_full_compilation=True,
)

torch.jit.save(trt_ts_module, "models/backbone.ts")
print('backbone.ts was saved')

input = torch.randn((1, 3, 384, 384)).half().cuda()
result = trt_ts_module(input)
print(result)
