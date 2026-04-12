import torch
import torch_tensorrt

model = torch.jit.load("models/weight_field.torchscript").eval().cuda()

# model = torch.jit.load("models/nlf_l_multi_0.3.2.torchscript").eval()
# backbone= model.crop_model.backbone


inputs = [torch.randn((1048,3)).half().cuda()] # define a list of representative inputs here

trt_ts_module = torch_tensorrt.compile(
    model, inputs=inputs, enabled_precisions={torch.float16}, ir='torchscript')

torch.jit.save(trt_ts_module, "models/weight_field.ts")
print('weight_field.ts was saved')

input = torch.randn((1048,3)).half().cuda()
result = trt_ts_module(input)
print(result)
