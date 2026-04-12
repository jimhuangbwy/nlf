import torch
import torch_tensorrt

model = torch.jit.load("models/nlf_l_multi_0.3.2.torchscript").eval()
weight_field = model.crop_model.heatmap_head.weight_field.eval()
weight_field = torch.jit.script(weight_field)
torch.jit.save(weight_field, 'models/weight_field.torchscript')
print(weight_field)
