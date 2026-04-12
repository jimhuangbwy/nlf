import torch
import torch_tensorrt

model = torch.jit.load("models/nlf_l_multi_0.3.2.torchscript").eval()
layer = model.crop_model.heatmap_head.layer.eval()
layer = torch.jit.script(layer)
torch.jit.save(layer, 'models/layer.torchscript')
print(layer)
