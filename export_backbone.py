import torch
import torch_tensorrt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = torch.jit.load("models/nlf_l_multi_0.3.2.torchscript").eval().half()
backbone = model.crop_model.backbone.eval().half()
backbone = torch.jit.script(backbone)
torch.jit.save(backbone, 'models/backbone.torchscript')
