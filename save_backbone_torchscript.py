import torch
import torchvision

model = torch.jit.load("models/nlf_l_multi_0.3.2.torchscript").eval()
backbone= model.crop_model.backbone
inputs = [torch.randn((1, 3, 384, 384)).half().cuda()] # define a list of representative inputs here
backbone = torch.jit.script(backbone)
torch.jit.save(backbone, "models/backbone.torchscript")