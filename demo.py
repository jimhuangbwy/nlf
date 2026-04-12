# PyTorch version

import torch
import torchvision  # Must import this for the model to load without error

model = torch.jit.load('models/nlf_l_multi_0.3.2.torchscript').cuda().eval()
image = torchvision.io.read_image('example_image.jpg').cuda()
frame_batch = image.unsqueeze(0)

with torch.inference_mode(), torch.device('cuda'):
   pred = model.detect_smpl_batched(frame_batch)

print(pred['joints3d'][0], pred['joints3d'][0].shape)