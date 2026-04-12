import torch

from nlf.pt.models import nlf_model_trt2_cpu
from nlf.pt.multiperson import person_detector, multiperson_model_trt

device = torch.device('cpu')  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = torch.jit.load("models/nlf_l_multi_0.3.2.torchscript").to(device).eval().half()

detector = person_detector.PersonDetector('models/yolo11n_cpu.torchscript').to(device).eval().half()
# detector = person_detector_trt.PersonDetector('models/yolo11n.engine').to(device).eval().half()

backbone = model.crop_model.backbone.to(device).eval().half()
# backbone = torch.jit.load("models/backbone.ts").to(device).eval()
# backbone = TRTInference('models/backbone.engine')

weight_field = model.crop_model.heatmap_head.weight_field.to(device).eval().half()
# weight_field = pt_field.build_field().eval().cuda()
# weight_field = torch.jit.load("models/weight_field.ts").to(device).eval().half()

layer = model.crop_model.heatmap_head.layer.to(device).eval().half()
# layer = torch.jit.load("models/layer.ts").to(device).eval().half()
# layer = TRTInference('models/layer.engine')

model_pytorch = nlf_model_trt2_cpu.NLFModel(backbone, weight_field, layer).to(device).eval().half()

model = multiperson_model_trt.MultipersonNLF(
    model_pytorch, detector, device=device).to(device).eval().half()

model = torch.jit.script(model)
torch.jit.save(model, 'models/end_model_cpu.torchscript')