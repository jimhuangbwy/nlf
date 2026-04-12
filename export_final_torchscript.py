import torch
import torchvision
from torch._export.converter import TS2EPConverter

from export_heatmap_trt import heatmap_head
from nlf.pt.multiperson import person_detector, multiperson_model_trt
import nlf.pt.models.nlf_model_trt2 as nlf_model_trt2
#from onnx_helper import TRTModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = torch.jit.load("models/nlf_l_multi_0.3.2.torchscript").to(device).eval().float()

backbone = model.crop_model.backbone.to(device).eval().float()
#backbone = torch.jit.load("models/backbone.ts").to(device).eval().half()
dummy_input = torch.randn(1, 3, 384, 384).cuda()
backbone = TS2EPConverter(backbone, (dummy_input,), {}).convert()

weight_field = model.crop_model.heatmap_head.weight_field.to(device).eval().float()
#weight_field = torch.jit.load("models/weight_field.torchscript").to(device).eval().half()
#weight_field = torch.jit.load("models/weight_field.ts").to(device).eval().half()
dummy_input = torch.randn(1048, 3).cuda()
weight_field = TS2EPConverter(weight_field, (dummy_input,), {}).convert()

heatmap_head = model.crop_model.heatmap_head.to(device).eval()
dummy_input = torch.randn(1048, 3).cuda()
weight_field = TS2EPConverter(weight_field, (dummy_input,), {}).convert()


layer = model.crop_model.heatmap_head.layer.to(device).eval().float()
#layer = torch.jit.load("models/layer.ts").to(device).eval().half()
#layer = torch.jit.load("models/layer.torchscript").to(device).eval().half()
model_pytorch = nlf_model_trt2.NLFModel(backbone, weight_field, layer).to(device).eval()

# yolo11nEngine = TRTModule('yolo11n.engine')
# detector = person_detector.PersonDetector(yolo11nEngine).to(device).eval().half()
detector = person_detector.PersonDetector('models/yolo11n.torchscript').to(device).eval().half()

model_nlf = multiperson_model_trt.MultipersonNLF(
    model_pytorch, detector, device=device).to(device).eval().half()

dummy_input = torch.randn(1, 3, 480, 640).cuda()

torch.onnx.export(
    model_nlf,
    dummy_input,
    "backbone.onnx",
    input_names=["input"],         # name of model input
    output_names=["output"],       # name of model output
    dynamo=True,
    do_constant_folding=True,
    dynamic_shapes=None,
)

multimodel = torch.jit.script(model_nlf)
#multimodel = torch.jit.trace(model_nlf, torch.randn(1, 3, 480, 640).to(device))

torch.jit.save(multimodel, 'models/end_model.torchscript')

