#import pickle

import torch
import torchvision
from torch._export.converter import TS2EPConverter
import torch._dynamo as torchdynamo
#import torch_tensorrt
import nlf.pt.models.nlf_model_trt as pt_nlf_model_trt

# class ModelWrapper(torch.nn.Module):
#     def __init__(self, script_module):
#         super().__init__()
#         self.model = script_module
#
#     def forward(self, x):
#         return self.model.detect_parametric_batched(x)  # Delegate to the original model's forward

#
#import nlf.pt.backbones.builder as backbone_builder
import nlf.pt.models.field as pt_field
# import nlf.pt.models.nlf_model as pt_nlf_model

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load full TorchScript model
#model_nlf = torch.jit.load("models/end_model.torchscript").eval()
# #model = torch.jit.load("models/nlf_l_multi_0.3.2.torchscript").eval()
# model_wrap = ModelWrapper(model)
#
# #print(model.crop_model.heatmap_head.weight_field.posenc_dim)
# #model_pytorch = model.crop_model.to(device).eval()
# backbone = model.crop_model.backbone.to(device).eval()
# heatmap_head = model.crop_model.heatmap_head.to(device).eval()
#
# #backbone = torch.jit.load("models/backbone_trt.ts").to(device).eval()
# # pred_mlp = model.crop_model.heatmap_head.weight_field.pred_mlp.to(device).eval()
# # print(pred_mlp)
# #weight_field = pt_field.build_field().to(device).eval()
#
# #layer = model.crop_model.heatmap_head.layer.float().to(device).eval()
# #heatmap_head = pt_nlf_model_trt.NLFModel(backbone, weight_field, layer)
# #model_pytorch = pt_nlf_model_trt.NLFModel(backbone, weight_field, layer).to(device).eval()
# model_pytorch = model.crop_model
# #model_pytorch = pt_nlf_model_trt.NLFModel(backbone, heatmap_head).to(device).eval()
#
# # backbone, normalizer, out_channels = backbone_builder.build_backbone()
# # weight_field = pt_field.build_field()
# # model_pytorch = pt_nlf_model.NLFModel(backbone, weight_field, normalizer, out_channels)
#
# #model_nlf = multiperson_model_trt.MultipersonNLF(
# #    model_pytorch, detector, skeleton_infos).to(device).eval()
#
# f = open('nlf_data_files/skeleton_types_huge8.pkl', 'rb')
# skeleton_infos = pickle.load(f)
#
# detector = model.detector.to(device).eval()
# #detector = person_detector.PersonDetector('yolo12n.torchscript')
#
# model_nlf = multiperson_model_trt.MultipersonNLF(
#     model_pytorch, detector, skeleton_infos).to(device).eval()

model = torch.jit.load("models/nlf_l_multi_0.3.2.torchscript").to(device).eval()

#crop_model =model.crop_model.to(device).eval()

# backbone = model.crop_model.backbone.to(device).eval()
weight_field = model.crop_model.heatmap_head.weight_field.to(device).eval()
#layer = model.crop_model.heatmap_head.layer.to(device).eval().float()
# model_pytorch = pt_nlf_model_trt2.NLFModel(backbone, weight_field, layer).to(device).eval()

#detector = person_detector.PersonDetector('models/yolo11n.torchscript').to(device).eval()
# detector = ScriptedDetectorWrapper(model.detector).to(device).eval()

# f = open('nlf_data_files/skeleton_types_huge8.pkl', 'rb')
# skeleton_infos = pickle.load(f)

# model_nlf = multiperson_model_onnx.MultipersonNLFExportable(
#     model_pytorch, detector).to(device).eval()

#define input
dummy_input = torch.rand(1048, 3).cuda()  # for weight_field
#dummy_input = torch.rand(1, 1280, 12, 12).cuda()  # for layer

#model_nlf = torch.export.export(model_nlf, (dummy_input,))

#model_nlf = torch.jit.trace(model_nlf, (dummy_input,))
#model_nlf = torch.jit.script(model_nlf)
print('okkk')
#model_nlf = TS2EPConverter(model_nlf, (dummy_input,), {}).convert()

#weight_field = TS2EPConverter(weight_field, (dummy_input,), {}).convert()


# 3. Export to ONNX
# torch.onnx.export(
#     weight_field,
#     dummy_input,
#     "models/weight_field.onnx",
#     input_names=["input"],         # name of model input
#     output_names=["output"],       # name of model output
#     opset_version=20,               # ONNX opset (12–13 recommended for TensorRT)
#     dynamo=False
# )