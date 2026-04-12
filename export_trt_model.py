import os
import sys

import torch
import torch_tensorrt

from nlf.pt.multiperson import person_detector, multiperson_model_trt
import nlf.pt.models.nlf_model_trt2 as pt_nlf_model_trt2


def export_trt_model(device):
    if getattr(sys, 'frozen', False):
        # Running in packaged EXE / MSIX
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(__file__)

    end_model_path = os.path.join(base_path, "models", "end_model.torchscript")

    if os.path.exists(end_model_path):
        print(f"[INFO] TensorRT engine already exists at: {end_model_path}")
        return torch_tensorrt.load(end_model_path).eval()

    print(f"[INFO] Engine not found. Compiling model...")

    backbone_path = os.path.join(base_path, "models", "backbone.torchscript")
    backbone_export_path = os.path.join(base_path, "models", "backbone.ts")
    weight_field_path = os.path.join(base_path, "models", "weight_field.torchscript")
    layer_path = os.path.join(base_path, "models", "layer.torchscript")
    end_model_path = os.path.join(base_path, "models", "end_model.torchscript")

    backbone = torch.jit.load(backbone_path).to(device).eval().half()

    if device == torch.device("cuda"):
        print('[INFO] compiling tensorRT model, wait...')
        inputs = [torch.randn((1, 3, 384, 384)).cuda().half()]  # define a list of representative inputs here
        backbone = torch_tensorrt.compile(backbone, ir="torchscript", inputs=inputs, enabled_precisions={torch.float16})
        torch.jit.save(backbone, backbone_export_path)
        backbone = torch.jit.load(backbone_export_path).to(device).eval().half()

    weight_field = torch.jit.load(weight_field_path).to(device).eval().half()
    layer = torch.jit.load(layer_path).to(device).eval().half()
    model_pytorch = pt_nlf_model_trt2.NLFModel(backbone, weight_field, layer).to(device).eval().half()

    detector = person_detector.PersonDetector('models/yolo11n.torchscript').to(device).eval().half()

    model_nlf = multiperson_model_trt.MultipersonNLF(
    model_pytorch, detector).to(device).eval().half()

    multimodel = torch.jit.script(model_nlf)

    torch.jit.save(multimodel, end_model_path)

    if os.path.exists(backbone_export_path):
        os.remove(backbone_export_path)

    print(f"[INFO] end_model export to: {end_model_path}")

    return multimodel