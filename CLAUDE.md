# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PinoFBT 2.0** — a real-time full body tracking application for VR, built on [Neural Localizer Fields (NLF)](https://arxiv.org/abs/2407.07532) (NeurIPS 2024, Sarandi & Pons-Moll). The app captures webcam video, runs 3D human pose estimation using TensorRT-accelerated models, and streams joint positions over OSC to VR applications (e.g. SteamVR via VMT).

This is a fork of the original NLF research codebase, adapted for real-time inference with a GUI and Microsoft Store distribution.

## Two Codebases in One Repo

### 1. Original NLF Training Code (`nlf/`)
- PyTorch (`nlf/pt/`) and TensorFlow (`nlf/tf/`) implementations of NLF
- Training entry point: `nlf/pt/main.py` with `florch.TrainingJob`
- Uses `simplepyutils.FLAGS` for configuration, `$DATA_ROOT`/`$PROJDIR` environment variables
- Not used at runtime by the tracking app

### 2. PinoFBT Tracking App (top-level files)
- **GUI**: `main_UI.py` (PySide6 + qasync + WinRT for MS Store)
- **Tracker**: `pino_tracker.py` — `PinoTracker` class with threaded capture/inference/display
- **Standalone**: `main.py` — same pipeline without GUI
- **TRT runtime**: `onnx_helper.py` (`TRTInference` nn.Module wrapping TensorRT engines)
- **Models**: `nlf/pt/models/nlf_model_trt.py` (NLFModel reassembled from TRT sub-engines), `nlf/pt/multiperson/multiperson_model_trt.py` (full multi-person pipeline), `nlf/pt/multiperson/person_detector_trt.py` (YOLO detector via TRT)
- **Trackers**: `nlf/pt/multiperson/nms_tracker.py` (IoU tracker), `nlf/pt/multiperson/single_person_tracker.py`

## Key Architecture

```
main_UI.py -> PinoTracker (pino_tracker.py)
  -> MultipersonNLF (multiperson_model_trt.py)
       -> PersonDetector (person_detector_trt.py) -> yolo12s.engine
       -> NLFModel (nlf_model_trt.py)
            -> backbone.engine, weight_field.engine, layer.engine
            -> LocalizerHead (heatmap decode + 3D reconstruction in PyTorch)
  -> OSC Client -> VMT (127.0.0.1:39570)
```

The model is decomposed into 4 TensorRT engines:
- `backbone.engine` — EfficientNetV2 feature extractor (1x3x384x384)
- `weight_field.engine` — GPS neural field (1048x3 -> conv weights)
- `layer.engine` — feature post-processing
- `yolo12s.engine` — YOLO person detector (1x3x640x640)

## Required Folder Structure

```
models/           # TensorRT .engine files (backbone, weight_field, layer, yolo12s)
nlf_data_files/   # canonical_eigval3.npy, canonical_nodes3.npy, canonical_verts/, canonical_joints/, body_models_partial_vertex_subset.npy
```

## Key Modifications from Original NLF

- `nlf/pt/models/field.py` — hardcoded dimensions (removed FLAGS/PROJDIR dependencies), paths -> `nlf_data_files/`
- `nlf/pt/multiperson/warping.py` — `n_pyramid_levels` reduced from 3 to 1 for speed
- Several files: `model` attribute renamed to `model_nlf` to avoid name collisions
- `nlf/pt/multiperson/person_detector.py` — added `.float()` on model output

## Export Pipeline

To go from pretrained TorchScript to TRT engines:
1. Extract sub-modules: `export_backbone.py`, `export_weight_field.py`, `export_layer.py`
2. Convert to ONNX or TorchScript-TRT: `export_backbone_onnx.py` / `export_backbone_trt.py`, etc.
3. Build TRT engines: `onnx2trt.py` or `export_yolo_trt.py`

## Running

- GUI: `python main_UI.py`
- Standalone: `python main.py` or `python pino_tracker.py`
- OSC target: `127.0.0.1:39570` (VMT protocol)

## Formatting

- Black with `line-length = 99` and `skip-string-normalization = true` (see `pyproject.toml`)

## Platform Notes

- Microsoft Store integration (`main_UI.py`, `ms_store/`) requires Windows + WinRT
- TensorRT engines are GPU-architecture-specific — must be rebuilt per target GPU
- Nuitka packaging notes in `conda_nlf_nuitka.txt` and `nuitka_log.txt`
