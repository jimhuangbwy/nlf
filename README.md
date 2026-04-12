# PinoFBT 2.0 — Real-Time Full Body Tracking with NLF

A real-time camera-based full body tracking application built on top of [Neural Localizer Fields (NLF)](https://arxiv.org/abs/2407.07532). Captures webcam video, runs multi-person 3D pose estimation via TensorRT-accelerated models, and streams joint positions over OSC (e.g. to VRChat via VMT).

Based on NLF (NeurIPS 2024, Sarandi & Pons-Moll) — see [original project page](https://istvansarandi.com/nlf).

## Features

- **TensorRT inference pipeline** — NLF backbone, weight field, and layer exported as TensorRT `.engine` files for low-latency GPU inference
- **YOLO-based person detection** — YOLOv8/v11/v12 person detector also running as a TensorRT engine
- **Multi-threaded capture/inference/display** — separate threads for camera capture, model inference, and rendering to avoid pipeline stalls
- **OSC output** — sends 3D joint positions via OSC to VMT (Virtual Motion Tracker) for VR full body tracking
- **Joint extrapolation** — linear velocity extrapolation between inference frames for smoother tracking
- **PySide6 GUI** (`main_UI.py`) — tabbed interface with Start/Stop tracking and Microsoft Store subscription management
- **Microsoft Store integration** — WinRT-based in-app purchase flow for subscription products (Windows only)

## Architecture Overview

```
main_UI.py (PySide6 GUI)
  |
  +-- PinoTracker (pino_tracker.py)
        |
        +-- MultipersonNLF (nlf/pt/multiperson/multiperson_model_trt.py)
        |     |
        |     +-- PersonDetector (nlf/pt/multiperson/person_detector_trt.py)
        |     |     \-- TRTInference (onnx_helper.py) -> yolo12s.engine
        |     |
        |     +-- NLFModel (nlf/pt/models/nlf_model_trt.py)
        |     |     +-- TRTInference -> backbone.engine
        |     |     +-- TRTInference -> weight_field.engine
        |     |     +-- TRTInference -> layer.engine
        |     |     \-- LocalizerHead (heatmap decoding, 3D reconstruction)
        |     |
        |     +-- warping.py (image crop/warp for detected persons)
        |     \-- plausibility_check.py (pose filtering)
        |
        +-- OSC Client (python-osc) -> VMT (127.0.0.1:39570)
        \-- OpenCV display (bounding boxes, FPS overlay)
```

### Key Components

| File | Purpose |
|------|---------|
| `main_UI.py` | PySide6 GUI with Home (tracking) and Subscription tabs |
| `pino_tracker.py` | `PinoTracker` class — loads TRT models, runs threaded capture/inference/display pipeline, sends OSC |
| `main.py` | Standalone threaded pipeline (no GUI, direct `cv2.imshow`) |
| `onnx_helper.py` | `TRTInference(nn.Module)` — generic TensorRT engine loader and runner |
| `nlf/pt/models/nlf_model_trt.py` | `NLFModel` — reassembled NLF model using TRT sub-engines (proc_side=384) |
| `nlf/pt/models/nlf_model_trt_s.py` | `NLFModel` small variant (proc_side=256) |
| `nlf/pt/multiperson/multiperson_model_trt.py` | `MultipersonNLF` — orchestrates detection, cropping, pose estimation, 3D reconstruction |
| `nlf/pt/multiperson/person_detector_trt.py` | `PersonDetector` — YOLO-based detector using TRT engine |
| `nlf/pt/multiperson/nms_tracker.py` | `IoUTracker` — simple IoU-based multi-frame tracking |
| `nlf/pt/multiperson/single_person_tracker.py` | `SinglePersonTracker` — single-target tracker with IoU+distance+size scoring |
| `wrapper.py` | `NLFWrapper` — wraps TorchScript model for TensorRT compilation |
| `ms_store/main.py` | Standalone Microsoft Store purchase test (tkinter + WinRT) |

### Export Scripts

Scripts for decomposing the pretrained NLF TorchScript model into TRT engines:

| Script | Output |
|--------|--------|
| `export_backbone_trt.py` | `models/backbone.ts` (TorchScript+TRT) |
| `export_backbone_onnx.py` | `models/backbone.onnx` |
| `export_weight_field_trt.py` | `models/weight_field.ts` |
| `export_weight_field.py` | Extracts weight_field sub-module |
| `export_layer.py` | Extracts layer sub-module |
| `export_layer_trt.py` | `models/layer.ts` (TorchScript+TRT) |
| `export_onnx.py` | ONNX export of sub-components |
| `export_yolo_trt.py` | Builds YOLO `.engine` from ONNX |
| `export_yolo_onnx.py` | Exports YOLO to ONNX |
| `export_trt_model.py` | End-to-end TRT compilation of full model |
| `onnx2trt.py` | Generic ONNX to TensorRT engine builder |

## Prerequisites

- **OS**: Windows 10/11 (required for Microsoft Store integration; tracking works on Linux without Store features)
- **GPU**: NVIDIA GPU with CUDA support
- **CUDA**: 12.x
- **TensorRT**: 10.x
- **Python**: 3.10+

## Required Folder Structure

```
nlf-fork/
  models/
    nlf_l_multi_0.3.2.torchscript   # Original pretrained NLF model (from GitHub Releases)
    backbone.engine                   # TRT engine (exported)
    weight_field.engine               # TRT engine (exported)
    layer.engine                      # TRT engine (exported)
    yolo12s.engine                    # YOLO person detector TRT engine
  nlf_data_files/
    canonical_eigval3.npy             # Eigenvalues for GPS field
    canonical_nodes3.npy              # Node positions for GPS field
    canonical_verts/
      smpl.npy                        # Canonical SMPL vertices
    canonical_joints/
      smpl.npy                        # Canonical SMPL joints
    body_models_partial_vertex_subset.npy  # 1024-vertex subset indices
  nlf/                                # NLF library code
  main_UI.py                          # GUI entry point
  pino_tracker.py                     # Tracking logic
  onnx_helper.py                      # TRT inference helper
  requirements.txt                    # Python dependencies
```

## Installation

1. **Install CUDA and TensorRT** following NVIDIA's documentation for your GPU.

2. **Create a Python environment**:
   ```bash
   conda create -n pinofbt python=3.10
   conda activate pinofbt
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the pretrained NLF model** from the original NLF GitHub Releases and place it in `models/`.

5. **Prepare `nlf_data_files/`**: Extract the canonical body data files required by the model. These include eigenvalue/eigenvector files for the GPS field and canonical SMPL vertex/joint positions. They can be extracted from the original NLF data or generated using `body_models_partial_vertex_subset.py`.

## Step-by-Step: Export TRT Engines

Starting from the pretrained TorchScript model, export the sub-components as TensorRT engines:

1. **Extract sub-modules** from the TorchScript model:
   ```bash
   python export_weight_field.py    # saves models/weight_field.torchscript
   python export_layer.py           # saves models/layer.torchscript
   python export_backbone.py        # saves models/backbone.torchscript
   ```

2. **Export YOLO detector**:
   ```bash
   python export_yolo_onnx.py       # exports YOLO to ONNX
   python export_yolo_trt.py        # builds yolo12s.engine from ONNX
   ```

3. **Build TensorRT engines** (ONNX path):
   ```bash
   python export_backbone_onnx.py   # backbone.onnx
   python onnx2trt.py               # backbone.onnx -> backbone.engine
   ```
   Or via TorchScript-TRT path:
   ```bash
   python export_backbone_trt.py    # backbone.ts (TorchScript+TRT)
   python export_weight_field_trt.py
   python export_layer_trt.py
   ```

4. **Verify** all `.engine` files exist in `models/`:
   ```
   models/backbone.engine
   models/weight_field.engine
   models/layer.engine
   models/yolo12s.engine
   ```

## Running the Application

### Option A: GUI Mode (recommended)

```bash
python main_UI.py
```

1. The app opens with a "Home" tab showing **START TRACKING**.
2. Click **START TRACKING** — models load (first time takes ~30 seconds for TRT engine warm-up).
3. An OpenCV window opens showing the webcam feed with:
   - Bounding box around detected person (green)
   - FPS counter (top-left)
4. Joint positions are streamed via OSC to `127.0.0.1:39570` (VMT protocol).
5. Click **STOP TRACKING** or press `Q` in the OpenCV window to stop.
6. The **Subscription** tab connects to the Microsoft Store for in-app purchases (Windows only).

### Option B: Standalone (no GUI)

```bash
python main.py
```

Runs the same threaded pipeline with direct OpenCV display. Press `Q` to quit.

### Option C: PinoTracker as a module

```bash
python pino_tracker.py
```

### OSC Configuration

By default, joint data is sent to:
- **IP**: `127.0.0.1`
- **Port**: `39570`
- **Protocol**: VMT (Virtual Motion Tracker) for SteamVR

OSC messages:
- `/VMT/SetRoomMatrix` — room calibration matrix
- `/VMT/Follow/Unity` — per-joint position and rotation data (index, enable, timeoffset, pos_xyz, rot_xyzw, serial)

## Changes from Original NLF

### Modified Files

- `nlf/pt/models/field.py` — Hardcoded dimensions (removed `FLAGS` dependency), paths changed to `nlf_data_files/`
- `nlf/pt/ptu.py` — Removed unused `linspace` comment block
- `nlf/pt/multiperson/warping.py` — Reduced pyramid levels from 3 to 1 for speed
- `nlf/pt/multiperson/person_detector.py` — Added `.float()` cast on model output
- `nlf/pt/main.py`, `nlf/pt/render_callback.py` — Renamed `model` attribute to `model_nlf`
- `nlf/tf/main.py`, `nlf/tf/render_callback.py`, `nlf/tf/backbones/efficientnet/effnetv2_model.py` — Same `model` -> `model_nlf` rename

### New TRT Inference Architecture

The original NLF uses a monolithic TorchScript model. This fork decomposes it into four TensorRT engines:

1. **backbone.engine** — EfficientNetV2 image feature extractor (input: 1x3x384x384)
2. **weight_field.engine** — GPS neural field mapping canonical 3D coordinates to conv weights (input: 1048x3)
3. **layer.engine** — Feature post-processing layer
4. **yolo12s.engine** — YOLO person detector (input: 1x3x640x640)

These are reassembled in `nlf_model_trt.py` with the heatmap decoding and 3D reconstruction logic kept in PyTorch.

## Original NLF

Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation
* [NeurIPS'24 paper](https://arxiv.org/abs/2407.07532) by Istvan Sarandi and Gerard Pons-Moll
* [Project page](https://istvansarandi.com/nlf)

Pretrained models available under GitHub Releases (noncommercial research use).

## BibTeX
```
@article{sarandi2024nlf,
    title     = {Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation},
    author    = {S\'ar\'andi, Istv\'an and Pons-Moll, Gerard},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year      = {2024}
}
```
