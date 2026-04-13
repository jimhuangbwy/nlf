# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neural Localizer Fields (NLF) — a method for continuous 3D human pose and shape estimation from images (NeurIPS 2024, Sárándi & Pons-Moll). The model predicts 3D body pose and shape by learning a neural field over canonical body coordinates, enabling continuous query of any point on the body surface.

## Environment Setup

- Python 3.10, managed via micromamba (conda env named `nlf`)
- Install: `envsubst < environment_comfortable_py10.yml > env_subst.yml && micromamba env create --name=nlf --file=env_subst.yml -y`
- Key external libraries authored by the same group (installed as editable from `$CODE_DIR`): `fleras`, `florch`, `simplepyutils`, `cameralib`, `boxlib`, `poseviz`, `smplfitter`, `rlemasklib`, `barecat3`, `posepile`
- `simplepyutils.FLAGS` is used throughout as a global flag/config object (argparse-based)

## Key Environment Variables

- `DATA_ROOT` — root path for datasets, annotations, and experiments (from `posepile.paths`)
- `PROJDIR` — project-specific data directory, defaults to `$DATA_ROOT/projects/localizerfields` (see `nlf/paths.py`)
- `CODE_DIR` — root for sibling editable-install packages

## Training (PyTorch)

```bash
python -m nlf.pt.main --train --logdir=<name> [flags...]
```

Entry point: `nlf/pt/main.py`. Key flags defined in `nlf/pt/init.py:get_parser()` include `--backbone`, `--proc-side`, `--training-steps`, `--batch-size`, `--load-path`, `--checkpoint-dir`, `--multi-gpu`, `--dtype` (default bfloat16). Logs go to `$DATA_ROOT/experiments/<logdir>`.

## Code Architecture

### Dual Framework: `nlf/pt/` (PyTorch) and `nlf/tf/` (TensorFlow)

Both implement the same NLF approach with parallel structure. PyTorch is the primary/active codebase.

### PyTorch (`nlf/pt/`)

- **`main.py`** — Training entry point. `LocalizerFieldJob` extends `florch.TrainingJob`, orchestrates data loading, training loop, and rendering.
- **`init.py`** — CLI argument parsing and initialization. All flags via `simplepyutils.FLAGS`.
- **`models/nlf_model.py`** — `NLFModel(nn.Module)`: backbone + `LocalizerHead`. Takes image + intrinsic matrix + canonical body points → 3D pose. Handles left/right symmetry of canonical locations.
- **`models/nlf_trainer.py`** — `NLFTrainer(florch.ModelTrainer)`: manages SMPL/SMPLH/SMPLX body models for ground truth generation during training.
- **`models/field.py`** — `GPSField` / `GPSNet`: the neural field that maps canonical 3D coordinates to feature space using Laplace-Beltrami operator eigenfunctions.
- **`backbones/builder.py`** — Factory for image backbones: EfficientNetV2, ResNet, MobileNet, DINOv2.
- **`loading/`** — Data loaders for different supervision types: `keypoints3d.py`, `keypoints2d.py`, `parametric.py` (SMPL params), `densepose.py`.
- **`multiperson/`** — Multi-person inference pipeline: person detection, warping, plausibility checks.
- **`ptu.py` / `ptu3d.py`** — PyTorch utility functions for 2D/3D geometry operations.

### Shared (`nlf/common/`)

- **`augmentation/`** — Image augmentation (color, background, border, appearance).
- **`improc.py`** — Image processing utilities.
- **`util.py` / `util3d.py`** — General and 3D geometry utilities.

### Inference

- `nlf/pt/inference_scripts/` — Benchmark prediction scripts (H36M, 3DPW, MuPoTS, EMDB, SSP-3D).
- `demo.ipynb` — Usage examples for running pretrained models.
- Pretrained models available under GitHub Releases (noncommercial research use).

## Formatting

- Black with `line-length = 99` and `skip-string-normalization = true` (see `pyproject.toml`)
