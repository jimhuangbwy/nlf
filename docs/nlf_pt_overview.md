# NLF (Neural Localizer Fields) — `nlf/pt/` Architecture

## The Algorithm

NLF treats the human body as a **continuous neural field** rather than a fixed set of keypoints. Given an image, you can query *any* canonical body point `p ∈ ℝ³` (a vertex on a canonical SMPL template, a joint, or an arbitrary surface/internal point) and get back its 2D + 3D location in the image.

**Core idea (hypernetwork + localizer head):**

1. **Backbone** (`backbones/builder.py`: EffNetV2 / ResNet / DINOv2) encodes image → feature map `F ∈ ℝ^{N×c×H×W}`.
2. **Weight field** `GPSField` (`models/field.py`) maps each canonical point `p` → a per-point tensor of conv weights + bias, shape `[(c+1)·(depth+2)]`. The field acts as a *hypernetwork*: different query points produce different 1×1 "detectors" to slide over the feature map.
3. **Localizer head** (`LocalizerHead` in `models/nlf_model.py`) applies those per-point weights to features via 1×1 conv/einsum → for each query point it produces:
   - an uncertainty channel,
   - a metric-XY channel (for weak/full-perspective reconstruction),
   - a 3D heatmap (2 + `depth` channels) decoded via soft-argmax to `(u, v, z_rel)`.
4. **Absolute reconstruction** (`ptu3d.reconstruct_absolute`) combines 2D image coords, relative depth, and intrinsics to recover absolute 3D in mm, weighted by uncertainty.

## Key Terms

- **Canonical space**: a rest-pose template body; every query point has a canonical 3D coord. File `canonical_loc_symmetric_init_866.npy` seeds 866 joints; `canonical_nodes3.npy` defines the full manifold.
- **GPS / LBO positional encoding** (`GPSNet`): inputs are encoded with **Learnable Fourier Features**, passed through an MLP that approximates eigenfunctions of the **Laplace–Beltrami Operator** on the body mesh (eigenvalues in `canonical_eigval3.npy`, scaled by `1/√λ`). This "GPS embedding" gives a smooth, geometry-aware coordinate system where nearby body points have similar codes — critical for the field to generalize across the surface.
- **Symmetry handling** (`NLFModel.canonical_locs`): only left-side joints are learned; right side is mirrored (`x → -x`). Hands are frozen via `canonical_delta_mask`.
- **Flip-aware inference** (`decode_features_multi_same_weights`): `dynamic_partition` / `dynamic_stitch` route horizontally-flipped images through mirrored weights so the same canonical point set can be queried consistently.
- **Heatmap decoding** (`models/util.py`, `ptu.soft_argmax`, `ptu.decode_heatmap`): 2.5D heatmap → `(u, v)` soft-argmax + marginalized `z`; `heatmap_to_metric` converts discretized bins to meters using `box_size_m` and `stride_test`.
- **Uncertainty**: a per-point scalar `softplus(Σ uncert_map · heatmap2d + bias)`; gates points in absolute reconstruction (`point_validity_mask = unc < 0.3`).

## Training (`models/nlf_trainer.py`)

- Jointly supports SMPL / SMPL-H / SMPL-X (male/female/neutral) bodies via `smplfitter`. Ground truth is generated *on GPU* by forwarding posed body models, then interpolating arbitrary query points using precomputed **Sibson (natural-neighbor) coordinates** over vertices+joints — this is what lets supervision be continuous, not just at keypoints.
- Loaders in `loading/` provide four supervision flavors: `keypoints3d`, `keypoints2d`, `parametric` (SMPL params → densely sampled surface/internal points), `densepose` (UV correspondences).
- `main.py` wires it together via `florch.TrainingJob`; `render_callback.py` visualizes with `poseviz`.

## Inference

- **Single-person**: `NLFModel.predict_multi_same_canonicals` / `predict_multi_same_weights` — the latter pre-computes weights once for a fixed query set (e.g., 768 SMPL verts) and reuses them across a batch via `F.conv2d`.
- **Multi-person** (`multiperson/`): person detector → crop / `warping.py` → NLF → `plausibility_check.py` filters.
- **Scripts** under `inference_scripts/` evaluate H36M, 3DPW, EMDB, MuPoTS, SSP-3D.

## Key Flags (`init.py`)

`--proc-side` (crop size), `--depth` (z bins), `--box-size-m` (metric volume), `--backbone-link-dim` (c), `--field-posenc-dim`, `--field-hidden-size` / `--field-hidden-layers`, `--stride-test`, `--weak-perspective`, `--uncert-bias`.
