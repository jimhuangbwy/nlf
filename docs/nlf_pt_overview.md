# NLF (Neural Localizer Fields) — `nlf/pt/` Architecture

## The Algorithm

NLF treats the human body as a **continuous neural field** rather than a fixed set of keypoints. Given an image, you can query *any* canonical body point `p ∈ ℝ³` (a vertex on a canonical SMPL template, a joint, or an arbitrary surface/internal point) and get back its 2D + 3D location in the image.

Traditional 3D pose estimators commit to a discrete output vocabulary at training time — typically a fixed skeleton of 17 or 24 joints, or the ~6890 vertices of SMPL. NLF instead decouples *what is asked* from *what is trained*: the model conditions on the query location at inference time, so the same trained network can produce a skeleton in one call, a dense surface mesh in the next, and a sparse set of anatomical landmarks in a third, without retraining or per-topology heads. This is why the paper calls the output a "field" — it is defined everywhere on (and inside) the canonical body, and evaluation is a lookup.

**Core idea (hypernetwork + localizer head):**

1. **Backbone** (`backbones/builder.py`: EffNetV2 / ResNet / DINOv2) encodes image → feature map `F ∈ ℝ^{N×c×H×W}`.
2. **Weight field** `GPSField` (`models/field.py`) maps each canonical point `p` → a per-point tensor of conv weights + bias, shape `[(c+1)·(depth+2)]`. The field acts as a *hypernetwork*: different query points produce different 1×1 "detectors" to slide over the feature map.
3. **Localizer head** (`LocalizerHead` in `models/nlf_model.py`) applies those per-point weights to features via 1×1 conv/einsum → for each query point it produces:
   - an uncertainty channel,
   - a metric-XY channel (for weak/full-perspective reconstruction),
   - a 3D heatmap (2 + `depth` channels) decoded via soft-argmax to `(u, v, z_rel)`.
4. **Absolute reconstruction** (`ptu3d.reconstruct_absolute`) combines 2D image coords, relative depth, and intrinsics to recover absolute 3D in mm, weighted by uncertainty.

The hypernetwork pattern is what makes the approach scalable: instead of the backbone emitting `K·(depth+2)` channels for `K` output points (which would force a fixed `K` and blow up memory as `K` grows), the backbone emits a compact feature map of `c` channels, and *per-query* convolution kernels are generated on demand. At inference you can choose `K` adaptively — a 256-joint skeleton for speed, a 6890-vertex mesh for rendering — and the only cost is the field MLP forward pass plus a batched 1×1 conv. The uncertainty channel is essential because many canonical queries (e.g., occluded internal torso points, or surface points behind the person) have no clean image evidence; weighting by `1/σ²` during absolute-depth reconstruction lets confident points dominate and ignores the rest without hand-crafted masks.

## Key Terms

- **Canonical space**: a rest-pose template body; every query point has a canonical 3D coord. File `canonical_loc_symmetric_init_866.npy` seeds 866 joints; `canonical_nodes3.npy` defines the full manifold.
- **GPS / LBO positional encoding** (`GPSNet`): inputs are encoded with **Learnable Fourier Features**, passed through an MLP that approximates eigenfunctions of the **Laplace–Beltrami Operator** on the body mesh (eigenvalues in `canonical_eigval3.npy`, scaled by `1/√λ`). This "GPS embedding" gives a smooth, geometry-aware coordinate system where nearby body points have similar codes — critical for the field to generalize across the surface.
- **Symmetry handling** (`NLFModel.canonical_locs`): only left-side joints are learned; right side is mirrored (`x → -x`). Hands are frozen via `canonical_delta_mask`.
- **Flip-aware inference** (`decode_features_multi_same_weights`): `dynamic_partition` / `dynamic_stitch` route horizontally-flipped images through mirrored weights so the same canonical point set can be queried consistently.
- **Heatmap decoding** (`models/util.py`, `ptu.soft_argmax`, `ptu.decode_heatmap`): 2.5D heatmap → `(u, v)` soft-argmax + marginalized `z`; `heatmap_to_metric` converts discretized bins to meters using `box_size_m` and `stride_test`.
- **Uncertainty**: a per-point scalar `softplus(Σ uncert_map · heatmap2d + bias)`; gates points in absolute reconstruction (`point_validity_mask = unc < 0.3`).

The LBO eigenfunctions deserve emphasis: they are the geometric analogue of a Fourier basis, but defined *on the mesh surface* rather than on Euclidean space. Low-frequency eigenfunctions vary smoothly across the body (e.g., roughly separating head from feet), higher frequencies encode fine detail (fingers, facial features). Encoding canonical coordinates in this basis means the hypernetwork sees each query as a point on an intrinsic body manifold rather than as raw `(x, y, z)` — two points close on the surface but far in Euclidean distance (e.g., two sides of a folded arm) get distinct codes, and two points close on the surface stay close in code. Combined with the learnable Fourier features on the raw `(x, y, z)`, the field gets both extrinsic and intrinsic positional signals. The symmetry trick halves the number of learnable canonical anchors and implicitly regularizes the field to produce left/right consistent predictions, while the flip-aware inference path lets test-time horizontal flipping be used as free augmentation without breaking canonical-ID correspondence.

## Training (`models/nlf_trainer.py`)

- Jointly supports SMPL / SMPL-H / SMPL-X (male/female/neutral) bodies via `smplfitter`. Ground truth is generated *on GPU* by forwarding posed body models, then interpolating arbitrary query points using precomputed **Sibson (natural-neighbor) coordinates** over vertices+joints — this is what lets supervision be continuous, not just at keypoints.
- Loaders in `loading/` provide four supervision flavors: `keypoints3d`, `keypoints2d`, `parametric` (SMPL params → densely sampled surface/internal points), `densepose` (UV correspondences).
- `main.py` wires it together via `florch.TrainingJob`; `render_callback.py` visualizes with `poseviz`.

Sibson natural-neighbor interpolation is the mechanism that turns sparse body-model outputs into a dense supervisory signal. Given a query canonical point `p`, it finds the set of template vertices/joints whose Voronoi cells would be "stolen from" if `p` were inserted, and weights each by the stolen area. Because the weights are precomputed against the canonical template (pose-independent), training-time cost is just a gather + weighted sum over the posed vertices — cheap enough to sample thousands of random query points per image per step. This means supervision can actually match the continuous model: every gradient step trains the field at a fresh, randomly sampled set of canonical points rather than at a fixed grid, which discourages the field from overfitting to specific query locations. The multi-body-model support (SMPL/SMPL-H/SMPL-X, gendered) also matters because datasets come with different annotation formats; unifying them through GPU body-model evaluation avoids a preprocessing step that would otherwise be the throughput bottleneck.

## Inference

- **Single-person**: `NLFModel.predict_multi_same_canonicals` / `predict_multi_same_weights` — the latter pre-computes weights once for a fixed query set (e.g., 768 SMPL verts) and reuses them across a batch via `F.conv2d`.
- **Multi-person** (`multiperson/`): person detector → crop / `warping.py` → NLF → `plausibility_check.py` filters.
- **Scripts** under `inference_scripts/` evaluate H36M, 3DPW, EMDB, MuPoTS, SSP-3D.

The `same_weights` path is the deployment workhorse: when the set of queried canonical points is fixed across a batch (the common case — you want the same skeleton/mesh topology for every frame of a video), the hypernetwork only runs once, producing a stack of 1×1 conv kernels that are then applied to every image's feature map via a single batched `F.conv2d`. This amortizes the field MLP cost to near zero per additional frame and is what makes real-time multi-person inference feasible. The multi-person pipeline mirrors common top-down pose pipelines (detect → crop-and-warp → per-person prediction → stitch back), but the plausibility check is NLF-specific: because the model will predict *something* for any crop, a post-hoc filter using body-shape prior, joint-angle limits, and uncertainty is needed to reject false-positive detections and hallucinated poses.

## Key Flags (`init.py`)

`--proc-side` (crop size), `--depth` (z bins), `--box-size-m` (metric volume), `--backbone-link-dim` (c), `--field-posenc-dim`, `--field-hidden-size` / `--field-hidden-layers`, `--stride-test`, `--weak-perspective`, `--uncert-bias`.

These flags trade off capacity, resolution, and metric accuracy. `--proc-side` and `--stride-test` jointly determine the spatial grid over which soft-argmax localizes 2D coordinates — higher proc-side gives sub-pixel 2D accuracy but quadratic compute; `--depth` controls the z-resolution of the 2.5D heatmap and interacts with `--box-size-m` to set the per-bin metric size (coarser bins = larger tolerated depth range but worse precision). `--backbone-link-dim` (`c`) is the channel width of the bridge between backbone and hypernetwork and directly controls the size of each generated kernel `(c+1)·(depth+2)`, so it is the main capacity knob for the field output. `--field-hidden-size` / `--field-hidden-layers` size the hypernetwork MLP itself, affecting how sharply the field can vary across canonical space. `--weak-perspective` switches the reconstruction head between full-perspective (uses intrinsics) and weak-perspective (scale + 2D translation), useful when intrinsics are unknown. `--uncert-bias` shifts the default uncertainty floor — tuning it affects how aggressively the absolute-depth fusion rejects noisy points.
