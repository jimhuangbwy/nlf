import numpy as np
import smplfitter.pt

body_models_partial_vertex_subset = smplfitter.pt.BodyModel('smpl', num_betas=10, vertex_subset_size=1024).vertex_subset
print(body_models_partial_vertex_subset)
np.save("body_models_partial_vertex_subset.npy", np.array(body_models_partial_vertex_subset))