
import jax
from jax import numpy as jnp, random as rnd, jit
from pd_code.config import Config
from jax_md.partition import neighbor_list, NeighborListFormat
from jax_md import space

import numpy as onp

cfg = Config()
key = rnd.PRNGKey(cfg.seed)

positions = rnd.uniform(key, (cfg.n_particles, cfg.n_dim))

displacement_fn, shift_fn = space.periodic(cfg.side, wrapped=True)

# Dense: (N, n_max_neighbors_per_atom), Sparse (2, n_max_neighbors): Ordered Sparse: Sparse but half (no mirrors)
neighbor_fn = neighbor_list(displacement_fn, box_size=1., r_cutoff=0.5, dr_threshold=0.01, format=NeighborListFormat.Sparse)
nbrs = neighbor_fn.allocate(positions)

# Confirm the neighbors list works, 
# NB // writing your own displacement function doesn't work because t=0 is input as a variable. Need to include **unused kwargs to inputs

displace
