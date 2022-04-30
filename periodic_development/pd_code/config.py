from dataclasses import dataclass

import numpy as np
import jax.numpy as jnp
import jax
from datetime import datetime
from math import ceil


@dataclass
class Config:
    seed: int = 1

    n_particles: int = 10
    n_dim: int = 2
    side: float = 1.
    nv_features: int = 5
    graph_mlp_features: int = 16
    receivers_idx: int = 0
    senders_idx: int = 1