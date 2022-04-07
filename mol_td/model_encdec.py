from flax import linen as nn
from jax import numpy as jnp, nn as jnn
from tensorflow_probability.substrates.jax import distributions as tfd
from functools import partial
from .config import Config

leaky_relu = partial(nn.leaky_relu, negative_slope=.2)  # TF default


class MLPEncoder(nn.Module):
    
    cfg: Config

    @nn.compact
    def __call__(self, x):
        for n_hidden in self.cfg.enc_hidden: # the first contains the feature dimension
            x = jnn.relu(nn.Dense(n_hidden)(x))
        mean = nn.Dense(n_hidden)(x)
        std = jnn.softplus(nn.Dense(n_hidden)(x)) + self.cfg.latent_dist_min_std
        # sigma = jnp.sqrt(jnp.exp(nn.Dense(n_hidden)(x)))
        return dict(mean=mean,
                    std=std)

    
class MLPDecoder(nn.Module):
    
    cfg: Config

    @nn.compact
    def __call__(self, z):
        for n_hidden in self.cfg.dec_hidden[1:-1]: # the first contains the feature dimension
            z = jnn.relu(nn.Dense(n_hidden)(z))
        z = jnp.tanh(nn.Dense(self.cfg.dec_hidden[-1])(z))  # values in dataset restricted between 1 and -1
        return z