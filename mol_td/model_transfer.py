from flax import linen as nn
from jax import numpy as jnp, nn as jnn
from tensorflow_probability.substrates.jax import distributions as tfd
from functools import partial
from .config import Config


leaky_relu = partial(nn.leaky_relu, negative_slope=.2)  # TF default


class MLPTransfer(nn.Module):
    
    cfg: Config

    def zero_state(self, leading_dims):
        mean=jnp.zeros(leading_dims+(self.cfg.n_embed,))
        std=jnp.ones(leading_dims+(self.cfg.n_embed,))
        mean = tfd.MultivariateNormalDiag(mean, std).sample(seed=self.make_rng('sample'))
        # return dict(sample=jnp.zeros(leading_dims+(self.cfg.n_embed,)),
        return dict(mean=mean,
                    std=std)


    @nn.compact
    def __call__(self, z_t0, posteriors):
        
        z_t1 = MLPPrior(self.cfg)(z_t0['mean'])
        
        return z_t1, (z_t0, )


class MLPPrior(nn.Module):

    cfg: Config

    @nn.compact
    def __call__(self, hidden):
        
        for _ in range(self.cfg.n_transfer_layers):
            hidden = jnn.relu(nn.Dense(self.cfg.n_embed)(hidden))
     
        mean = nn.Dense(self.cfg.n_embed)(hidden)
        std = jnn.softplus(nn.Dense(self.cfg.n_embed)(hidden + 0.54)) + self.cfg.latent_dist_min_std

        # sample = tfd.Normal(mean, std).sample(seed=self.make_rng('sample'))
        # sample = tfd.MultivariateNormalDiag(mean, std).sample(seed=self.make_rng('sample'))

        return dict(mean=mean, 
                    std=std)


