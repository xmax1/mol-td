from flax import linen as nn
from jax import numpy as jnp, nn as jnn
from tensorflow_probability.substrates.jax import distributions as tfd
from functools import partial
from .config import Config

leaky_relu = partial(nn.leaky_relu, negative_slope=.2)  # TF default


class MLPTransfer(nn.Module):
    
    cfg: Config

    def zero_state(self, leading_dims):
        # return dict(sample=jnp.zeros(leading_dims+(self.cfg.n_embed,)),
        return      dict( mean=jnp.zeros(leading_dims+(self.cfg.n_embed,)),
                    std=jnp.zeros(leading_dims+(self.cfg.n_embed,)))

    @nn.compact
    def __call__(self, z_t0, posteriors):
        
        z_t1 = MLPPrior(self.cfg)(z_t0)
        # posterior = MLPPosterior(inputs)
        
        return z_t1, (z_t0, )


class MLPPrior(nn.Module):

    cfg: Config

    @nn.compact
    def __call__(self, latent_zt0):
        
        for _ in range(self.cfg.n_transfer_layers):
            hidden = jnn.relu(nn.Dense(self.cfg.n_embed)(latent_zt0['mean']))
     
        mean = nn.Dense(self.cfg.n_embed)(hidden)
        std = jnn.softplus(nn.Dense(self.cfg.n_embed)(hidden)) + self.cfg.latent_dist_min_std

        # sample = tfd.Normal(mean, std).sample(seed=self.make_rng('sample'))
        # sample = tfd.MultivariateNormalDiag(mean, std).sample(seed=self.make_rng('sample'))

        return dict(mean=mean, 
                    std=std)





class SimpleTransfer(nn.Module):

    cfg: dict

    @nn.compact
    def __call__(self, z_t0, priors):

        transfer = MLPTransfer(self.cfg)

        scan = nn.scan(lambda transfer, carry, inputs: transfer(carry, inputs),
                       variable_broadcast='params',
                       split_rngs=dict(params=False, sample=True),
                       in_axes=1, out_axes=1)

        initial = jnp.zeros((inputs.shape[0], self.cfg['tr_hidden'])) if initial is None else initial
        priors, posteriors = scan(transfer, initial, (inputs, ))

        return priors, posteriors


class MLPPosterior(nn.Module):

    cfg: dict

    @nn.compact
    def __call__(self, inputs):

        mean = nn.Dense(self.cfg['enc_hidden'][-1])(inputs)  
        std = jnn.softplus(nn.Dense(self.cfg['enc_hidden'][-1]))(inputs)

        return dict(mean=mean, 
                    std=std)



