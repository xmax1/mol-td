from socket import SO_PRIORITY
from flax import linen as nn
from jax import numpy as jnp, nn as jnn, random as rnd
from tensorflow_probability.substrates.jax import distributions as tfd
from functools import partial
from .config import Config
from .model_base import activations


class GRUCell(nn.Module):

    cfg: Config

    @nn.compact
    def __call__(self, h0, z_t0):

        # for _ in range(self.cfg.n_transfer_layers):
        z_t0 = activations[self.cfg.latent_activation](nn.Dense(self.cfg.n_embed)(z_t0))

        h1, cell_out = nn.GRUCell()(h0, z_t0)  # (carry, outputs) = f(carry, inputs) # https://flax.readthedocs.io/en/latest/_autosummary/flax.linen.GRUCell.html

        cell_out = activations[self.cfg.latent_activation](nn.Dense(self.cfg.n_embed)(cell_out))

        mean = nn.Dense(self.cfg.n_embed)(cell_out)
        std = nn.softplus(nn.Dense(self.cfg.n_embed)(cell_out + 0.54)) + self.cfg.latent_dist_min_std
        dist = tfd.Normal(mean, std)
        z_t1 = dist.sample(seed=self.make_rng('sample'))

        return dict(h=h1,
                    cell_out=cell_out,
                    z=z_t1,
                    mean=mean,
                    std=std,
                    context=jnp.concatenate([cell_out, z_t1], axis=-1)) 


def reparametrisation(mean, std):
    normal = tfd.Normal(jnp.zeros_like(mean), jnp.ones_like(mean))
    eps = normal.sample(seed=self.make_rng('sample'))
    z_t1 = mean + std * eps
    return z_t1
    


class MLPPosterior(nn.Module):

    cfg: Config

    @nn.compact
    def __call__(self, prior, embedding):

        embedding = activations[self.cfg.latent_activation](nn.Dense(self.cfg.n_embed)(embedding))
        
        mean = nn.Dense(self.cfg.n_embed)(embedding)
        std = jnn.softplus(nn.Dense(self.cfg.n_embed)(embedding + 0.54)) + self.cfg.latent_dist_min_std
        dist = tfd.Normal(mean, std)
        z_t1 = dist.sample(seed=self.make_rng('sample'))
        return dict(z=z,
                    mean=mean,
                    std=std,
                    context=jnp.concatenate([prior['cell_out'], z_t1], axis=-1))


cells = {'LSTM': nn.LSTMCell,
         'GRU': nn.GRUCell
}


class Distribution(nn.Module):

    cfg: Config

    @nn.compact
    def __call__(self, s):

        for _ in range(self.cfg.n_transfer_layers):
            s = activations[self.cfg.latent_activation](nn.Dense(self.cfg.n_embed)(s))

        mean = nn.Dense(self.cfg.n_embed)(s)
        std = jnn.softplus(nn.Dense(self.cfg.n_embed)(s + 0.54)) + self.cfg.latent_dist_min_std
        dist = tfd.Normal(mean, std)
        z_t1 = dist.sample(seed=self.make_rng('sample'))

        return dict(mean=mean,
                    std=std,
                    z=z_t1,
        )


class MLPTransfer(nn.Module):
    
    cfg: Config

    def setup(self):

        self.transfer = cells[self.cfg.transfer_fn](self.cfg)
        self.posterior = Distribution(self.cfg)
        self.prior = Distribution(self.cfg)

    def zero_state(self, leading_dims):
        mean=jnp.zeros(leading_dims+(self.cfg.n_embed,))
        std=jnp.ones(leading_dims+(self.cfg.n_embed,))
        dist = tfd.Normal(mean, std)
        cell_out = dist.sample(seed=self.make_rng('sample'))
        z_prior = dist.sample(seed=self.make_rng('sample'))
        z_posterior = dist.sample(seed=self.make_rng('sample'))
        
        if self.cfg.transfer_fn == 'MLP':
            state =  dict(z=z_prior, mean=mean, std=std)
        elif self.cfg.transfer_fn == 'LSTM':
            h = nn.LSTMCell.initialize_carry(rnd.PRNGKey(self.cfg.seed), leading_dims, self.cfg.n_embed)
            # default is a matrix of zeros. h is a tuple
            state =  dict(carry=h, z=mean)
        elif self.cfg.transfer_fn == 'GRU':
            h = nn.GRUCell.initialize_carry(rnd.PRNGKey(self.cfg.seed), leading_dims, self.cfg.n_embed)
            # default is a matrix of zeros. h is an array
            state =  dict(carry=mean, z=mean)

        return state

    @nn.compact
    def __call__(self, prev_state, inputs, training):
        embedding, context = inputs

        print('latent')
        print(embedding.shape, context.shape, prev_state['z'].shape, prev_state['carry'].shape)
        
        if self.cfg.n_latent > 1:
            s_t0 = jnp.concatenate([prev_state['z'], context], axis=-1)
        else:
            s_t0 = prev_state['z']
        
        s_t0 = activations[self.cfg.latent_activation](nn.Dense(self.cfg.n_embed)(s_t0))
        
        if self.cfg.transfer_fn == 'GRU':
            carry, cell_out = nn.GRUCell()(prev_state['carry'], s_t0)  # both gru outputs are the same https://blog.floydhub.com/gru-with-pytorch/
        elif self.cfg.transfer_fn == 'LSTM':
            carry, cell_out = nn.LSTMCell()(prev_state['carry'], s_t0)
        
        # s_t0 = activations[self.cfg.latent_activation](nn.Dense(self.cfg.n_embed)(s_t0))

        prior = self.prior(cell_out)

        s_posterior_t0 = jnp.concatenate([cell_out, embedding], axis=-1)

        # s_posterior_t0 = activations[self.cfg.latent_activation](nn.Dense(self.cfg.n_embed)(s_posterior_t0))
        # s_posterior_t0 = activations[self.cfg.latent_activation](nn.Dense(self.cfg.n_embed)(s_posterior_t0))
        
        posterior = self.posterior(s_posterior_t0) if training else prior

        posterior['context'] = jnp.concatenate([posterior['z'], cell_out], axis=-1)
        
        next_state = dict(z=posterior['z'],
                          carry=carry,
        )
        
        return next_state, (prior, posterior)

import flax.linen.initializers
import flax.linen




