from socket import SO_PRIORITY
from flax import linen as nn
from jax import numpy as jnp, nn as jnn, random as rnd
from tensorflow_probability.substrates.jax import distributions as tfd
from functools import partial
from .config import Config
from .model_base import activations


class MLPTransfer(nn.Module):
    
    cfg: Config

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
            state =  dict(h_prior=h, z_prior=z_prior, z_posterior=z_posterior)
        elif self.cfg.transfer_fn == 'GRU':
            h = nn.GRUCell.initialize_carry(rnd.PRNGKey(self.cfg.seed), leading_dims, self.cfg.n_embed)
            state =  dict(h_prior=h, z_prior=z_prior, z_posterior=z_posterior)
        
        
        return state

    @nn.compact
    def __call__(self, prev_state, inputs, training):
        embedding, context = inputs
        
        if self.cfg.n_latent > 1:
            s_prior = jnp.concatenate([prev_state['z_prior'], context['z_prior']], axis=-1)
            s_posterior = jnp.concatenate([embedding, prev_state['z_posterior'], context['z_posterior']], axis=-1)
        else:
            s_prior = prev_state['z_prior']
            s_posterior = jnp.concatenate([embedding, prev_state['z_posterior']], axis=-1)
            
        if self.cfg.transfer_fn == 'MLP':
            prior = MLPPrior(self.cfg)(s_prior)
        elif self.cfg.transfer_fn == 'LSTM':
            prior = LSTMPrior(self.cfg)(prev_state['h_prior'], s_prior)
        elif self.cfg.transfer_fn == 'GRU':
            prior = GRUPrior(self.cfg)(prev_state['h_prior'], s_prior)

        posterior = MLPPosterior(self.cfg)(s_posterior) if training else prior
        
        next_state = dict(z_prior=prior['z'],
                          h_prior=prior['h'],
                          z_posterior=posterior['z']
        )
        
        return next_state, (prior, posterior)

class GRUPrior(nn.Module):

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
                    std=std) 


def reparametrisation(mean, std):
    normal = tfd.Normal(jnp.zeros_like(mean), jnp.ones_like(mean))
    eps = normal.sample(seed=self.make_rng('sample'))
    z_t1 = mean + std * eps
    return z_t1
    

class LSTMPrior(nn.Module):

    cfg: Config

    @nn.compact
    def __call__(self, h0, z_t0):
        
        z_t0 = activations[self.cfg.latent_activation](nn.Dense(self.cfg.n_embed)(z_t0))

        h1, cell_out = nn.LSTMCell()(h0, z_t0)

        cell_out = activations[self.cfg.latent_activation](nn.Dense(self.cfg.n_embed)(cell_out))

        mean = nn.Dense(self.cfg.n_embed)(cell_out)
        std = jnn.softplus(nn.Dense(self.cfg.n_embed)(cell_out + 0.54)) + self.cfg.latent_dist_min_std
        dist = tfd.MultivariateNormalDiag(mean, std)
        z_t1 = dist.sample(seed=self.make_rng('sample'))

        return dict(h=h1,
                    cell_out=cell_out,
                    z=z_t1,
                    mean=mean,
                    std=std) 


class MLPPosterior(nn.Module):

    cfg: Config

    @nn.compact
    def __call__(self, embedding):

        embedding = activations[self.cfg.latent_activation](nn.Dense(self.cfg.n_embed)(embedding))
        
        mean = nn.Dense(self.cfg.n_embed)(embedding)
        std = jnn.softplus(nn.Dense(self.cfg.n_embed)(embedding + 0.54)) + self.cfg.latent_dist_min_std
        dist = tfd.Normal(mean, std)
        z = dist.sample(seed=self.make_rng('sample'))
        return dict(z=z,
                    mean=mean,
                    std=std)


class MLPPrior(nn.Module):

    cfg: Config

    @nn.compact
    def __call__(self, z_t0):
        
        for _ in range(self.cfg.n_transfer_layers):
            z_t0 = activations[self.cfg.latent_activation](nn.Dense(self.cfg.n_embed)(z_t0))
     
        mean = nn.Dense(self.cfg.n_embed)(z_t0)
        std = jnn.softplus(nn.Dense(self.cfg.n_embed)(z_t0 + 0.54)) + self.cfg.latent_dist_min_std
        dist = tfd.Normal(mean, std)
        z_t1 = dist.sample(seed=self.make_rng('sample'))

        return dict(z=z_t1,
                    mean=mean, 
                    std=std)


