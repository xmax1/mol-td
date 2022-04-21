from errno import ESTALE
from pickle import FALSE
from flax import linen as nn
from jax import numpy as jnp, nn as jnn
from tensorflow_probability.substrates.jax import distributions as tfd
from functools import partial
from .config import Config


from typing import Callable
from jax import lax
from math import ceil, floor


activations = {'leaky_relu': partial(nn.leaky_relu, negative_slope=.2), 
               'relu': nn.relu}


class MLPEncoder(nn.Module):
    
    cfg: Config

    @nn.compact
    def __call__(self, x, training=False):
        # x input shape is n_atom, nt, n_atom, n_features
        bs, nt, n_atom, nf = x.shape
        x = x.reshape((bs, nt, n_atom * nf))

        for n_hidden in self.cfg.enc_hidden: # the first contains the feature dimension
            x = nn.Dense(n_hidden)(x)
            x = activations[self.cfg.map_activation](x)
            x = nn.Dropout(rate=self.cfg.dropout)(x, deterministic=not training)  # https://github.com/google/flax/issues/1004
            # x = nn.Dropout(rate=self.cfg.dropout, broadcast_dims=(0,))(x, deterministic=eval)  # rate is the dropout probability not the keep rate
        return x

    
class MLPDecoder(nn.Module):
    
    cfg: Config

    @nn.compact
    def __call__(self, z, training=False):
        bs, nt = z.shape[:2]
        for n_hidden in self.cfg.dec_hidden: # the first contains the feature dimension
            z = activations[self.cfg.map_activation](nn.Dense(n_hidden)(z))
            z = nn.Dropout(rate=self.cfg.dropout)(z, deterministic=not training)  # https://github.com/google/flax/issues/1004
        z = jnp.tanh(nn.Dense(self.cfg.dec_hidden[-1])(z))  # values in dataset restricted between 1 and -1
        z = z.reshape((bs, nt, self.cfg.n_atoms, 3))
        return z


class GNNEncoder(nn.Module):
    
    cfg: Config
    # adj = (jnp.eye((cfg.n_atoms, cfg.n_atoms)) * -1.) + 1.
    # d = 1./jnp.sqrt(jnp.sum(adj, keep_dims=True, axis=-1))
    # adj = d * adj * d
    # adj: d * adj * d

    @nn.compact
    def __call__(self, x, training=False):
        bs, nt, n_atoms, nf = x.shape
        print('GCN input: ', x.shape)

        adj = (jnp.eye(self.cfg.n_atoms) * -1.) + 1.
        d = 1./jnp.sqrt(jnp.sum(adj, keepdims=True, axis=-1))
        adj = d * adj * d

        for i, n_hidden in enumerate(self.cfg.enc_hidden): # the first contains the feature dimension
            neighbours = jnp.transpose(jnp.dot(adj, nn.Dense(x.shape[-1])(x)), (1, 2, 0, 3))
            x = (neighbours + nn.Dense(x.shape[-1])(x)) / 2. # dot is the sum product over axes -1 and -2 respectively
            x = activations[self.cfg.map_activation](x)
            # x = nn.Dropout(rate=self.cfg.dropout)(x, deterministic=training)  # rate is the dropout probability not the keep rate
            print(f'GCN layer_{i}:', x.shape)
        
        x = MLPEncoder(self.cfg)(x)
        # flatten
        # n_hidden = n_atoms * x.shape[-1]
        # x = x.reshape(bs, nt, n_hidden)
        # x = nn.Dense(self.cfg.n_embed)(x)
        # x = activations[self.cfg.map_activation](x)
        print('GCN output: ', x.shape)
        return x


class GNNDecoder(nn.Module):
    
    cfg: Config
    # adj: jnp.eye((cfg.n_atoms, cfg.n_atoms))
    # adj = (jnp.eye((cfg.n_atoms, cfg.n_atoms)) * -1.) + 1.
    # d = 1./jnp.sqrt(jnp.sum(adj, keep_dims=True, axis=-1))
    # adj: d * adj * d

    @nn.compact
    def __call__(self, x, eval=False):
        bs, nt, nf = x.shape[:3]

        adj = (jnp.eye(self.cfg.n_atoms) * -1.) + 1.
        d = 1./jnp.sqrt(jnp.sum(adj, keepdims=True, axis=-1))
        adj = d * adj * d

        n_hidden = max(2, ceil(nf / self.cfg.n_atoms))
        x = nn.Dense(self.cfg.n_atoms * n_hidden)(x)
        x = activations[self.cfg.map_activation](x)
        x = x.reshape(bs, nt, self.cfg.n_atoms, n_hidden)

        for n_hidden in self.cfg.enc_hidden: # the first contains the feature dimension
            # x = nn.Dense(n_hidden)(x)
            # x = jnp.dot(self.adj, x)  # dot is the sum product over axes -1 and -2 respectively
            neighbours = jnp.transpose(jnp.dot(adj, nn.Dense(n_hidden)(x)), (1, 2, 0, 3))
            x = (neighbours + nn.Dense(n_hidden)(x)) / 2. # dot is the sum product over axes -1 and -2 respectively
            x = activations[self.cfg.map_activation](x)

            

        x = jnp.tanh(nn.Dense(self.cfg.dec_hidden[-1])(x))  # values in dataset restricted between 1 and -1
        return x


# class GCNLayer(nn.Module):
#     features: int
#     kernel_init: Callable = nn.initializers.lecun_normal()
#     bias_init: Callable = nn.initializers.zeros

#     @nn.compact
#     def __call__(self, x):
#         w = self.param('kernel', self.kernel_init,  (x.shape[-1], self.features))
#         bias = self.param('bias', self.bias_init, (self.features,))
#         y = lax.dot_general(x, w, (((x.ndim - 1,), (0,)), ((), ())),) # TODO Why not jnp.dot?
#         y = y + bias
#         return y    

# key1, key2 = random.split(random.PRNGKey(0), 2)
# x = random.uniform(key1, (4,4))

# model = SimpleDense(features=3)
# params = model.init(key2, x)
# y = model.apply(params, x)

# print('initialized parameters:\n', params)
# print('output:\n', y)