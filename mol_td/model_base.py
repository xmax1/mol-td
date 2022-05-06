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

import jraph 
from typing import Sequence
from jax import numpy as jnp
from jraph._src import utils
from flax import linen as nn


activations = {'leaky_relu': partial(nn.leaky_relu, negative_slope=.2), 
               'relu': nn.relu}


class ExplicitMLP(nn.Module):
  """A flax MLP."""
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate([nn.Dense(feat) for feat in self.features]):
      x = lyr(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
    return x

# Functions must be passed to jraph GNNs, but pytype does not recognise
# linen Modules as callables to here we wrap in a function.
def make_embed_fn(latent_size):
  def embed(inputs):
    return nn.Dense(latent_size)(inputs)
  return embed


def make_mlp(features):
  @jraph.concatenated_args
  def update_fn(inputs):
    return ExplicitMLP(features)(inputs)
  return update_fn


class GraphNetwork(nn.Module):
  """A flax GraphNetwork."""
  cfg: Config

  @nn.compact
  def __call__(self, graph, training=True):
    # Add a global parameter for graph classification.
    graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))

    if self.cfg.edge_features is not None:
      embed_edge_fn = make_embed_fn(self.cfg.graph_latent_size)
      update_edge_fn = make_mlp(self.cfg.graph_mlp_features)
    else:
      embed_edge_fn = None
      update_edge_fn = None
    
    embedder = jraph.GraphMapFeatures(
        embed_node_fn=make_embed_fn(self.cfg.graph_latent_size),
        embed_edge_fn=embed_edge_fn,)
        # embed_global_fn=make_embed_fn(self.latent_size))
    
    net = jraph.GraphNetwork(
        update_node_fn=make_mlp(self.cfg.graph_mlp_features),
        update_edge_fn=update_edge_fn)
        # aggregate_edges_for_globals_fn=utils.segment_mean)
        # The global update outputs size 2 for binary classification.
        # update_global_fn=make_mlp(self.mlp_features + (2,)))  # pytype: disable=unsupported-operands
    
    h = net(embedder(graph))
    n_node = h.n_node[0]
    bsnt = len(h.n_node)
    h = h.nodes
    nt = bsnt // self.cfg.batch_size
    h = h.reshape(self.cfg.batch_size, nt, self.cfg.n_nodes, h.shape[-1])

    h = MLPEncoder(self.cfg)(h)

    return h


class MLPEncoder(nn.Module):
    
    cfg: Config

    @nn.compact
    def __call__(self, x, training=False):
        # x input shape is n_atom, nt, n_atom, n_features
        bs, nt, n_atom, nf = x.shape
        x = x.reshape((bs, nt, n_atom * nf))

        for n_hidden in self.cfg.enc_hidden: # the first contains the feature dimension
            x = activations[self.cfg.map_activation](nn.Dense(n_hidden)(x))
            x = nn.Dropout(rate=self.cfg.dropout)(x, deterministic=not training)  # https://github.com/google/flax/issues/1004
            # x = nn.Dropout(rate=self.cfg.dropout, broadcast_dims=(0,))(x, deterministic=eval)  # rate is the dropout probability not the keep rate
        return x

    
class MLPDecoder(nn.Module):
    
    cfg: Config

    @nn.compact
    def __call__(self, z, training=False, predict_sigma=False):
        bs, nt = z.shape[:2]
        for n_hidden in self.cfg.dec_hidden: # the first contains the feature dimension
            z = activations[self.cfg.map_activation](nn.Dense(n_hidden)(z))
            z = nn.Dropout(rate=self.cfg.dropout)(z, deterministic=not training)  # https://github.com/google/flax/issues/1004
        
        mean = nn.Dense(self.cfg.dec_hidden[-1])(z).reshape((bs, nt, -1, self.cfg.n_dim))
        if self.cfg.periodic:
            z = jnn.sigmoid(mean)
        else:
            z = jnp.tanh(mean)
        if predict_sigma:
            std = jnn.softplus(nn.Dense(1)(z + 0.54)) + self.cfg.latent_dist_min_std
            std = std.reshape((bs, nt, 1, 1))
            return z, std
        else:
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