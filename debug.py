from mol_td.config import Config
from mol_td.data_fns import create_dataloaders
import jraph 
from typing import Sequence
from jax import numpy as jnp
from jraph._src import utils
from flax import linen as nn

cfg = Config(r_cutoff=100., batch_size=16)

tr_loader, val_loader, test_loader = create_dataloaders(cfg)

graph, target = next(tr_loader)

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
  mlp_features: Sequence[int]
  latent_size: int

  @nn.compact
  def __call__(self, graph):
    # Add a global parameter for graph classification.
    graph = graph._replace(globals=jnp.zeros([graph.n_node.shape[0], 1]))

    embedder = jraph.GraphMapFeatures(
        embed_node_fn=make_embed_fn(self.latent_size),
        embed_edge_fn=make_embed_fn(self.latent_size),)
        # embed_global_fn=make_embed_fn(self.latent_size))
    
    net = jraph.GraphNetwork(
        update_node_fn=make_mlp(self.mlp_features),
        update_edge_fn=make_mlp(self.mlp_features),
        aggregate_edges_for_globals_fn=utils.segment_mean)
        # The global update outputs size 2 for binary classification.
        # update_global_fn=make_mlp(self.mlp_features + (2,)))  # pytype: disable=unsupported-operands
    return net(embedder(graph))

net = GraphNetwork(mlp_features=(cfg.graph_mlp_features, cfg.graph_mlp_features), 
                   latent_size=cfg.graph_mlp_features)


net = GraphNetwork(mlp_features=(14, 14), latent_size=14)

import jax

# Initialize the network.
params = net.init(jax.random.PRNGKey(42), graph)

out = net.apply(params, graph)

print(out.nodes.shape)

print(out.nodes.reshape(cfg.batch_size, cfg.n_timesteps, -1).shape)
