from flax import linen as nn
from jax import numpy as jnp, partial, nn as jnn
from tensorflow_probability.substrates.jax import distributions as tfd

leaky_relu = partial(nn.leaky_relu, negative_slope=.2)  # TF default

class MLPEncoder(nn.Module):
    
    cfg: dict

    @nn.compact
    def __call__(self, x):
        for n_hidden in self.cfg['enc_hidden']:
            x = leaky_relu(nn.Dense(n_hidden)(x))
        mu = nn.Dense(n_hidden)(x)
        sigma = nn.Dense(n_hidden)(x)
        return mu, sigma

    
class MLPDecoder(nn.Module):
    
    cfg: dict

    @nn.compact
    def __call__(self, x):
        for n_hidden in self.cfg['dec_hidden'][:-1]:
            x = leaky_relu(nn.Dense(n_hidden)(x))
        x = jnn.sigmoid(nn.Dense(n_hidden)(x))
        return x


class SimpleVAE(nn.Module):

    cfg: dict

    def setup(self):
        self.encoder = MLPEncoder(self.cfg)
        self.decoder = MLPDecoder(self.cfg)

    def __call__(self, x):
        mu, sigma = self.encoder(x)
        latent_dist = tfd.Independent(tfd.Normal(mu, sigma))
        y = self.decoder(mu)
        output_dist = tfd.Independent(tfd.Normal(y, 1))  # 1 is the sigma, when is 1 goes to mse

        nll_term = -jnp.mean(output_dist.log_prob(x), 0)
        prior = tfd.Independent(tfd.Normal(0, 1))
        kl_div = jnp.mean(latent_dist.kl_divergence(prior))
    
        signal = dict(loss=nll_term + kl_div,
                      kl_term=kl_div, 
                      nll_term=nll_term)
        
        return y, signal


# class Encoder(nn.Module):
#     """
#     Multi-level Video Encoder.
#     1. Extracts hierarchical features from a sequence of observations.
#     2. Encodes observations using Conv layers, uses them directly for the bottom-most level.
#     3. Uses dense features for each level of the hierarchy above the bottom-most level.
#     """
#     c: Config

#     @nn.compact
#     def __call__(self, obs):
#         """
#         Arguments:
#             obs : Tensor
#                 Un-flattened observations (videos) of shape (batch size, timesteps, height, width, channels)
#         """
#         # Merge batch and time dimensions.
#         x = obs.reshape((-1,) + obs.shape[2:])

#         Conv = partial(nn.Conv, kernel_size=(4, 4), strides=(2, 2), padding='VALID')
#         x = leaky_relu(Conv(self.c.total_filters)(x))
#         x = leaky_relu(Conv(self.c.total_filters * 2)(x))
#         x = leaky_relu(Conv(self.c.total_filters * 4)(x))
#         x = leaky_relu(Conv(self.c.total_filters * 8)(x))
#         x = x.reshape(obs.shape[:2] + (-1,))
#         layers = [x]
#         print(f"Input shape at level 0: {x.shape}")

#         feat_size = x.shape[-1]

#         for level in range(1, self.c.levels):
#             for _ in range(self.c.enc_dense_layers - 1):
#                 x = nn.relu(nn.Dense(self.c.enc_dense_embed_size)(x))
#             if self.c.enc_dense_layers > 0:
#                 x = nn.Dense(feat_size)(x)
#             layer = x
#             timesteps_to_merge = self.c.tmp_abs_factor ** level
#             # Padding the time dimension.
#             timesteps_to_pad = -layer.shape[1] % timesteps_to_merge
#             layer = jnp.pad(layer, ((0, 0), (0, timesteps_to_pad), (0, 0)))
#             # Reshaping and merging in time.
#             layer = layer.reshape((layer.shape[0], -1, timesteps_to_merge,
#                                    layer.shape[2]))
#             layer = jnp.sum(layer, axis=2)
#             layers.append(layer)
#             print(f"Input shape at level {level}: {layer.shape}")

#         return layers


# class Decoder(nn.Module):
#     """ States to Images Decoder."""
#     c: Config

#     @nn.compact
#     def __call__(self, bottom_layer_output):
#         """
#         Arguments:
#             bottom_layer_output : Tensor
#                 State tensor of shape (batch_size, timesteps, feature_dim)

#         Returns:
#             Output video of shape (batch_size, timesteps, 64, 64, out_channels)
#         """
#         x = nn.Dense(self.c.channels_mult * 1024)(bottom_layer_output)  # (B, T, 1024)
#         # Merge batch and time dimensions, expand two (spatial) dims.
#         x = jnp.reshape(x, (-1, 1, 1, x.shape[-1]))  # (BxT, 1, 1, 1024)

#         ConvT = partial(nn.ConvTranspose, strides=(2, 2), padding='VALID')
#         x = leaky_relu(ConvT(self.c.total_filters * 4, (5, 5))(x))  # (BxT, 5, 5, 128)
#         x = leaky_relu(ConvT(self.c.total_filters * 2, (5, 5))(x))  # (BxT, 13, 13, 64)
#         x = leaky_relu(ConvT(self.c.total_filters, (6, 6))(x))  # (BxT, 30, 30, 32)
#         x = nn.tanh(ConvT(self.c.channels, (6, 6))(x))  # (BxT, 64, 64, C)
#         return x.reshape(bottom_layer_output.shape[:2] + x.shape[1:])  # (B, T, 64, 64, C)
