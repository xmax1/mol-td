

import jax
from flax import nn
from jax import nn as jnn, numpy as jnp


class Encoder(nn.Module):
    def apply(self, x):
        x = nn.Dense(x, 400, name='enc_fc1')
        x = jnn.relu(x)
        mean_x = nn.Dense(x, 20, name='enc_fc21')
        logvar_x = nn.Dense(x, 20, name='enc_fc22')
        return mean_x, logvar_x


class Decoder(nn.Module):
    def apply(self, z):
        z = nn.Dense(z, 400, name='dec_fc1')
        z = jnn.relu(z)
        z = nn.Dense(z, 784, name='dec_fc2')
        z = jnn.sigmoid(z)
        return z

class VAE(nn.Module):
    
    def apply(self, x):
        mean, logvar = Encoder(x, name='encoder')
        z = reparameterize(mean, logvar)
        recon_x = Decoder(z, name='decoder')
        return recon_x, mean, logvar

    @nn.module_method
    def generate(self, z):
        params = self.get_param("decoder")
        return Decoder.call(params, z)


def reparameterize(mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = np.random.normal(size=logvar.shape)
    return mean + eps * std



@jax.vmap
def kl_divergence(mean, logvar):
    return - 0.5 * jnp.sum(1 + logvar - jnp.power(mean, 2) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy(probs, labels):
    return - jnp.sum(labels * jnp.log(probs + eps) + (1 - labels) * jnp.log(1 - probs + eps))


def compute_metrics(recon_x, x, mean, logvar):
    bce = binary_cross_entropy(recon_x, x)
    kld = kl_divergence(mean, logvar)
    return {'bce': jnp.mean(bce), 'kld': jnp.mean(kld), 'loss': jnp.mean(bce + kld)}