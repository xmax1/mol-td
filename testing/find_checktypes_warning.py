from flax import linen as nn
from jax import numpy as jnp, nn as jnn
from tensorflow_probability.substrates.jax import distributions as tfd
from functools import partial


leaky_relu = partial(nn.leaky_relu, negative_slope=.2)  # TF default


class MLPEncoder(nn.Module):
    
    cfg: dict

    @nn.compact
    def __call__(self, x):
        for n_hidden in self.cfg['enc_hidden']:
            x = jnn.relu(nn.Dense(n_hidden)(x))
        mu = nn.Dense(n_hidden)(x)
        sigma = jnp.sqrt(jnp.exp(nn.Dense(n_hidden)(x)))
        return mu, sigma

    
class MLPDecoder(nn.Module):
    
    cfg: dict

    @nn.compact
    def __call__(self, x):
        for n_hidden in self.cfg['dec_hidden'][:-1]:
            x = jnn.relu(nn.Dense(n_hidden)(x))
        x = jnn.sigmoid(nn.Dense(self.cfg['dec_hidden'][-1])(x))
        return x


def describe_distributions(distributions):
    print('\n'.join([str(d) for d in distributions]))


class SimpleVAE(nn.Module):

    cfg: dict

    def setup(self):
        self.encoder = MLPEncoder(self.cfg)
        self.decoder = MLPDecoder(self.cfg)

    def __call__(self, x):

        mu, sigma = self.encoder(x)
        latent_dist = tfd.Independent(tfd.Normal(mu, sigma))
        # latent_dist = tfd.MultivariateNormalDiag(mu, sigma)
        y = self.decoder(mu)
        output_dist = tfd.Independent(tfd.Normal(y, jnp.ones(y.shape)))  # 1 is the sigma, when is 1 goes to mse

        print('Dimensions: ', {'mu': mu.shape, 'sigma': sigma.shape, 'predicted': y.shape})
        
        nll_term = -jnp.mean(output_dist.log_prob(x), 0)
        prior = tfd.Independent(tfd.Normal(jnp.zeros(mu.shape), jnp.ones(mu.shape)))
        
        # describe_distributions([latent_dist, output_dist, prior])
        
        # kl_div = jnp.mean(tfd.kl_divergence(latent_dist, prior))
    
        # signal = dict(loss=nll_term + kl_div,
        #               kl_term=kl_div, 
        #               nll_term=nll_term,
        #               latent_mu=mu,
        #               latent_sigma=sigma)
        
        return y

from jax import random as rnd

cfg = {'MODEL':{'seed':1, 'enc_hidden':[11,12], 'dec_hidden':[12,10]}}
model = SimpleVAE(cfg['MODEL'])
batch_size = 10
rng, video_rng, params_rng, sample_rng = rnd.split(rnd.PRNGKey(cfg['MODEL']['seed']), 4)
ex_batch, ex_target = jnp.ones((batch_size, 10)), jnp.ones((batch_size, 10))
params = model.init(dict(params=params_rng, sample=sample_rng), ex_batch)

predicted = model.apply(params, ex_batch)

