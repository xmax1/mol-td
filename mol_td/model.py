from flax import linen as nn
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from functools import partial
from jax.tree_util import tree_map

from .model_encdec import *
from .model_transfer import *
from .config import Config

leaky_relu = partial(nn.leaky_relu, negative_slope=.2)  # TF default


def describe_distributions(distributions):
    print('\n'.join([str(d) for d in distributions]))


def mse(x, y):
    return jnp.mean((x-y)**2)


class SimpleTDVAE(nn.Module):

    cfg: Config

    def setup(self):
        
        self.encoder = MLPEncoder(self.cfg)
        self.transfer = MLPTransfer(self.cfg)
        self.decoder = MLPDecoder(self.cfg)

    def __call__(self, data, test=False):
        n_data, nt = data.shape[:2]

        posteriors = self.encoder(data)

        scan = nn.scan(lambda f, carry, inputs: f(carry, inputs),
                        variable_broadcast='params',
                        split_rngs=dict(params=False, sample=True),
                        in_axes=1, out_axes=1,
                        length=data.shape[1])

        initial = self.transfer.zero_state((n_data,)) if test is False else {k:v[:, 0, :] for k, v in posteriors.items()}
        
        _, (priors,) = scan(self.transfer, initial, (posteriors,))

        for k, v in priors.items():
            print(v.shape)
            
        decode = posteriors['mean'] if test is False else priors['mean']
        print(decode.shape)
        predictions = self.decoder(decode)

        exit()
        
        # print('Dimensions: ', {'mean': mean.shape, 'std': std.shape, 'predicted': y.shape})

        posteriors = tfd.Independent(tfd.Normal(posteriors['mean'], posteriors['std']))
        priors = tfd.Independent(tfd.Normal(priors['mean'], priors['std']))
        output_dist = tfd.Independent(tfd.Normal(predictions, self.cfg.prediction_std)) 
        # describe_distributions([latent_dist, output_dist, prior])
        
        data_target = data[..., :-self.cfg.n_atoms]
        nll = -jnp.mean(output_dist.log_prob(data_target))
        kl_div = jnp.mean(posteriors.kl_divergence(priors))  # mean over the batch dimension
        loss = nll + kl_div
        predictions_mse = mse(data_target, predictions)

        signal = dict(loss=loss,
                      kl_div=kl_div, 
                      nll=nll,
                      priors=priors,
                      posteriors=posteriors,
                      predictions=predictions,
                      predictions_mse=predictions_mse)

        return loss, signal




class SimpleVAE(nn.Module):

    cfg: Config

    def setup(self):

        self.encoder = MLPEncoder(self.cfg)
        self.decoder = MLPDecoder(self.cfg)

    def __call__(self, x):

        posterior = self.encoder(x)
        mean, std = posterior['mean'], posterior['std']
        
        y = self.decoder(mean)
        
        print('Dimensions: ', {'mean': mean.shape, 'std': std.shape, 'predicted': y.shape})

        latent_dist = tfd.Independent(tfd.Normal(mean, std))
        prior = tfd.Independent(tfd.Normal(jnp.zeros(mean.shape), jnp.ones(mean.shape)))
        output_dist = tfd.Independent(tfd.Normal(y, jnp.ones(y.shape)))  # 1 is the std, when is 1 goes to mse
        # describe_distributions([latent_dist, output_dist, prior])
        
        nll = -jnp.mean(output_dist.log_prob(x), 0)
        kl_div = jnp.mean(latent_dist.kl_divergence(prior))
        loss = nll + kl_div

        signal = dict(loss=loss,
                      kl_div=kl_div, 
                      nll=nll,
                      latent_mean=mean,
                      latent_std=std,
                      prediction=y)
        
        return loss, signal


